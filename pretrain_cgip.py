import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch_geometric
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.augmentation.image_augmentation import GaussianBlur
from dataloader.dual_dataloader import DualDataSetAug, DualCollater
from loss.losses import SupConLoss
from model.train.dual_model_utils import save_ckpt_common, write_result_dict_to_tb, load_ckpt_common_space
from model.train.dual_model_utils import train_one_epoch
from model.deepergcn import load_DeeperGCN
from model.resnet import load_model
from utils.public_utils import setup_device
from model.train.train_utils import fix_train_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of CGIP')

    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='path to data root')
    parser.add_argument('--n_device', default=12, type=int, help='count of device')
    parser.add_argument('--gpu', default=None, type=str, help='index of GPU to use, e.g. 0,1,2,3')
    parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')

    # model
    parser.add_argument('--num_layers', type=int, default=14, help='the num_layers of deepergcn')
    parser.add_argument('--t_dropout', type=float, default=0.5, help='the dropout of deepergcn')
    parser.add_argument('--feat_dim', type=int, default=256, help='the dimension of topological space')

    # resume
    parser.add_argument('--pretrained_pth', type=str, default=None, help='read pre-trained model to continue pre-train model')
    parser.add_argument('--load_optim_scheduler', action='store_true', default=False, help='whether to load optimizer and scheduler from pretrained model.')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34, LeViT, ResNetViT, ResNetViT_1')
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument('--lr_decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--graph_aug', type=str, default="dropN+permE+subgraph+maskN", help='e.g. none, dropN, permE, subgraph, maskN, random2, random3, random4, none+dropN+subgraph or any other plus sign combination of none, dropN, permE, subgraph, maskN')
    parser.add_argument('--graph_aug_ratio', type=float, default=0.2, help="select 20 percent of the local area in the graph for augmentation")
    parser.add_argument('--n_ckpt_save', default=1, type=int, help='save a checkpoint every n epochs, n_ckpt_save=0: no save')
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')
    parser.add_argument('--n_sub_checkpoints_each_epoch', type=int, default=4, help='save the sub-checkpoints in an epoch, 0 represent this param is not active. e.g. n=4, will save epoch.2, epoch.4, epoch.6, epoch.8')

    # log
    parser.add_argument('--log_dir', default='./experiments/pretrain/', help='path to log')

    return parser.parse_args()


def main(args):
    ############################################# installation
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print("using gpu: {}".format(args.gpu))
    fix_train_random_seed(args.seed)  # fix random seeds

    device, device_ids = setup_device(args.n_device)
    args.multigpu = True if len(device_ids) > 1 else False
    args.tb_dir = os.path.join(args.log_dir, "tbs")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print("run command: " + " ".join(sys.argv))
    print("log_dir: {}".format(args.log_dir))

    ############################################# load dataset
    print("load dataset")
    img_transformer = [transforms.CenterCrop(args.imageSize),
                       transforms.RandomApply([
                           transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                       ], p=0.8),
                       transforms.RandomGrayscale(p=0.2),
                       transforms.RandomRotation(degrees=360),
                       transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]
    img_transformer_no_aug = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dualDataset = DualDataSetAug(args.dataroot, args.dataset,
                                 img_transformer=[transforms.Compose(img_transformer_no_aug),
                                               transforms.Compose(img_transformer)],
                                 img_normalize=normalize,
                                 graph_aug=args.graph_aug, graph_aug_ratio=args.graph_aug_ratio, verbose=True,
                                 args=args)
    dataloader = DataLoader(dualDataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                                collate_fn=DualCollater(follow_batch=[], multigpu=args.multigpu))

    ############################################# load model and initializing training setting
    print("load model")
    image_branch = load_model(modelname=args.image_model)
    graph_branch = load_DeeperGCN(num_layers=args.num_layers, hidden_channels=512, dropout=args.t_dropout)
    graph_output = 2
    graph_branch.set_output_type(output=graph_output)
    image_branch = nn.Sequential(*list(image_branch.children())[:-1]).to(device)
    graph_branch = graph_branch.to(device)
    print(image_branch)
    print(graph_branch)

    # Optimizers and LR schedulers
    optimizer1 = torch.optim.Adam(filter(lambda x: x.requires_grad, image_branch.parameters()), lr=args.lr, weight_decay=10 ** args.weight_decay)
    optimizer2 = torch.optim.Adam(filter(lambda x: x.requires_grad, graph_branch.parameters()), lr=args.lr, weight_decay=10 ** args.weight_decay)

    # load pretrained checkpoint.
    if args.pretrained_pth is not None:
        print("load pretrained_pth from {}".format(args.pretrained_pth))
        loss_dict = load_ckpt_common_space(args.pretrained_pth, image_branch, graph_branch, optimizer1, optimizer2, load_optim_scheduler=args.load_optim_scheduler)
        args.start_epoch = int(loss_dict["epoch"] + 1)
        print(loss_dict)
    print("args: {}".format(args))

    criterionI = SupConLoss(temperature=args.temperature, base_temperature=args.base_temperature).to(device)  # 互信息

    if len(device_ids) > 1:
        image_branch = nn.DataParallel(image_branch, device_ids=device_ids)
        graph_branch = torch_geometric.nn.DataParallel(graph_branch, device_ids=device_ids)

    # initialize SummaryWriter from tensorboard.
    tb_writer = SummaryWriter(log_dir=args.tb_dir)
    optimizer_dict = {"optimizer1": optimizer1, "optimizer2": optimizer2}

    ############################################# start to train
    best_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_dict = train_one_epoch(
            branch1=image_branch, branch2=graph_branch, graph_output=graph_output,
            optimizer1=optimizer1, optimizer2=optimizer2, data_loader=dataloader,
            criterionI=criterionI, device=device, epoch=epoch,
            args=args)
        print(str(train_dict))

        cur_loss = train_dict["total_loss"]
        if best_loss > cur_loss:
            files2remove = glob.glob(os.path.join(args.log_dir, "ckpts", "best_epoch*"))
            for _i in files2remove:
                os.remove(_i)
            best_loss = cur_loss
            best_pre = "best_epoch={}_loss={:.2f}".format(epoch, best_loss)
            save_ckpt_common(image_branch, graph_branch, optimizer1, optimizer2,
                             train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                             name_pre=best_pre, name_post="")

        if args.n_ckpt_save > 0 and epoch % args.n_ckpt_save == 0:
            ckpt_pre = "ckpt_epoch={}_loss={:.2f}".format(epoch, cur_loss)
            save_ckpt_common(image_branch, graph_branch,
                             optimizer1, optimizer2,
                             train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                             name_pre=ckpt_pre, name_post="")

        write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)