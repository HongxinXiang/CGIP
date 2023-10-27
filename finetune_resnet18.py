import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from model.resnet import load_model
from model.train.cnn_model_utils import train_one_epoch_multitask, evaluate_on_multitask
from model.train.dual_model_utils import load_pretrained_component, save_finetune_ckpt
from model.train.train_utils import fix_train_random_seed
from utils.public_utils import is_left_better_right, get_tqdm_desc
from utils.splitter import get_split_data


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of CGIP-ResNet18')

    # basic
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='path to data root')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--epochs', type=int, default=151, help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"], help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./experiments/finetune/image/', help='path to log')

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    fix_train_random_seed(args.runseed)  # fix random seeds

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")

    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}; valid_select: {}".format(eval_metric, valid_select))

    ##################################### load data #####################################
    if args.image_aug:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                           transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                           transforms.ToTensor()]
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    train_idx, val_idx, test_idx = get_split_data(args.dataset, args.dataroot)
    name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize, args=args)
    val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args)
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

    ##################################### load model #####################################
    model = load_model("ResNet18", num_classes=num_tasks)
    load_flag, desc = load_pretrained_component(model, args.resume, model_key="model_state_dict1", consistency=False)
    if load_flag:
        print(desc)

    print(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.weight_decay,
    )
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    results = {'highest_valid': min_value, 'final_train': min_value, 'final_test': min_value,
               'highest_train': min_value}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                                  device=device, epoch=epoch, task_type=args.task_type, tqdm_desc=tqdm_train_desc)
        # evaluate
        train_loss, train_results = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                          criterion=criterion, device=device, epoch=epoch,
                                                          task_type=args.task_type,
                                                          tqdm_desc=tqdm_eval_train_desc, type="train")
        val_loss, val_results = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                      criterion=criterion, device=device, epoch=epoch,
                                                      task_type=args.task_type,
                                                      tqdm_desc=tqdm_eval_val_desc, type="valid")
        test_loss, test_results = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                        criterion=criterion, device=device, epoch=epoch,
                                                        task_type=args.task_type,
                                                        tqdm_desc=tqdm_eval_test_desc, type="test")

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"dataset": args.dataset, "epoch": epoch, "Loss": train_loss, 'Train': train_result, 'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("final results: {}\n".format(results))


if __name__ == "__main__":
    args = parse_args()
    main(args)