import os
from collections import OrderedDict

import torch
from torch import nn
from torch_geometric.data import DataLoader

from dataloader.graph_dataloader import GraphDataset
from model.config.config_deepergcn import parse_args
from model.deepergcn import DeeperGCN
from model.train.graph_model_utils import evaluation
from utils.splitter import get_split_data


def main(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.model_save_path = os.path.join(args.log_dir, "model_ckpt")

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = GraphDataset(root=os.path.join(args.dataroot, args.dataset), dataset=args.dataset,
                           raw_dirname="processed", task_type=args.task_type)

    if args.task_type == "classification":
        dataset.eval_metric = "rocauc"
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8":
            dataset.eval_metric = "mae"
        else:
            dataset.eval_metric = "rmse"
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(dataset.eval_metric))

    args.num_tasks = dataset.num_tasks

    test_idx = get_split_data(args.dataset, args.dataroot)[2].tolist()

    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch, shuffle=False,
                             num_workers=args.workers)

    model = DeeperGCN(args)
    model.set_output_type(1)

    model.graph_pred_linear = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(model.graph_pred_linear.in_features, 128)),
        ('leakyreLU', nn.LeakyReLU()),
        ('dropout', nn.Dropout(0.1)),
        ('linear2', nn.Linear(128, dataset.num_tasks))
    ]))
    model = model.to(device)

    # load pre-training parameters from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # evaluation
    test_dict = evaluation(model, device, test_loader, task_type=args.task_type, tqdm_desc="evaluation test dataset")

    test_result = test_dict[dataset.eval_metric.upper()]
    print("[test] {}: {:.1f}%".format(dataset.eval_metric, test_result * 100))


if __name__ == "__main__":
    args = parse_args()
    main(args)

