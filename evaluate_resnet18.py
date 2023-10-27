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
from model.train.cnn_model_utils import evaluate_on_multitask
from utils.splitter import get_split_data


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of evaluating CGIP-ResNet18')

    # basic
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='path to data root')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # train
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"], help='task type')

    return parser.parse_args()


def main(args):
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")

    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
    elif args.task_type == "regression":
        eval_metric = "rmse"
        valid_select = "min"
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}; valid_select: {}".format(eval_metric, valid_select))

    ##################################### load data #####################################
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    test_idx = get_split_data(args.dataset, args.dataroot)[2]
    name_test, labels_test = names[test_idx], labels[test_idx]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

    ##################################### load model #####################################
    model = load_model("ResNet18", num_classes=num_tasks)
    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = model.to(device=device)

    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    ##################################### evaluation #####################################
    test_loss, test_results = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                    criterion=criterion, device=device, epoch=-1,
                                                    task_type=args.task_type, type="test")
    test_result = test_results[eval_metric.upper()]

    print("[test] {}: {:.1f}%".format(eval_metric, test_result * 100))


if __name__ == "__main__":
    args = parse_args()
    main(args)