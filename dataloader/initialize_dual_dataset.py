import sys
sys.path.append("./")
import argparse

from dataloader.dual_dataloader import initialize_exp_dataset

parser = argparse.ArgumentParser(description='initializing dataset for CGIP')
parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, ...')
parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"])
args = parser.parse_args()

if __name__ == '__main__':
    initialize_exp_dataset(args.dataroot, args.dataset, args.task_type)