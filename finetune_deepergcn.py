import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch_geometric.data import DataLoader

from dataloader.graph_dataloader import GraphDataset
from model.config.config_deepergcn import parse_args
from model.deepergcn import DeeperGCN
from model.train.dual_model_utils import load_pretrained_component, save_finetune_ckpt
from model.train.graph_model_utils import train, evaluation
from model.train.train_utils import fix_train_random_seed
from utils.public_utils import is_left_better_right, get_tqdm_desc
from utils.splitter import get_split_data


def main(args):

    fix_train_random_seed(seed=args.runseed)  # run seed

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.model_save_path = os.path.join(args.log_dir, "model_ckpt")

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    filename_pre = 'BS_{}-LR_{}'.format(args.batch, args.lr)

    dataset = GraphDataset(root=os.path.join(args.dataroot, args.dataset), dataset=args.dataset, raw_dirname="processed", task_type=args.task_type)

    if args.task_type == "classification":
        dataset.eval_metric = "rocauc"  # 更多取值查看evaluator = Evaluator里面的eval_metric
        valid_select = "max"
        min_value = -np.inf
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8":
            dataset.eval_metric = "mae"
        else:
            dataset.eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
        criterion = torch.nn.MSELoss()
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(dataset.eval_metric))

    args.num_tasks = dataset.num_tasks
    print("args: {}\n".format(args))

    train_idx, val_idx, test_idx = get_split_data(args.dataset, args.dataroot)
    train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch, shuffle=True,
                              num_workers=args.workers)
    valid_loader = DataLoader(dataset[val_idx], batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch, shuffle=False,
                             num_workers=args.workers)
    print("num_train: {}".format(len(dataset[train_idx])))
    print("num_valid: {}".format(len(dataset[val_idx])))
    print("num_test: {}".format(len(dataset[test_idx])))

    model = DeeperGCN(args)
    model.set_output_type(1)
    # load pre-training parameters from checkpoint
    load_flag, desc = load_pretrained_component(model, args.resume, model_key="model_state_dict2", consistency=False)
    print(desc)

    model.graph_pred_linear = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(model.graph_pred_linear.in_features, 128)),
        ('leakyreLU', nn.LeakyReLU()),
        ('dropout', nn.Dropout(0.1)),
        ('linear2', nn.Linear(128, dataset.num_tasks))
    ]))
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value
               }

    early_stop = 0
    patience = 30
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)

        print('Training...')
        epoch_loss = train(model, criterion, device, train_loader, optimizer, dataset.task_type, grad_clip=args.grad_clip, tqdm_desc=tqdm_train_desc)

        print('Evaluating...')
        train_dict = evaluation(model, device, train_loader, task_type=args.task_type, tqdm_desc=tqdm_eval_train_desc)
        valid_dict = evaluation(model, device, valid_loader, task_type=args.task_type, tqdm_desc=tqdm_eval_val_desc)
        test_dict = evaluation(model, device, test_loader, task_type=args.task_type, tqdm_desc=tqdm_eval_test_desc)

        train_result = train_dict[dataset.eval_metric.upper()]
        valid_result = valid_dict[dataset.eval_metric.upper()]
        test_result = test_dict[dataset.eval_metric.upper()]

        print(str({'epoch': epoch, 'Train loss': epoch_loss, 'Train': train_result, 'Validation': valid_result, 'Test': test_result}))

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(epoch_loss, 4), epoch,
                                   args.model_save_path, filename_pre,
                                   lr_scheduler=None, result_dict=results)

            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("results: {}\n".format(results))


if __name__ == "__main__":
    args = parse_args()
    main(args)

