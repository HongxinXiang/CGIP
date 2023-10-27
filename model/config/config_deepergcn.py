import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of CGIP-DeeperGCN')
    # dataset
    parser.add_argument('--dataroot', type=str, default="dataroot", help='e.g. ./dataset/mpp/BBBP')
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name (default: bbbp)')
    parser.add_argument('--workers', type=int, default=2, help='number of workers (default: 2)')
    parser.add_argument('--batch', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--add_virtual_node', action='store_true')
    # training & eval settings
    parser.add_argument('--graph_aug', type=str, default="dropN+permE+subgraph+maskN",
                        help='e.g. none, dropN, permE, subgraph, maskN, random2, random3, random4, none+dropN+subgraph '
                             'or any other plus sign combination of none, dropN, permE, subgraph, maskN')
    parser.add_argument('--graph_aug_ratio', type=float, default=0.2,
                        help="select 20 percent of the local area in the graph for augmentation")
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 300)')
    parser.add_argument('--seed', type=int, default=42, help='random seed to split dataset (default: 42)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=0., help='gradient clipping')
    parser.add_argument('--split_path', type=str, help='e.g. ./dataset/mpp/BBBP/scaffold.npy')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # model
    parser.add_argument('--num_layers', type=int, default=14, help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=1, help='the number of layers of mlp in conv')
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--block', default='res+', type=str,
                        help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen', help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='batch', help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1, help='the number of prediction tasks')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')

    # learnable parameters
    parser.add_argument('--t', type=float, default=1.0, help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0, help='the power of PowerMean')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--y', type=float, default=0.0, help='the power of softmax_sum and powermean_sum')
    parser.add_argument('--learn_y', action='store_true')

    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # encode edge in conv
    parser.add_argument('--conv_encode_edge', action='store_true')
    # graph pooling type
    parser.add_argument('--graph_pooling', type=str, default='mean', help='graph pooling method')
    # save model
    parser.add_argument('--log_dir', default='./experiments/finetune/graph/', help='path to log')
    # load model
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='resume model')

    # task
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')

    # pretrain params (unused)
    parser.add_argument('--n_device', default=12, type=int, help='count of device')
    parser.add_argument('--gpu', default=None, type=str, help='index of GPU to use, e.g. 0,1,2,3')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--t_dropout', type=float, default=0.5, help='the dropout of deepergcn')
    parser.add_argument('--feat_dim', type=int, default=256, help='the dimension of topological space')
    parser.add_argument('--pretrained_pth', type=str, default=None,
                        help='read pre-trained model to continue pre-train model')
    parser.add_argument('--load_optim_scheduler', action='store_true', default=False,
                        help='whether to load optimizer and scheduler from pretrained model.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18",
                        help='e.g. ResNet18, ResNet34, LeViT, ResNetViT, ResNetViT_1')
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument('--lr_decay_epoch', type=int, default=10,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')
    parser.add_argument('--n_sub_checkpoints_each_epoch', type=int, default=4,
                        help='save the sub-checkpoints in an epoch, 0 represent this param is not active. e.g. n=4, will save epoch.2, epoch.4, epoch.6, epoch.8')

    args = parser.parse_args()

    return args
