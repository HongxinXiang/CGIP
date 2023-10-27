import warnings

import numpy as np

from dataloader.augmentation.utils_graph_aug import drop_nodes, permute_edges, subgraph, mask_nodes


def check_support_aug_type(aug):
    '''
    Checking support augmentation value. the value must meet the format: {aug1}+{aug2}+{aug3}, e.g. dropN+permE
    :return: True (success) or False (fail)
    '''
    aug_list = aug.split("+")
    for aug_item in aug_list:
        if aug_item not in ["none", "dropN", "permE", "subgraph", "maskN"]:
            warnings.warn("{} is not supported".format(aug_item))
            return False
    return True


def graphCLAug(data, aug, aug_ratio):
    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)
    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'subgraph':
        data = subgraph(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'none':
        data = data
    else:
        if not check_support_aug_type(aug):
            print('augmentation error')
            raise ValueError("Error aug parameter: {} ".format(aug))
        aug_list = aug.split("+")
        ri = np.random.randint(len(aug_list))
        if aug_list[ri] == 'dropN':
            data = drop_nodes(data, aug_ratio)
        elif aug_list[ri] == 'permE':
            data = permute_edges(data, aug_ratio)
        elif aug_list[ri] == 'subgraph':
            data = subgraph(data, aug_ratio)
        elif aug_list[ri] == 'maskN':
            data = mask_nodes(data, aug_ratio)
        elif aug_list[ri] == 'none':
            data = data
    return data

