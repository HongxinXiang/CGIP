import numpy as np
import torch


def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj_attr = torch.zeros((node_num, node_num, data.edge_attr.shape[1]), dtype=data.edge_attr.dtype)
    adj[edge_index[0], edge_index[1]] = 1
    adj_attr[edge_index[0], edge_index[1]] = data.edge_attr
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    adj_attr = adj_attr[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero(as_tuple=False).t()
    edge_attr = adj_attr[edge_index[0], edge_index[1]]

    try:
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def permute_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    adj_attr = torch.zeros((node_num, node_num, data.edge_attr.shape[1]), dtype=data.edge_attr.dtype)
    adj_attr[edge_index[0], edge_index[1]] = data.edge_attr

    idx_add = np.random.choice(node_num, (2, permute_num))

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    edge_attr = adj_attr[edge_index[0], edge_index[1]]

    data.edge_index = torch.tensor(edge_index)
    data.edge_attr = edge_attr

    return data


def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj_attr = torch.zeros((node_num, node_num, data.edge_attr.shape[1]), dtype=data.edge_attr.dtype)
    adj[edge_index[0], edge_index[1]] = 1
    adj_attr[edge_index[0], edge_index[1]] = data.edge_attr
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    adj_attr = adj_attr[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero(as_tuple=False).t()
    edge_attr = adj_attr[edge_index[0], edge_index[1]]

    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.float().mean(dim=0).round().numpy()  # set mean if atom is masked.
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=data.x.dtype)

    return data