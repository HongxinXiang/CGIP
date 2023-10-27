import os
from itertools import repeat

import numpy as np
import pandas as pd
import torch
from PIL import Image
from numpy import random
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from dataloader.augmentation.graph_augment import check_support_aug_type
from dataloader.augmentation.graph_augment import graphCLAug
from dataloader.graph_dataloader import create_graph_from_smiles
from dataloader.image_dataloader import Smiles2Img


def check_processed_dataset(dataroot, dataset):
    processed_path = os.path.join(dataroot, dataset, "processed")

    processed_file_path = os.path.join(processed_path, "{}_processed_ac.csv".format(dataset))
    graph_file = os.path.join(processed_path, "geometric_data_processed.pt")
    image_folder = os.path.join(processed_path, "224")

    if not (os.path.exists(processed_file_path) and os.path.exists(graph_file) and os.path.exists(image_folder)):
        return False

    # check file processed_ac.csv
    df = pd.read_csv(processed_file_path)
    cols = df.columns
    if not ("smiles" in cols and "index" in cols and "label" in cols):
        return False
    index = df["index"].values.astype(str).tolist()

    # check geometric_data_processed.pt and image folder
    graph_data = torch.load(graph_file)
    graph_index = [str(item) for item in graph_data[0].index]
    image_index = [str(os.path.splitext(item)[0]) for item in os.listdir(image_folder)]

    if not len(index) == len(graph_index) == len(image_index):
        return False
    if len(set(index) - set(graph_index)) != 0 or len(set(index) - set(image_index)) != 0:
        return False
    return True

def initialize_exp_dataset(dataroot, dataset, task_type="classification", pre_filter=None, pre_transform=None):
    if check_processed_dataset(dataroot, dataset):
        print("The required file already exists, use existing data.")
        return
    raw_file_path = os.path.join(dataroot, dataset, "raw", "{}.csv".format(dataset))
    save_root = os.path.join(dataroot, dataset, "processed")
    img_save_root = os.path.join(dataroot, dataset, "processed", "224")
    graph_save_path = os.path.join(save_root, "geometric_data_processed.pt")
    algn_save_path = os.path.join(save_root, "{}_processed_ac.csv".format(dataset))
    error_save_path = os.path.join(save_root, "error_smiles.csv")
    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path)
    index, smiles, label = df["index"].values, df["smiles"].values, get_label_from_align_data(df["label"], task_type=task_type)
    graph_list = []
    processed_ac_data = []
    error_smiles = []
    for i, s, l in tqdm(zip(index, smiles, label), total=len(index)):
        graph = create_graph_from_smiles(s, l, i, pre_filter=pre_filter, pre_transform=pre_transform, task_type=task_type)
        if graph is None:
            error_smiles.append(s)
            continue
        img = Smiles2Img(s)
        if img is None:
            error_smiles.append(s)
            continue
        # can save
        img.save(os.path.join(img_save_root, "{}.png".format(i)))
        graph_list.append(graph)
        processed_ac_data.append([i, s, " ".join([str(item) for item in l])])
    data, slices = InMemoryDataset.collate(graph_list)
    torch.save((data, slices), graph_save_path)
    processed_ac_data = np.array(processed_ac_data)
    pd.DataFrame({
        "index": processed_ac_data[:, 0],
        "smiles": processed_ac_data[:, 1],
        "label": processed_ac_data[:, 2],
    }).to_csv(algn_save_path, index=False)

    if len(error_smiles) > 0:
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)
    print("process completed !")


def load_dual_data_files(dataroot, dataset):
    processed_root = os.path.join(dataroot, dataset, "processed")
    img_folder = os.path.join(processed_root, "224")
    graph_path = os.path.join(processed_root, "geometric_data_processed.pt")
    align_path = os.path.join(processed_root, "{}_processed_ac.csv".format(dataset))
    return img_folder, graph_path, align_path


def get_label_from_align_data(label_series, task_type="classification"):
    '''e.g. get_label_from_align_data(df["label"])'''
    if task_type == "classification":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())
    elif task_type == "regression":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(float).tolist()).tolist())
    else:
        raise UserWarning("{} is undefined.".format(task_type))


def load_dual_align_data(dataroot, dataset, task_type="classification", verbose=False):
    if not check_processed_dataset(dataroot, dataset):
        raise ValueError("{}/{}: data is not completed.".format(dataroot, dataset))
    if verbose:
        print("checking processed dataset completed. ")

    img_folder, graph_path, align_path = load_dual_data_files(dataroot, dataset)

    df = pd.read_csv(align_path)
    index = df["index"].astype(str).tolist()
    if verbose:
        print("reading align data completed. ")

    # load graph data
    graph_data, graph_slices = torch.load(graph_path)
    graph_index = np.array(graph_data.index).astype(str).tolist()
    assert (np.array(graph_index) == np.array(index)).sum() == len(index), "index from graph  and index from csv file is inconsistent"

    new_index = df["index"].astype(str).values
    label = get_label_from_align_data(df["label"], task_type=task_type)
    image_path = (img_folder + "/" + new_index + ".png").tolist()

    return {
        "index": new_index,
        "label": label,
        "graph_data": graph_data,
        "graph_slices": graph_slices,
        "image_path": image_path
    }


class DualCollater(object):
    def __init__(self, follow_batch, multigpu=False):
        self.follow_batch = follow_batch
        self.multigpu = multigpu

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            if self.multigpu:
                return batch
            else:
                return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DualDataSet(Dataset):
    def __init__(self,
                 dataroot, dataset,
                 img_transformer, img_normalize=None,
                 graph_aug="none", graph_aug_ratio=0,
                 verbose=False, args=None):
        assert graph_aug in ["none", "dropN", "permE", "subgraph", "maskN", "random2", "random3", "random4"] or check_support_aug_type(graph_aug), "Unknown graph_aug parameter: {}".format(graph_aug)
        assert len(img_transformer) == 2

        self.args = args
        self.data_dict = load_dual_align_data(dataroot, dataset, verbose=verbose)
        self.total = len(self.data_dict["index"])
        self.img_normalize = img_normalize
        self._image_transformer_no_aug = img_transformer[0]
        self._image_transformer = img_transformer[1]
        self.graph_aug = graph_aug
        self.graph_aug_ratio = graph_aug_ratio
        self.num_tasks = self.data_dict["label"].shape[1]
        if args is not None:
            random.seed(args.seed)

    def get_image(self, index):
        ir = random.randint(2)
        path = self.data_dict["image_path"][index]
        img = Image.open(path).convert('RGB')
        if ir == 0:
            data = self._image_transformer(img)
        else:
            data = self._image_transformer_no_aug(img)
        if self.img_normalize is not None:
            data = self.img_normalize(data)
        return data

    def get_graph(self, index):
        data = self.data_dict["graph_data"].__class__()

        if hasattr(self.data_dict["graph_data"], '__num_nodes__'):
            data.num_nodes = self.data_dict["graph_data"].__num_nodes__[index]

        for key in self.data_dict["graph_data"].keys:
            item, slices = self.data_dict["graph_data"][key], self.data_dict["graph_slices"][key]
            start, end = slices[index].item(), slices[index + 1].item()
            # print(slices[index], slices[index + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data_dict["graph_data"].__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        if self.graph_aug != "none":
            return graphCLAug(data, self.graph_aug, self.graph_aug_ratio)
        return data

    def __getitem__(self, item_index):
        img_data = self.get_image(item_index)
        graph_data = self.get_graph(item_index)
        label = self.data_dict["label"][item_index]
        return img_data, graph_data, label

    def get_batch_by_item_index(self, batch_item_index):
        img_data = []
        graph_data = []
        for item_index in batch_item_index:
            img = self.get_image(item_index)
            graph = self.get_graph(item_index)

            img_data.append(img)
            graph_data.append(graph)
        return img_data, graph_data

    def __len__(self):
        return self.total


class DualDataSetAug(Dataset):
    def __init__(self,
                 dataroot, dataset,
                 img_transformer, img_normalize=None,
                 graph_aug="none", graph_aug_ratio=0,
                 verbose=False, args=None):

        assert graph_aug in ["none", "dropN", "permE", "subgraph", "maskN", "random2", "random3", "random4"] or check_support_aug_type(graph_aug), "Unknown graph_aug parameter: {}".format(graph_aug)
        assert graph_aug != "none"

        self.args = args
        self.data_dict = load_dual_align_data(dataroot, dataset, verbose=verbose)
        self.total = len(self.data_dict["index"])
        self.img_normalize = img_normalize
        self._image_transformer_no_aug = img_transformer[0]
        self._image_transformer = img_transformer[1]
        self.graph_aug = graph_aug
        self.graph_aug_ratio = graph_aug_ratio
        self.num_tasks = self.data_dict["label"].shape[1]
        if args is not None:
            random.seed(args.seed)

    def get_image(self, index, is_aug=True):
        path = self.data_dict["image_path"][index]
        img = Image.open(path).convert('RGB')
        if is_aug:
            data = self._image_transformer(img)
        else:
            data = self._image_transformer_no_aug(img)
        if self.img_normalize is not None:
            data = self.img_normalize(data)
        return data

    def get_graph(self, index, is_aug=True):
        data = self.data_dict["graph_data"].__class__()

        if hasattr(self.data_dict["graph_data"], '__num_nodes__'):
            data.num_nodes = self.data_dict["graph_data"].__num_nodes__[index]

        for key in self.data_dict["graph_data"].keys:
            item, slices = self.data_dict["graph_data"][key], self.data_dict["graph_slices"][key]
            start, end = slices[index].item(), slices[index + 1].item()
            # print(slices[index], slices[index + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data_dict["graph_data"].__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        if is_aug:
            return graphCLAug(data, self.graph_aug, self.graph_aug_ratio)
        return data

    def __getitem__(self, item_index):
        img_ori = self.get_image(item_index, is_aug=False)
        img_aug = self.get_image(item_index, is_aug=True)
        graph_ori = self.get_graph(item_index, is_aug=False)
        graph_aug = self.get_graph(item_index, is_aug=True)
        label = self.data_dict["label"][item_index]
        return img_ori, img_aug, graph_ori, graph_aug, label

    def get_batch_by_item_index(self, batch_item_index):
        img_data = []
        graph_data = []
        for item_index in batch_item_index:
            img = self.get_image(item_index)
            graph = self.get_graph(item_index)

            img_data.append(img)
            graph_data.append(graph)
        return img_data, graph_data

    def __len__(self):
        return self.total