import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils import splitter


def mol_to_graph_data_obj(smiles_str):
    """
    using ogb to extract graph features

    :param smiles_str:
    :return:
    """
    graph_dict = smiles2graph(smiles_str)  # introduction of features: https://blog.csdn.net/qq_38862852/article/details/106312171
    edge_attr = torch.tensor(graph_dict["edge_feat"], dtype=torch.long)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    x = torch.tensor(graph_dict["node_feat"], dtype=torch.long)
    return x, edge_index, edge_attr


class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset, raw_dirname="raw", transform=None, pre_transform=None, pre_filter=None, smiles_filename=None,
                 task_type="classification"):
        assert task_type in ["classification", "regression"]
        self.dataset = dataset
        self.root = root
        self.raw_dirname = raw_dirname
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.smiles_filename = smiles_filename
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_tasks = self.data.y.shape[1]
        self.total = len(self)
        self.task_type = task_type

    @property
    def raw_file_names(self):
        #  A list of files in the raw_dir which needs to be found in order to skip the download.
        '''
        file format:
            index, smiles, label [support multi-label, using space to split]
            1    , ...   , 1 0 1
            2    , ...   , 0 0 1
            3    , ...   , 1 1 1
            ...
        :return:
        '''
        if self.smiles_filename is None:
            raw_file_name = os.path.join(self.raw_dir, "{}_processed_ac.csv".format(self.dataset))
        else:
            raw_file_name = os.path.join(self.raw_dir, self.smiles_filename)
        assert os.path.isfile(raw_file_name), "{} is not a file.".format(raw_file_name)
        return [os.path.split(raw_file_name)[1]]

    @property
    def smiles_filepath(self):
        if self.smiles_filename is None:
            raw_file_name = os.path.join(self.raw_dir, "{}_processed_ac.csv".format(self.dataset))
        else:
            raw_file_name = os.path.join(self.raw_dir, self.smiles_filename)
        return raw_file_name

    @property
    def raw_dir(self):
        return osp.join(self.root, self.raw_dirname)

    @property
    def processed_file_names(self):
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
        return "geometric_data_processed.pt"

    def download(self):
        # Downloads raw data into raw_dir.
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        # Processes raw data and saves it into the processed_dir.
        assert len(self.raw_file_names) == 1
        raw_file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_file_path)
        columns = df.columns.tolist()
        assert 'index' in columns and 'smiles' in columns and 'label' in columns
        index, smiles, label = df["index"].values, df["smiles"].values, self.get_label(df)
        # Read data into huge `Data` list.
        data_list = []
        success_data_smiles_list = []
        error_data_smiles_list = []
        for i, s, l in zip(index, smiles, label):
            try:  # error will be occurred when mol is None in mol.GetAtoms().
                graph_dict = smiles2graph(s)
                edge_attr = torch.tensor(graph_dict["edge_feat"], dtype=torch.long)
                edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
                x = torch.tensor(graph_dict["node_feat"], dtype=torch.long)
                y = torch.tensor(l, dtype=torch.long).view(1,-1)
                graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=i)
                data_list.append(graph)
                success_data_smiles_list.append(s)
            except:
                print(i, s, l)
                error_data_smiles_list.append(s)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        for type, data_smiles_list in [("success", success_data_smiles_list), ("error", error_data_smiles_list)]:
            data_smiles_series = pd.Series(data_smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir, '{}_smiles.csv'.format(type)), index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_label(self, df):
        return np.array(df["label"].apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())

    def get_idx_split(self, split_type="naive", frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False, seed=42, split_path=None):
        assert split_type in ["split_file", "naive", "stratified", "scaffold", "random_scaffold"]
        idx = list(range(self.total))

        if split_type == "split_file":
            assert split_path is not None
            train_idx, valid_idx, test_idx = splitter.split_train_val_test_idx_split_file(split_path=split_path, sort=sort)
            assert len(train_idx) + len(valid_idx) + len(test_idx) == self.total
        elif split_type == "naive":
            train_idx, valid_idx, test_idx = splitter.split_train_val_test_idx(idx,
                                                                               frac_train=frac_train,
                                                                               frac_valid=frac_valid,
                                                                               frac_test=frac_test,
                                                                               sort=sort,
                                                                               seed=seed)
        elif split_type == "stratified":
            if self.num_tasks != 1:
                raise ValueError("multiple task is not supported by stratified split.")
            else:
                train_idx, valid_idx, test_idx = \
                    splitter.split_train_val_test_idx_stratified(idx,
                                                                 self.data.y.numpy().flatten(),
                                                                 frac_train=frac_train,
                                                                 frac_valid=frac_valid,
                                                                 frac_test=frac_test,
                                                                 sort=sort,
                                                                 seed=seed)
        elif split_type == "scaffold" or split_type == "random_scaffold":
            smiles_list = self.get_smiles_list()
            train_idx, valid_idx, test_idx = \
                splitter.scaffold_split_train_val_test(idx, smiles_list,
                                                       frac_train=frac_train,
                                                       frac_valid=frac_valid,
                                                       frac_test=frac_test,
                                                       sort=sort) if split_type == "scaffold" \
                    else splitter.random_scaffold_split_train_val_test(idx, smiles_list,
                                                                       frac_train=frac_train,
                                                                       frac_valid=frac_valid,
                                                                       frac_test=frac_test,
                                                                       sort=sort,
                                                                       seed=seed)
        else:
            raise ValueError("split_type: {} is not undefined. ".format(split_type))

        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def get_smiles_list(self):
        smiles_filepath = self.smiles_filepath
        if not os.path.exists(smiles_filepath):
            raise FileNotFoundError("smiles file is not exist: {} ".format(smiles_filepath))
        df = pd.read_csv(smiles_filepath)
        df["index"] = df["index"].astype("str")
        index_df = pd.DataFrame({"index": self.data.index}, dtype=str)  # current data
        merge_df = pd.merge(left=index_df, right=df, on="index")
        assert merge_df.shape[0] == index_df.shape[0], "error on merge(), please check your index_df and df."
        smiles_list = merge_df["smiles"].values.flatten().tolist()
        return smiles_list


def create_graph_data(raw_file_path, save_root, pre_filter=None, pre_transform=None):
    '''
    The structure of output file:
    * save_root
        * geometric_data_processed.pt
        * error_smiles.csv
        * success_smiles.csv
    :param raw_file_path:
    :param save_root:
    :param pre_filter:
    :param pre_transform:
    :return:
    '''
    # Processes raw data and saves it into the processed_dir.
    if not os.path.exists(save_root):
        print("create dir {}".format(save_root))
        os.makedirs(save_root)
    df = pd.read_csv(raw_file_path)
    columns = df.columns.tolist()
    assert 'index' in columns and 'smiles' in columns and 'label' in columns
    index, smiles, label = df["index"].values, df["smiles"].values, np.array(df["label"].apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())
    # Read data into huge `Data` list.
    data_list = []
    success_data_smiles_list = []
    error_data_idx_list = []
    error_data_smiles_list = []
    for i, s, l in zip(index, smiles, label):
        try:  # error will be occurred when mol is None in mol.GetAtoms().
            graph_dict = smiles2graph(s)
            edge_attr = torch.tensor(graph_dict["edge_feat"], dtype=torch.long)
            edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            x = torch.tensor(graph_dict["node_feat"], dtype=torch.long)
            y = torch.tensor(l, dtype=torch.long).view(1, -1)
            graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=i)
            data_list.append(graph)
            success_data_smiles_list.append(s)
        except:
            print(i, s, l)
            error_data_smiles_list.append(s)
            error_data_idx_list.append(i)

    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]

    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]

    # write data_smiles_list in processed paths
    for type, data_smiles_list in [("success", success_data_smiles_list), ("error", error_data_smiles_list)]:
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(save_root, '{}_smiles.csv'.format(type)), index=False,
                                  header=False)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(save_root, "geometric_data_processed.pt"))
    return error_data_smiles_list


def create_graph_from_smiles(smiles, label, index=None, pre_filter=None, pre_transform=None, task_type="classification"):
    assert task_type in ["classification", "regression"]
    try:
        x, edge_index, edge_attr = mol_to_graph_data_obj(smiles)
        if task_type == "classification":
            y = torch.tensor(label, dtype=torch.long).view(1, -1)
        else:
            y = torch.tensor(label, dtype=torch.float).view(1, -1)
        graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=index)
        if pre_filter is not None and pre_filter(graph):
            return None
        if pre_transform is not None:
            graph = pre_transform(graph)
        return graph
    except:
        return None