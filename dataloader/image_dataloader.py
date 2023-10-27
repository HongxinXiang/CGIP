from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
from rdkit.Chem import Draw
from rdkit import Chem

class ImageDataset(Dataset):
    def __init__(self, filenames, labels, index=None, img_transformer=None, normalize=None, ret_index=False, args=None):
        '''
        :param names: image path, e.g. ["./data/1.png", "./data/2.png", ..., "./data/n.png"]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param normalize:
        :param args:
        '''

        self.args = args
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ret_index = ret_index
        if index is not None:
            self.index = index
        else:
            self.index = []
            for filename in filenames:
                self.index.append(os.path.splitext(os.path.split(filename)[1])[0])

    def get_image(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        data = self.get_image(index)
        if self.normalize is not None:
            data = self.normalize(data)
        if self.ret_index:
            return data, self.labels[index], self.index[index]
        else:
            return data, self.labels[index]

    def __len__(self):
        return self.total


def load_filenames_and_labels(image_folder, txt_file, task_type="classification"):
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = df.label.values.astype(int) if task_type == "classification" else df.label.values.astype(float)
    names = [os.path.join(image_folder, str(item)+".png") for item in index]
    return names, labels


def load_filenames_and_labels_multitask(image_folder, txt_file, task_type="classification"):
    '''
    multi-task version, label must be split by blank. e.g. binary classification, the label is ['0 1', '1 0', ...]
    :param image_folder:
    :param txt_file:
    :return:
    '''
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = [os.path.join(image_folder, str(item)+".png") for item in index]
    assert len(index) == labels.shape[0] == len(names)
    return names, labels

def load_filenames_labels_smiles(image_folder, txt_file, task_type="classification"):
    '''
    multi-task version, label must be split by blank. e.g. binary classification, the label is ['0 1', '1 0', ...]
    :param image_folder:
    :param txt_file:
    :return:
    '''
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    try:
        index = df["index"].values.astype(int)
    except:
        index = df["index"].values
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = [os.path.join(image_folder, str(item)+".png") for item in index]
    smiles = df["smiles"].tolist()
    assert len(index) == labels.shape[0] == len(names)
    return names, labels, smiles

def get_datasets(dataset, dataroot, data_type="raw"):
    '''
    目录结构必须满足：
    [dataroot] F:/data/生信/finetune数据集临时调试使用/SARS-CoV-2/
        * [datasetname] SARS-CoV_Pseudotyped_particle_entry_(VeroE6_tox_counterscreen)
            * [data_type] raw
                * 224
                    * 1.png
                    * 2.png
                    * ..
                * [datasetname_processed_ac.csv] SARS-CoV_Pseudotyped_particle_entry_(VeroE6_tox_counterscreen)_processed_ac.csv

    :param dataset:
    :param dataroot:
    :param data_type:
    :return:
    '''

    assert data_type in ["raw", "processed"]

    image_folder = os.path.join(dataroot, "{}/{}/224/".format(dataset, data_type))
    txt_file = os.path.join(dataroot, "{}/{}/{}_processed_ac.csv".format(dataset, data_type, dataset))

    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)

    return image_folder, txt_file


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    try:
        mol = Chem.MolFromSmiles(smis)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        if savePath is not None:
            img.save(savePath)
        return img
    except:
        return None

