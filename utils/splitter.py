import os

import numpy as np


def get_split_data(dataset, dataroot):
    npz_file = os.path.join(dataroot, "{}/processed/{}.npz".format(dataset, dataset))
    assert os.path.isfile(npz_file), "{} is not a file.".format(npz_file)
    data_split = np.load(npz_file, allow_pickle=True)
    return data_split["idx_train"], data_split["idx_val"], data_split["idx_test"]
