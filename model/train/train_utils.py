import random
import torch
import numpy as np
import os


def fix_train_random_seed(seed=2021):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # single gpu
    torch.cuda.manual_seed_all(seed)  # multiple gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # no cudnn acceleration