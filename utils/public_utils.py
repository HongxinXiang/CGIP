import os
import csv
import shutil
import platform
import pickle


# device
def setup_device(n_gpu_use):
    import torch
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def save_best_result(list_of_dict, file_name, dir_path='best_result'):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Directory ", dir_path, " is created.")
    csv_file_name = '{}/{}.csv'.format(dir_path, file_name)
    with open(csv_file_name, 'a+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for _ in range(len(list_of_dict)):
            csv_writer.writerow(list_of_dict[_].values())


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def is_windows():
    sys = platform.system()
    if sys == "Windows":
           return True
    else:
        return False


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def is_left_better_right(left_num, right_num, standard):
    '''

    :param left_num:
    :param right_num:
    :param standard: if max, left_num > right_num is true, if min, left_num < right_num is true.
    :return:
    '''
    assert standard in ["max", "min"]
    if standard == "max":
        return left_num > right_num
    elif standard == "min":
        return left_num < right_num


def get_tqdm_desc(dataset, epoch):
    tqdm_train_desc = "[train] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_train_desc = "[eval on train set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_val_desc = "[eval on valid set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_test_desc = "[eval on test set] dataset: {}; epoch: {}".format(dataset, epoch)
    return tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc
