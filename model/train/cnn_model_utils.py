import sys

import torch
from sklearn import metrics
from tqdm import tqdm

from eval.evaluate import metric as utils_evaluate_metric
from eval.evaluate import metric_multitask as utils_evaluate_metric_multitask
from eval.evaluate import metric_reg as utils_evaluate_metric_reg
from eval.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask


def metric(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "ROCAUC": auc
    }


def train_one_epoch_multitask(model, optimizer, data_loader, criterion, device, epoch, task_type, tqdm_desc=""):
    assert task_type in ["classification", "regression"]

    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion(pred.double(), labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification", tqdm_desc="", type="valid"):
    assert task_type in ["classification", "regression"]

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        with torch.no_grad():
            pred = model(images)
            labels = labels.view(pred.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "[evaluation epoch {}] {} loss: {:.3f}".format(epoch, type, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")
