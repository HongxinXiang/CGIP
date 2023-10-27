import statistics

import torch
from tqdm import tqdm

from eval.evaluate import metric, metric_reg, metric_multitask, metric_reg_multitask


def train(model, criterion, device, loader, optimizer, task_type, grad_clip=0., tqdm_desc=""):
    loss_list = []
    accu_loss = torch.zeros(1).to(device)
    model.train()

    with tqdm(total=len(loader), desc=tqdm_desc) as t:
        for step, batch in enumerate(loader):
            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                batch = batch.to(device)

                optimizer.zero_grad()
                pred = model(batch)
                if "classification" in task_type:
                    is_labeled = batch.y != -1  # batch.y == batch.y,  -1 is null label
                    loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
                    loss = criterion(pred.to(torch.float32), batch.y.to(torch.float32))

                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()

                loss_list.append(loss.item())

                accu_loss += loss.detach()
                t.set_postfix(loss=accu_loss.item() / (step + 1))
                t.update(1)

    return statistics.mean(loss_list)


@torch.no_grad()
def evaluation(model, device, loader, task_type="classification", tqdm_desc=""):
    assert task_type in ["classification", "regression"]

    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc=tqdm_desc)):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_pred))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            metric_dict = metric(y_true, y_pred, y_pro, empty=-1)
            return metric_dict
        else:
            metric_dict = metric_reg(y_true, y_pred)
            return metric_dict
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_pred))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            metric_dict = metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
            return metric_dict
        else:
            metric_dict = metric_reg_multitask(y_true, y_pred, num_tasks=y_true.shape[1])
            return metric_dict
    else:
        raise Exception("error in the number of task.")

