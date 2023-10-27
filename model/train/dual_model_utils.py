import logging
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(branch1, branch2, optimizer1, optimizer2, data_loader, criterionI,
                                 device, epoch, graph_output=2, args=None):

    n_sub_ckpt_list_step = ((np.arange(1, args.n_sub_checkpoints_each_epoch + 1) / (args.n_sub_checkpoints_each_epoch + 1)) * len(data_loader)).astype(int)

    branch1.train()
    branch2.train()

    accu_loss = torch.zeros(1).to(device)
    accu_view1_I_loss = torch.zeros(1).to(device)
    accu_view2_I_loss = torch.zeros(1).to(device)
    accu_I_cross = torch.zeros(1).to(device)

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, data in enumerate(data_loader):
        view1_aug1, view1_aug2, view2_aug1, view2_aug2, _ = data

        if view1_aug1.shape[0] <= 1:
            continue
        if args.multigpu:
            view1_aug1, view1_aug2, view2_aug1, view2_aug2 = view1_aug1.to(device), view1_aug2.to(device), view2_aug1, view2_aug2
        else:
            view1_aug1, view1_aug2, view2_aug1, view2_aug2 = view1_aug1.to(device), view1_aug2.to(device), view2_aug1.to(device), view2_aug2.to(device)
        sample_num += view1_aug1.shape[0]

        X_v1_a1 = branch1(view1_aug1).reshape(view1_aug1.shape[0], -1)  # the space of view 1: aug 1
        X_v1_a2 = branch1(view1_aug2).reshape(view1_aug2.shape[0], -1)  # the space of view 1: aug 2

        if graph_output == 2:
            X_v2_a1, _ = branch2(view2_aug1)  # the space of view 2: aug 1
            X_v2_a2, _ = branch2(view2_aug2)  # the space of view 2: aug 2
        elif graph_output == 0 or graph_output == 1:
            X_v2_a1 = branch2(view2_aug1)  # the space of view 2: aug 1
            X_v2_a2 = branch2(view2_aug2)  # the space of view 2: aug 2

        # common-modality contrastive learning
        L_I1 = criterionI(X_v1_a1, X_v1_a2, labels=None)
        L_I2 = criterionI(X_v2_a1, X_v2_a2, labels=None)
        L_I_cross = criterionI(X_v1_a1, X_v2_a1, labels=None)

        loss = L_I1 + L_I2 + L_I_cross
        loss.backward()

        # logger
        accu_loss += loss.detach()
        accu_view1_I_loss += (L_I1).detach()
        accu_view2_I_loss += (L_I2).detach()
        accu_I_cross += L_I_cross.detach()

        data_loader.desc = "[train epoch {}] total loss: {:.3f}; " \
                           "view1 loss: {:.3f}; view2 loss: {:.3f}; cross loss: {:.3f}". \
            format(epoch, accu_loss.item() / (step + 1), accu_view1_I_loss.item() / (step + 1),
                   accu_view2_I_loss.item() / (step + 1), accu_I_cross.item() / (step + 1))

        if (epoch == 0 and step == 0) or step % (len(data_loader) // 10) == 0:
            msg = "=== [train epoch {}, step {}] total loss: {:.3f}; " \
                       "view1 loss: {:.3f}; view2 loss: {:.3f}; cross loss: {:.3f}". \
                format(epoch, step, accu_loss.item() / (step + 1), accu_view1_I_loss.item() / (step + 1),
               accu_view2_I_loss.item() / (step + 1), accu_I_cross.item() / (step + 1))
            print(msg)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if step % args.n_batch_step_optim == 0:
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

        if step != 0 and step in n_sub_ckpt_list_step:
            train_dict = {
                "step": step + len(data_loader) * epoch,
                "epoch": epoch + step / len(data_loader),
                "total_loss": accu_loss.item() / (step + 1),
                "view1_I_loss": accu_view1_I_loss.item() / (step + 1),
                "view2_I_loss": accu_view2_I_loss.item() / (step + 1),
                "cross_loss": accu_I_cross.item() / (step + 1),
            }
            ckpt_pre = "ckpt_epoch={}[{:.2f}%]_loss={:.2f}".format(epoch, step/len(data_loader)*100, train_dict["total_loss"])
            save_ckpt_common(branch1, branch2, optimizer1, optimizer2,
                             train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                             name_pre=ckpt_pre, name_post="")

    return {
        "step": step + len(data_loader) * epoch,
        "epoch": epoch + step / len(data_loader),
        "total_loss": accu_loss.item() / (step + 1),
        "view1_I_loss": accu_view1_I_loss.item() / (step + 1),
        "view2_I_loss": accu_view2_I_loss.item() / (step + 1),
        "cross_loss": accu_I_cross.item() / (step + 1),
    }


def save_ckpt_common(model1, model2, optimizer1, optimizer2,
                     loss, epoch, save_path, name_pre, name_post='best'):
    model1_cpu = {k: v.cpu() for k, v in model1.state_dict().items()}
    model2_cpu = {k: v.cpu() for k, v in model2.state_dict().items()}
    state = {
            'epoch': epoch,
            'model_state_dict1': model1_cpu,
            'model_state_dict2': model2_cpu,
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'loss': loss
        }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


def load_ckpt_common_space(pretrained_pth, model1, model2, optimizer1, optimizer2, load_optim_scheduler=True):
    pretrained_model = torch.load(pretrained_pth)
    loss_dict = pretrained_model["loss"]

    model_list = [("model1", model1, "model_state_dict1"), ("model2", model2, "model_state_dict2"),
                  ("optimizer1", optimizer1, "optimizer1_state_dict"),
                  ("optimizer2", optimizer2, "optimizer2_state_dict")]
    if load_optim_scheduler:
        for name, model, model_key in model_list:
            model_flag, _ = load_pretrained_component(model, pretrained_pth, model_key, consistency=False)
            assert model_flag, "error in loading {} from pretrained model.".format(name)
    else:
        for name, model, model_key in model_list[:4]:
            model_flag, _ = load_pretrained_component(model, pretrained_pth, model_key, consistency=False)
            assert model_flag, "error in loading {} from pretrained model.".format(name)
    return loss_dict


def load_pretrained_component(model, pretrained_pth, model_key, consistency=True, logger=None):
    log = logger if logger is not None else logging
    flag = False  # load successfully when only flag is true
    desc = None
    if pretrained_pth:
        if os.path.isfile(pretrained_pth):
            log.info("===> Loading checkpoint '{}'".format(pretrained_pth))
            checkpoint = torch.load(pretrained_pth)

            # load parameters
            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                model.load_state_dict(ckpt_model_state_dict)
                log.info("load all the parameters of pre-trianed model.")
            else:  # load parameter of layer-wise, resnet18 should load 120 layer at head.
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(model.state_dict())
                len_ckp_keys = len(ckp_keys)
                len_cur_keys = len(cur_keys)
                model_sd = model.state_dict()
                for idx in range(min(len_ckp_keys, len_cur_keys)):
                    ckp_key, cur_key = ckp_keys[idx], cur_keys[idx]
                    # print(ckp_key, cur_key)
                    model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                model.load_state_dict(model_sd)
                log.info("load the first {} parameters. layer number: model({}), pretrain({})"
                         .format(min(len_ckp_keys, len_cur_keys), len_cur_keys, len_ckp_keys))

            desc = "[resume model info] The pretrained_model is at checkpoint {}. \t Loss value: {}"\
                .format(checkpoint['epoch'], checkpoint["loss"])
            flag = True
        else:
            log.info("===> No checkpoint found at '{}'".format(pretrained_pth))
    else:
        log.info('===> No pre-trained model')
    return flag, desc


def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'loss': loss,
            'result_dict': result_dict
        }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


def write_result_dict_to_tb(tb_writer: SummaryWriter, result_dict: dict, optimizer_dict: dict, show_epoch=True):
    loop = result_dict["epoch"] if show_epoch else result_dict["step"]
    try:
        total_loss = result_dict["total_H_loss"] + result_dict["total_I_loss"]
        tb_writer.add_scalar("total_loss", total_loss, loop)
    except:
        pass
    for key in result_dict.keys():
        if key == "epoch" or key == "step":
            continue
        tb_writer.add_scalar(key, result_dict[key], loop)
    for key in optimizer_dict.keys():
        optimizer = optimizer_dict[key]
        tb_writer.add_scalar(key, optimizer.param_groups[0]["lr"], loop)

