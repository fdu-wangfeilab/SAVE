import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler


import sys
import os

from model.VAE_model import VAE
from model.utils.process_h5ad import convert_adata_to_dataloader


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


def kl_div_dis(mu1, var1, mu2, var2):
    return (
        kl_divergence(Normal(mu1, var1.sqrt()), Normal(mu2, var2.sqrt()))
        .sum(dim=1)
        .mean()
    )


def covariance_matrix(z1, z2):
    # 确保输入为2D [batch_size, dimension]
    assert z1.dim() == 2 and z2.dim() == 2
    # 计算样本均值
    z1_mean = z1.mean(dim=0, keepdim=True)
    z2_mean = z2.mean(dim=0, keepdim=True)

    # 计算中心化后的 z1 和 z2
    z1_centered = z1 - z1_mean
    z2_centered = z2 - z2_mean

    # 计算协方差矩阵
    cov_matrix = (z1_centered.T @ z2_centered) / (z1.size(0) - 1)
    # print(cov_matrix.shape)

    return cov_matrix


def independence_loss(z1, z2):
    cov_matrix = covariance_matrix(z1, z2)
    # 只保留协方差矩阵的非对角线元素
    off_diagonal_cov = cov_matrix - torch.diag(torch.diag(cov_matrix))

    # 最小化非对角线元素的平方和
    loss = torch.sum(off_diagonal_cov**2)

    return loss


def mi_caculate(c_pred, idx, num_class, group_num):
    O = torch.zeros((num_class, group_num)).cuda()
    for b in range(group_num):
        O[:, b] = torch.sum(c_pred[idx.squeeze() == b], dim=0)
    O[O <= 0] = torch.min(O[O > 0]) / 10
    pcg = O / torch.sum(O)
    pc = torch.sum(pcg, dim=1, keepdim=True)
    pg = torch.sum(pcg, dim=0, keepdim=True)
    return torch.sum(pcg * torch.log(pcg / (pc * pg)))


def vae_train(
    vae,
    dataloader,
    num_step,
    kl_scale=0.5,
    cov_scale=None,
    cls_scale=None,
    device=torch.device("cuda:0"),
    is_tensorboard=False,
    lr=2e-4,
    lr_milestone=1000,
    grad_clip=False,
    seed=1202,
    is_lr_scheduler=False,
    weight_decay=5e-4,
    capacity=15,
    capacity_milestone=1000,
    col_msk_threshold=0.8,
    loss_monitor=None,
    session=None,
    is_period_save=False,
    **kwargs,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if is_tensorboard:
        log_dir = "./model/log/attn"
        writer = SummaryWriter(log_dir=log_dir)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step):
        if step < lr_milestone:
            return step / lr_milestone
        else:
            return 1.0

    scheduler1 = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_step - lr_milestone, last_epoch=-1
    )

    scheduler = lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[lr_milestone],
    )
    capacity_schedule = np.concatenate(
        [
            np.linspace(0, 1, num_step - capacity_milestone),
            np.array([1] * capacity_milestone),
        ]
    )

    step = 1
    num_epoch = num_step // len(dataloader) + 1
    tq = tqdm(range(num_epoch), ncols=80)
    for _ in tq:
        vae.train()
        # epoch_loss = defaultdict(float)
        for _, (x, batch_idx) in enumerate(dataloader):
            x, idx = x.float().to(device), batch_idx.long().to(device)

            z, mu, var, z_dis, mu_dis, var_dis = vae.encoder(x)
            recon_x = vae.decoder(
                z,
                idx,
                col_msk_threshold=col_msk_threshold,
            )

            cls_loss = vae.q_net(z_dis, idx, F.cross_entropy)

            # using bce loss estimating the error
            recon_loss = F.binary_cross_entropy(recon_x, x) * x.size(-1)
            kl_loss = (
                torch.abs(kl_div(mu, var) - capacity * capacity_schedule[step - 1])
                * kl_scale
            )

            # cov_loss = independence_loss(z, z_dis) * cov_scale
            # mi_loss = mi_caculate(c_pred, idx, num_class, group_num) * mi_scale
            # kl_dis_loss = -kl_div_dis(mu, var, mu_dis, var_dis) * kl_scale * 0.01

            if loss_monitor is not None:
                loss_dict = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    # "cov_loss": cov_loss,
                    "mu": mu.mean(),
                    "var": var.mean(),
                    "capacity": capacity * capacity_schedule[step - 1],
                }
                loss_dict.update({k: v.item() * cls_scale for k, v in cls_loss.items()})
                loss_monitor.log(
                    loss_dict,
                )

            loss = {
                # "cov_loss": cov_loss,
                "cls_loss": sum(cls_loss.values()) * cls_scale,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
            }

            if session is not None:
                session.report({"loss": sum(loss.values()).item()})

            optimizer.zero_grad()
            sum(loss.values()).backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 and is_period_save:
                torch.save(
                    vae.state_dict(), "./ckpt/period/SAVE_period_{}.pt".format(step)
                )
            if step >= num_step:
                break

            step += 1

            # for k, v in loss.items():
            #     epoch_loss[k] += loss[k].item()

        # if is_lr_scheduler:

        # epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
        # epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
        # tq.set_postfix_str(epoch_info)

        # if is_tensorboard:
        #     writer.add_scalars(
        #         "train",
        #         {
        #             "kl_loss": epoch_loss["kl_loss"],
        #             "recon_loss": epoch_loss["recon_loss"],
        #         },
        #         epoch,
        #     )
    # for some config record
    return


def train_script(model_path, num_epoch, batch_size):
    model = VAE(
        input_dim=2000,
        hidden_dim=1024,
        enc_dim=10,
        enc_num_heads=8,
        dec_num_heads=8,
        enc_depth=4,
        dec_depth=2,
    )

    device = torch.device("cuda:2")
    model.to(device)
    print(model)

    # dataloader, atac_adata, rna_adata = get_vae_dataloader(
    # dir=pwd + 'data/pbmc/train',
    # batch_size = 128,
    # is_shuffle= True,
    # split_type='train',
    # seq=['atac','rna']
    # )

    adata = sc.read_h5ad("data/pbmc_RNA-ATAC.h5ad")
    dataloader = convert_adata_to_dataloader(
        adata=adata,
        batch_size=batch_size,
        is_shuffle=True,
    )

    model_save_path = "model/ckpt/" + model_path
    vae_train(vae=model, dataloader=dataloader, num_epoch=num_epoch, device=device)

    torch.save(model.state_dict(), model_save_path)
