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

pwd = "/home/lijiahao/workbench/sc-save/"
sys.path.append(pwd)
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


def vae_train(
    vae,
    dataloader,
    num_epoch,
    kl_scale=0.5,
    device=torch.device("cuda:0"),
    is_tensorboard=False,
    lr=2e-4,
    grad_clip=False,
    seed=1202,
    is_lr_scheduler=False,
    weight_decay=5e-4,
    col_msk_threshold=0.8,
    loss_monitor=None,
    session = None,
    **kwargs,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if is_tensorboard:
        log_dir = pwd + "model/log/attn"
        writer = SummaryWriter(log_dir=log_dir)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, last_epoch=-1
    )

    tq = tqdm(range(num_epoch), ncols=80)
    for epoch in tq:
        vae.train()
        epoch_loss = defaultdict(float)
        for i, (x, batch_idx) in enumerate(dataloader):
            x, idx = x.float().to(device), batch_idx.long().to(device)

            z, mu, var = vae.encoder(x)
            recon_x = vae.decoder(
                z,
                idx,
                col_msk_threshold=col_msk_threshold,
            )

            # using bce loss estimating the error
            recon_loss = F.binary_cross_entropy(recon_x, x) * x.size(-1)
            kl_loss = kl_div(mu, var)

            if loss_monitor is not None:
                loss_monitor.log({"recon_loss": recon_loss, "kl_loss": kl_scale * kl_loss})
            loss = {"recon_loss": recon_loss, "kl_loss": kl_scale * kl_loss}
            if session is not None:
                session.report({"loss": sum(loss.values()).item()})

            optimizer.zero_grad()
            sum(loss.values()).backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            for k, v in loss.items():
                epoch_loss[k] += loss[k].item()

        if is_lr_scheduler:
            scheduler.step()

        epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
        epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
        tq.set_postfix_str(epoch_info)

        if is_tensorboard:
            writer.add_scalars(
                "train",
                {
                    "kl_loss": epoch_loss["kl_loss"],
                    "recon_loss": epoch_loss["recon_loss"],
                },
                epoch,
            )
    # for some config record
    return epoch_loss


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

    adata = sc.read_h5ad(pwd + "data/pbmc_RNA-ATAC.h5ad")
    dataloader = convert_adata_to_dataloader(
        adata=adata,
        batch_size=batch_size,
        is_shuffle=True,
    )

    model_save_path = pwd + "model/ckpt/" + model_path
    vae_train(vae=model, dataloader=dataloader, num_epoch=num_epoch, device=device)

    torch.save(model.state_dict(), model_save_path)
