# %%
import scanpy as sc
import torch
import wandb

import os
import sys

pwd = "/home/lijiahao/workbench/SAVE/"
sys.path.append(pwd)
os.chdir(pwd)


def do_train():
    adata = sc.read_h5ad("/home/lijiahao/workbench/SAVE/processed_pancreas.h5ad")
    device = torch.device("cuda:0")
    seed = 1202
    kwargs = {
        "device": device,
        "seed": seed,
        "is_data_scaled": True,
    }
    import yaml

    with open(pwd + "/model_setting.yml", "r") as f:
        setting = yaml.safe_load(f)

    kwargs.update(setting["SAVE-B"]["train"])
    kwargs.update(setting["SAVE-B"]["model"])
    kwargs["iter"] = 5000
    kwargs["lr_milestone"] = 2000
    kwargs["lr"] = 0.0004971869093472658
    kwargs["weight_decay"] = 0.0005638976710982813
    kwargs["cov_scale"] = 1
    kwargs["cls_scale"] = 10
    kwargs["expand_dim"] = 512
    kwargs["enc_dim"] = 8

    kwargs["kl_scale"] = 2
    kwargs["capacity"] = 0
    kwargs["capacity_milestone"] = 2000

    from model.save_model import SAVE

    run = wandb.init(project="SAVE_mi", config=kwargs)
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    save_model.train(loss_monitor=run, session=None, is_period_save=False, **kwargs)
    save_model.save_ckpt("./ckpt/SAVE_1022_cov_capacity_pancreas.pt")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    do_train()
