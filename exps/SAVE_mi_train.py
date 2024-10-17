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
    kwargs['iter'] = 5000
    kwargs["batch_size"] = 1024
    kwargs['lr_milestone'] = 1000

    kwargs['lr'] = 0.00475898
    kwargs['expand_dim'] = 16
    kwargs['weight_decay'] = 6.83645e-05

    from model.save_model import SAVE

    run = wandb.init(project="SAVE_mi", config=kwargs)
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    save_model.train(loss_monitor=run, session=None, **kwargs)
    save_model.save_ckpt('./ckpt/SAVE_MI_loss_1016_warmup.pt')


if __name__ == "__main__":
    # os.environ['WANDB_MODE'] = 'offline'
    do_train()