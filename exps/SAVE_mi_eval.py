# %%
import scanpy as sc
import torch
import pandas as pd
import wandb

import os
import sys

from scib_metrics.benchmark import Benchmarker
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
    kwargs["batch_size"] = 48

    kwargs["lr"] = 0.0008761953163128317
    kwargs["expand_dim"] = 32
    kwargs["weight_decay"] = 2.4443717513092533e-05

    from model.save_model import SAVE

    # for step in range(0, 10000, 1000):
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    save_model.load_ckpt(f"./ckpt/SAVE_MI_1016_warmup_opt.pt")
    latent = save_model.get_latent(batch_size=32)
    adata.obsm['SAVE'] = latent

    bm = Benchmarker(
        adata=adata,
        batch_key="batch",
        label_key='cell_type',
        embedding_obsm_keys=['SAVE']
    )
    bm.benchmark()
    resdf = bm.get_results(min_max_scale=False)
    resdf.to_csv(f"./ckpt/SAVE_MI_1016_warmup_opt.pt")

    print(resdf)


if __name__ == "__main__":
    do_train()
