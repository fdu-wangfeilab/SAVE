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
    kwargs["lr_milestone"] = 2000
    kwargs["lr"] = 0.0016374239273347798
    kwargs["weight_decay"] = 0.00027092871182009536
    kwargs["cov_scale"] = 1
    kwargs["cls_scale"] = 2
    kwargs["expand_dim"] = 512
    kwargs["enc_dim"] = 8

    kwargs["kl_scale"] = 2
    kwargs["capacity"] = 0
    kwargs["capacity_milestone"] = 2000

    from model.save_model import SAVE

    # for step in range(0, 10000, 1000):
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    # for i in range(0, 5001, 100):
    save_model.load_ckpt(f"./ckpt/SAVE_1022_cov_capacity_pancreas.pt")
    latent = save_model.get_latent(batch_size=1024, latent_pos=0)
    print(latent.shape)
    latent2 = save_model.get_latent(batch_size=1024, latent_pos=1)
    print(latent2.shape)

    adata.obsm['SAVE'] = latent
    adata.obsm['SAVE_mu'] = latent2

    bm = Benchmarker(
        adata=adata,
        batch_key="batch",
        label_key='cell_type',
        embedding_obsm_keys=['SAVE', 'SAVE_mu']
    )
    bm.benchmark()
    resdf = bm.get_results(min_max_scale=False)
    resdf.to_csv(f"./result/SAVE_1022_cov_capacity_pancreas.csv")
    print(resdf)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    do_train()