# %%
import scanpy as sc
import torch
import wandb
from scib_metrics.benchmark import Benchmarker

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
    # sota training paramters 0.717519 https://wandb.ai/argon_cs-fudan-university/SAVE_mi/runs/nitajxoh?nw=nwuserargon_cs
    kwargs["iter"] = 5000
    kwargs["expand_dim"] = 256
    kwargs["lr_milestone"] = 2000
    kwargs["capacity_milestone"] = 2000
    kwargs["enc_dim"] = 8
    kwargs["cls_scale"] = 200
    kwargs["cov_scale"] = 1
    
    kwargs['lr'] = 0.000735422158880082
    kwargs['weight_decay'] = 0.00012432096152802417

    

    kwargs["kl_scale"] = 2
    kwargs["capacity"] = 5
    kwargs["capacity_milestone"] = 2000

    from model.save_model import SAVE

    run = wandb.init(project="SAVE_mi", config=kwargs)

    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        col_msk_threshold=-1,
        **kwargs,
    )

    save_model.train(loss_monitor=run, session=None, is_period_save=False, **kwargs)
    save_model.save_ckpt("./ckpt/SAVE_1025_cov_pancreas.pt")

    latent2 = save_model.get_latent(batch_size=1024, latent_pos=1)

    print(latent2.shape)
    adata.obsm["SAVE_mu_cell"] = latent2[:, :-2]
    adata.obsm["SAVE_mu_batch"] = latent2[:, -2:]
    adata.obsm["SAVE_mu_total"] = latent2

    bm = Benchmarker(
        adata=adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["SAVE_mu_cell", "SAVE_mu_batch", "SAVE_mu_total"],
    )
    bm.benchmark()
    resdf = bm.get_results(min_max_scale=False)
    resdf.to_csv(f"./result/SAVE_1025_cov_pancreas.csv")
    print(resdf)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    do_train()
