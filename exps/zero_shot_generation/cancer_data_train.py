import numpy as np
import pickle
import torch
from rich import print

from scipy.sparse import csr_matrix
import scanpy as sc

import sys
import os

pwd = os.getcwd()
sys.path.append(pwd)
os.chdir(pwd)

from model.utils.process_h5ad import preprocessing_rna, batch_scale

if __name__ == "__main__":
    device = torch.device("cuda:1")
    gpu_num = 1
    seed = 1202

    adata = sc.read_h5ad("data/cancer_lung/cancer_lung_hvg_train_scale.h5ad")

    import yaml

    yaml_path = "./model_setting.yml"
    with open(yaml_path, "r") as f:
        model_setting = yaml.load(f, Loader=yaml.FullLoader)
    vae_setting = model_setting["SAVE-B"]["model"]
    train_setting = model_setting["SAVE-B"]["train"]

    kwargs = {
        "gpu_num": gpu_num,
        "device": device,
        "seed": seed,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_SAVE-B",
        "is_data_scaled": True,
    }

    kwargs.update(vae_setting)
    kwargs.update(train_setting)

    from model.save_model import SAVE

    save_model = SAVE(
        adata=adata,
        is_initialized=True,
        condition_cols=[
            "batch",
            "cell_type",
            "development_stage",
            "disease",
            "uicc_stage",
        ],
        **kwargs,
    )

    save_model.train(col_msk_threshold=0.2, **kwargs)
    save_model.save_ckpt(
        "total_exp_res/04_cond_gen/ckpt/cancer_lung.pt"
    )
