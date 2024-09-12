import sys
import os
import numpy as np
import logging

pwd = os.getcwd()
os.chdir(pwd)
sys.path.append(pwd)

import torch
import scanpy as sc
from model.utils.process_h5ad import preprocessing_rna, batch_scale


if __name__ == "__main__":
    logging.basicConfig(
        filename="./multi-cond/cancer/save_train.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    # data can be download at: https://datasets.cellxgene.cziscience.com/99040b08-7e1a-4d81-911c-4d2fd2335757.h5ad
    adata = sc.read_h5ad("data/cancer_lung/cancer_lung_hvg_55_70.h5ad")
    adata.obs["batch"] = adata.obs["assay"]
    # adata, _ = batch_scale(adata)
    test_cond_1 = {
        "development_stage": "65-year-old human stage",
        "batch": "10x 3' v2",
        "disease": "chronic obstructive pulmonary disease",
        "uicc_stage": "non-cancer",
    }
    test_cond_2 = {
        "development_stage": "70-year-old human stage",
        "batch": "Smart-seq2",
        "disease": "squamous cell lung carcinoma",
        "uicc_stage": "I",
    }

    test_msk1 = (
        (adata.obs["development_stage"] == test_cond_1["development_stage"])
        & (adata.obs["batch"] == test_cond_1["batch"])
        & (adata.obs["disease"] == test_cond_1["disease"])
        & (adata.obs["uicc_stage"] == test_cond_1["uicc_stage"])
    )
    test_msk2 = (
        (adata.obs["development_stage"] == test_cond_2["development_stage"])
        & (adata.obs["batch"] == test_cond_2["batch"])
        & (adata.obs["disease"] == test_cond_2["disease"])
        & (adata.obs["uicc_stage"] == test_cond_2["uicc_stage"])
    )


    adata_train = adata[~test_msk1 & ~test_msk2]

    adata_train, _ = batch_scale(adata_train)

    import yaml

    yaml_path = os.path.join(pwd, "./model_setting.yml")
    with open(yaml_path, "r") as f:
        model_setting = yaml.load(f, Loader=yaml.FullLoader)
    vae_setting = model_setting["SAVE-B"]["model"]
    train_setting = model_setting["SAVE-B"]["train"]

    kwargs = {
        "gpu_num": 1,
        "device": torch.device("cuda:1"),
        "seed": 1202,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_SAVE-B",
        "is_data_scaled": True,
    }

    kwargs.update(vae_setting)
    kwargs.update(train_setting)

    from model.save_model import SAVE

    try:
        save_model = SAVE(
            adata=adata_train,
            is_initialized=True,
            condition_cols=["batch", "development_stage", "disease", "uicc_stage"],
            **kwargs,
        )

        ckpt_dir = os.path.join(
            "./ckpt",
        )
        ckpt_path = os.path.join(ckpt_dir, "cancer_multi.pt")

        save_model.train(col_msk_threshold=0.2, **kwargs)
        save_model.save_ckpt(ckpt_path)

    except Exception as e:
        print(e)
