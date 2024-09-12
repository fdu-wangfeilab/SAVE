import sys
import os
import numpy as np
import logging
import pickle
import pandas as pd

pwd = os.getcwd()
os.chdir(pwd)
sys.path.append(pwd)

import torch
import scanpy as sc
from model.utils.process_h5ad import preprocessing_rna, batch_scale

if __name__ == "__main__":
    logging.basicConfig(
        filename="./multi-cond/cancer/save_eval.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    res_saving_dir = "./multi-cond/cancer/res"
    adata = sc.read_h5ad("data/cancer_lung/cancer_lung_hvg_55_70.h5ad")
    adata.obs["batch"] = adata.obs["assay"]
    # adata, scaler = batch_scale(adata)
    # pickle.dump(scaler, open(os.path.join(res_saving_dir, "scaler.pkl"), "wb"))

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

    input_conds1 = [
        {
            "development_stage": "55-year-old human stage",
            "batch": "10x 3' v2",
            "disease": "chronic obstructive pulmonary disease",
            "uicc_stage": "non-cancer",
        },
        {
            "development_stage": "60-year-old human stage",
            "batch": "10x 3' v2",
            "disease": "chronic obstructive pulmonary disease",
            "uicc_stage": "non-cancer",
        },
        {
            "development_stage": "70-year-old human stage",
            "batch": "10x 3' v2",
            "disease": "chronic obstructive pulmonary disease",
            "uicc_stage": "non-cancer",
        },
    ]

    input_conds2 = [
        {
            "development_stage": "60-year-old human stage",
            "batch": "10x 3' v2",
            "disease": "squamous cell lung carcinoma",
            "uicc_stage": "I",
        },
        {
            "development_stage": "70-year-old human stage",
            "batch": "Smart-seq2",
            "disease": "squamous cell lung carcinoma",
            "uicc_stage": "III",
        },
        {
            "development_stage": "55-year-old human stage",
            "batch": "10x 3' v2",
            "disease": "normal",
            "uicc_stage": "non-cancer",
        },
    ]

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

    adata_train = adata[~test_msk1 & ~test_msk2].copy()
    adata_train, train_scaler = batch_scale(adata_train)
    pickle.dump(
        train_scaler, open(os.path.join(res_saving_dir, "train_scaler.pkl"), "wb")
    )

    adata_test = adata[test_msk1 | test_msk2].copy()
    adata_test, test_scaler = batch_scale(adata_test)
    pickle.dump(
        test_scaler, open(os.path.join(res_saving_dir, "test_scaler.pkl"), "wb")
    )

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
        save_model.load_ckpt(ckpt_path)

    except Exception as e:
        print(e)


    stage_info = pd.read_csv(
        "total_exp_res/01_perturb_prediction/multi-cond/cancer/stage_distribution.csv"
    )
    os.makedirs(os.path.join(res_saving_dir, "trans_loop"), exist_ok=True)
    for i in range(len(stage_info)):
        ori_condition = {
            "development_stage": stage_info.loc[i, "year"],
            "batch": stage_info.loc[i, "batch"],
            "disease": stage_info.loc[i, "disease"],
            "uicc_stage": stage_info.loc[i, "stage"],
        }
        if i in [13, 20]:
            print("test to train")
            input_data = adata_test[
                np.logical_and.reduce(
                    [adata_test.obs[x[0]] == x[1] for x in ori_condition.items()]
                )
            ]
        else:
            input_data = adata_train[
                np.logical_and.reduce(
                    [adata_train.obs[x[0]] == x[1] for x in ori_condition.items()]
                )
            ]
        if input_data.shape[0] == 0:
            continue
        os.makedirs(
            os.path.join(res_saving_dir, "trans_loop", f"input_cond_{i}"), exist_ok=True
        )
        for j in range(len(stage_info)):
            if i == j:
                continue
            target_condition = {
                "development_stage": stage_info.loc[j, "year"],
                "batch": stage_info.loc[j, "batch"],
                "disease": stage_info.loc[j, "disease"],
                "uicc_stage": stage_info.loc[j, "stage"],
            }
            pred_h5ad = save_model.transfer_cond(input_data, target_condition)
            scaler = pickle.load(
                open(os.path.join(res_saving_dir, "train_scaler.pkl"), "rb")
            )
            for s in scaler:
                if s[0] == target_condition["batch"]:
                    select_scaler = s[1]
                    break
            pred_h5ad.X = select_scaler.inverse_transform(pred_h5ad.X)

            pred_h5ad.obs["cell_type"] = list(input_data.obs["cell_type"])
            pred_h5ad.obs["cell_type"] = pred_h5ad.obs["cell_type"].astype("category")
            pred_h5ad.write_h5ad(
                os.path.join(
                    res_saving_dir,
                    "trans_loop",
                    f"input_cond_{i}",
                    f"target_cond_{j}.h5ad",
                )
            )
