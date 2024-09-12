import numpy as np
import torch
import pandas as pd
from pathlib import Path
from rich import print
import logging
import pickle

import scanpy as sc

import sys, os

pwd = os.getcwd()
sys.path.append(pwd)
os.chdir(pwd)

from model.utils.process_h5ad import batch_scale

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


def calc_gen_ref_sim(gen_data, ref_data):
    mean_gene_exp = np.mean(ref_data.X.toarray(), axis=0)
    mean_gene_gen = np.mean(gen_data.X, axis=0)
    mean_r2 = r2_score(mean_gene_exp, mean_gene_gen)
    mean_mse = mean_squared_error(mean_gene_exp, mean_gene_gen)
    correlation, pval = pearsonr(mean_gene_exp, mean_gene_gen)
    print(
        f"mean gene expression r2: {mean_r2:.4f}\n mean gene expression mse: {mean_mse:.4f}\n mean gene expression correlation: {correlation:.4f} pval: {pval:.4f}"
    )
    return (round(mean_r2, 4), round(mean_mse, 4), round(correlation, 4))


def get_deg_list(adata, cell_type: str = "1", threshold: float = 0.05):
    gene_list = adata.uns["rank_genes_groups"]["names"][cell_type]
    msk = adata.uns["rank_genes_groups"]["pvals"][cell_type] <= threshold
    return list(gene_list[msk])


def calc_recall(gt_list, pred_list):
    return len(set(gt_list) & set(pred_list)) / len(gt_list)


def calc_precision(gt_list, pred_list):
    if len(pred_list) == 0:
        return 0
    return len(set(gt_list) & set(pred_list)) / len(pred_list)


def calc_f1(gt_list, pred_list):
    recall = calc_recall(gt_list, pred_list)
    precision = calc_precision(gt_list, pred_list)
    if recall + precision == 0:
        return 0
    return 2 * recall * precision / (recall + precision)


def re_adata_rank_gene(adata, group):
    # group means group by
    adata_re = sc.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
    adata_re = remove_single_sample_groups(adata_re, column_name=group)
    sc.tl.rank_genes_groups(adata_re, group, method="wilcoxon")
    return adata_re


def remove_single_sample_groups(adata, column_name="group"):
    """
    从 AnnData 对象中删除只包含一个样本的细胞群
    :param adata: AnnData 对象
    :return: 处理后的 AnnData 对象
    """
    # 获取每个细胞群的样本数量
    group_counts = adata.obs.groupby(column_name).size()

    # 找出样本数量大于1的细胞群
    keep_groups = group_counts[group_counts > 1].index

    # 过滤出属于保留细胞群的细胞
    adata_filtered = adata[adata.obs[column_name].isin(keep_groups)]

    return adata_filtered


def calc_deg_score(
    gen_adata,
    ref_adata,
    group_by="cell_type",
    target_cell_type="1",
    gen_suffix="_guide_0.5",
):
    gen_data_re = re_adata_rank_gene(gen_adata, group_by)
    ref_data_re = re_adata_rank_gene(ref_adata, group_by)

    gen_deg = get_deg_list(gen_data_re, target_cell_type + gen_suffix)
    ref_deg = get_deg_list(ref_data_re, target_cell_type)
    return calc_f1(ref_deg, gen_deg)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./cancer_lung/split_valid.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    adata_test = sc.read_h5ad("data/cancer_lung/cancer_lung_hvg_test_scale.h5ad")
    adata = sc.read_h5ad("data/cancer_lung/cancer_lung_hvg_train_scale.h5ad")

    import yaml

    yaml_path = "./model_setting.yml"
    with open(yaml_path, "r") as f:
        model_setting = yaml.load(f, Loader=yaml.FullLoader)
    vae_setting = model_setting["SAVE-B"]["model"]
    train_setting = model_setting["SAVE-B"]["train"]

    kwargs = {
        "gpu_num": 0,
        "device": torch.device("cpu"),
        "seed": 1202,
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

    save_model.load_ckpt("total_exp_res/04_cond_gen/ckpt/cancer_lung.pt")

    l_mean = 0
    l_std = 1
    logging.info(f"l mean: {l_mean}, l std: {l_std}")
    print(l_mean, l_std)

    adata_test.obs["development_stage"].value_counts()
    omegas = [0]
    years = ["53", "78"]

    for omega in omegas:
        for year in years:
            logging.info(f"start {year}")
            logging.info("-" * 30)

            test_cell = list(
                adata_test[
                    adata_test.obs["development_stage"]
                    == f"{year}-year-old human stage"
                ]
                .obs["cell_type"]
                .unique()
            )

            # ref data 表示某一年的数据
            ref = adata_test[
                adata_test.obs["development_stage"] == f"{year}-year-old human stage"
            ]

            stage = "IV" if year == "53" else "III"

            gen_cond = {
                "batch": ref.obs["batch"].values[0],
                "disease": "lung adenocarcinoma",
                "development_stage": f"{year}-year-old human stage",
                "uicc_stage": stage,
            }

            one_year_score = {}
            sc._settings.ScanpyConfig.figdir = Path(
                os.path.join(
                    f"/home/lijiahao/workbench/sc-save/total_exp_res/04_cond_gen/cancer_lung/fig/cancer_valid_fig_guidance_{omega}",
                    f"{year}_year",
                )
            )
            npy_dir = f"/home/lijiahao/workbench/sc-save/total_exp_res/04_cond_gen/cancer_lung/cancer_valid_fig_guidance_{omega}/{year}_year_h5ad"

            Path(npy_dir).mkdir(parents=True, exist_ok=True)

            ref.write_h5ad(f"{npy_dir}/{year}_ref.h5ad")

            for c in test_cell:
                logging.info(f"start celltype {c}")
                gen_cond["cell_type"] = c

                gen_ood = save_model.cond_generate(
                    generate_count=500,
                    target_cond=gen_cond,
                    mean=l_mean,
                    std=l_std,
                )

                gen_ood.write_h5ad(f"{npy_dir}/{c}_gen.h5ad")