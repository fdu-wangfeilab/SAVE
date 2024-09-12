import numpy as np

from scipy import stats
import scanpy as sc




# get the batch idx
def get_batch_corres_idx(adata: sc.AnnData, condition_cols: list=['batch']):
    batch_idx = {}
    idx_max = 0

    for col in condition_cols:
        batch_name = list(adata.obs[col].unique())
        batch_codes = adata.obs[col].cat.codes.values + idx_max
        for b in batch_name:
            msk = adata.obs[col] == b
            id = np.unique(batch_codes[msk])[0]
            batch_idx[b] = id
        idx_max = idx_max + batch_codes.max() + 1
    return batch_idx


def get_one_batch(adata: sc.AnnData, batch_name: str):
    msk = adata.obs["batch"] == batch_name
    ret_data = adata[msk]
    ret_data.obs["batch"] = ret_data.obs["batch"].astype("category")
    return ret_data


# # do interp
# def interp_data(
#     ori_batch_data: sc.AnnData,
#     save_obj: SAVE,
#     batch_idx: dict,
#     target_batch: str,
#     interp_batch_suffix: str = "",
# ):
#     interp_adata = ori_batch_data.copy()
#     interp_adata.X = csr_matrix(
#         save_obj.intepret_batch(
#             input_adata=interp_adata,  # type: ignore
#             target_batch_id=batch_idx[target_batch],
#         )
#     )

#     interp_adata.obs["batch"] = interp_batch_suffix + target_batch
#     interp_adata.obs["batch"] = interp_adata.obs["batch"].astype("category")

#     return interp_adata


# def all_batch_to_target(
#     adata: sc.AnnData,
#     model: SAVE,
#     ori_batch_names: list,
#     target_batch_name: str,
#     total_batch_idx: dict,
#     interp_batch_suffix: str = "_trans_",
# ):
#     total_adata = []
#     for batch_name in ori_batch_names:
#         ori_adata = get_one_batch(adata, batch_name=batch_name)
#         interped_adata = interp_data(
#             ori_batch_data=ori_adata,
#             save_obj=model,
#             batch_idx=total_batch_idx,
#             target_batch=target_batch_name,
#             interp_batch_suffix=batch_name + interp_batch_suffix,
#         )

#         total_adata.append((ori_adata, interped_adata))

#     return total_adata


# calc rowwise pcc
def calc_row_pcc(gt_npy: np.ndarray, recon_npy: np.ndarray):
    row_pccs = [stats.pearsonr(x, y)[0] for x, y in zip(gt_npy, recon_npy)]
    return np.mean(row_pccs)
