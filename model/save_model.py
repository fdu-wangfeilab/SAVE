import torch
import scanpy as sc
import numpy as np

from tqdm import tqdm
import scipy
from scipy.sparse import csr
from anndata import AnnData
import sys

sys.path.append("/home/lijiahao/workbench/sc-save")
from model.utils.process_h5ad import convert_adata_to_dataloader
from model.VAE_train import vae_train
from model.VAE_model import VAE
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

CHUNK_SIZE = 40000


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print("There are {} gene in selected genes".format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size : (i + 1) * chunk_size, idx] = adata[
                i * chunk_size : (i + 1) * chunk_size, genes[idx]
            ].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={"var_names": genes})
    return adata


def batch_scale(adata, chunk_size=CHUNK_SIZE, is_uniform=False):
    """
    Batch-specific scale data

    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    # 就是 b 就是 batchname，如 rna，atac 的数据，b就是('rna','atac')
    for b in adata.obs["batch"].unique():
        # 取某个batch 所有数据的 index
        idx = np.where(adata.obs["batch"] == b)[0]

        # 对某个batch中的数据做归一化, 这样也是对列做归一化
        if is_uniform:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=False).fit(
                adata.X[idx]
            )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])

        tr = tqdm(range(len(idx) // chunk_size + 1), ncols=80)
        tr.set_description_str(desc="batch_scale")
        for i in tr:
            adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
            )

    return adata


class SAVE:
    def __init__(
        self,
        adata: sc.AnnData,
        device,
        is_initialized=True,
        is_print_model=False,
        seed=1202,
        hidden_dim=64,
        enc_dim=22,
        expand_dim=256,
        enc_num_heads=4,
        dec_num_heads=4,
        enc_depth=3,
        dec_depth=3,
        enc_affine=True,
        enc_norm_type="ln",
        dec_norm_init=True,
        dec_blk_type="adaLN",
        condition_cols=["batch"],
        **kwargs,
    ) -> None:
        self.adata = adata.copy()
        self.num_cells = adata.shape[0]
        self.num_genes = adata.shape[1]
        self.device = device
        self.is_trained = False

        self.condition_cols = condition_cols
        self.condition_dict = self.get_cond_correspond_idx()

        # to stablize model initial performance
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.model = VAE(
            input_dim=self.num_genes,
            hidden_dim=hidden_dim,
            enc_dim=enc_dim,
            expand_dim=expand_dim,
            enc_num_heads=enc_num_heads,
            dec_num_heads=dec_num_heads,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            enc_affine=enc_affine,
            enc_norm_type=enc_norm_type,
            dec_norm_init=dec_norm_init,
            dec_blk_type=dec_blk_type,
            is_initialize=is_initialized,
            len_col_comb=len(self.condition_cols),
        )

        if is_initialized:
            print(f"initialze model weight with seed {seed}")
        if is_print_model:
            print(self.model)

    def get_cond_correspond_idx(self):
        condition_cols = self.condition_cols
        idx_max = 0
        total_cond_dict = {}

        for col in condition_cols:
            batch_idx = {}
            # get all cond in one col
            cond_name = list(self.adata.obs[col].unique())
            # convert to int16 for complex condition
            cond_codes = self.adata.obs[col].cat.codes.values.astype("int16") + idx_max

            # get id for each cond in one col
            for b in cond_name:
                msk = self.adata.obs[col] == b
                unique_id = np.unique(cond_codes[msk])
                assert len(unique_id) == 1, "unique id length should be 1"

                batch_idx[b] = unique_id[0]

            total_cond_dict[col] = batch_idx
            idx_max = idx_max + cond_codes.max() + 1
        return total_cond_dict

    def train(
        self,
        epoch: int = 15,
        batch_size: int = 64,
        lr: float = 0.00015,
        seed: int = 1202,
        is_grad_clip: bool = True,
        iter: int = 8000,
        weight_decay: float = 5e-4,
        is_lr_scheduler: bool = False,
        col_msk_threshold: int = -1,
        loss_monitor = None,
        session = None,
        **kwargs,
    ):

        dataloader = convert_adata_to_dataloader(
            self.adata,
            batch_size=batch_size,
            is_shuffle=True,
            condition_cols=self.condition_cols,
        )

        # if epoch * len(dataloader) < iter:

        epoch = int(iter / len(dataloader))
        print(f"total iter: {epoch * len(dataloader)}")

        self.model.to(self.device)

        vae_train(
            vae=self.model,
            dataloader=dataloader,
            num_epoch=epoch,
            device=self.device,
            kl_scale=0.5,
            is_tensorboard=False,
            grad_clip=is_grad_clip,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            is_lr_scheduler=is_lr_scheduler,
            col_msk_threshold=col_msk_threshold,
            loss_monitor = loss_monitor,
            session = session,
        )

        self.is_trained = True

    def get_latent(self, batch_size: int = 1024, latent_pos: int = 1):
        dataloader = convert_adata_to_dataloader(
            adata=self.adata,
            batch_size=batch_size,
            is_shuffle=False,
            condition_cols=self.condition_cols,
        )
        self.model.to(self.device)
        self.model.eval()
        latents = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = x.float().to(self.device), idx.long().to(self.device)
            z = self.model.encoder(x)[latent_pos]
            latents.append(z.detach().cpu().numpy())
        return np.concatenate(latents, axis=0)

    def get_recon(
        self,
        batch_size: int = 128,
        latent_pos: int = 1,
        target_msk_col=None,
        target_msk_val=None,
    ):
        dataloader = convert_adata_to_dataloader(
            adata=self.adata,
            batch_size=batch_size,
            is_shuffle=False,
            condition_cols=self.condition_cols,
        )

        self.model.to(self.device)
        self.model.eval()

        if target_msk_col is not None:
            print(f"generate with mask col: {target_msk_col}")

        total_recon = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = x.float().to(self.device), idx.long().to(self.device)
            z = self.model.encoder(x)[latent_pos]
            if target_msk_col is None:
                recon = self.model.decoder(z, idx, col_msk_threshold=-1)
            elif target_msk_col is not None and target_msk_val is not None:
                # change the condition of one column
                idx[:, target_msk_col] = target_msk_val
                idx.long().to(self.device)
                recon = self.model.decoder(
                    z, idx, col_msk_threshold=-1
                )
            else:
                # only mask one column
                recon = self.model.decoder(
                    z,
                    idx,
                    col_msk_threshold=1,
                    target_msk_col=target_msk_col,
                )
            total_recon.append(recon.detach().cpu().numpy())
        return np.concatenate(total_recon, axis=0)

    def save_ckpt(self, ckpt_path: str):
        if not self.is_trained:
            print("model not trained!")
            return

        torch.save(self.model.state_dict(), ckpt_path)
        print(f"save ckpt to {ckpt_path}!")

    def load_ckpt(self, ckpt_path: str):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.is_trained = True
        print(f"successfully load {ckpt_path}!")

    def transfer_cond(
        self, input_adata: sc.AnnData, target_cond: dict, batch_size: int = 1024
    ):
        self.model.to(self.device)
        self.model.eval()

        assert (
            target_cond.keys() == self.condition_dict.keys()
        ), "input keys don't match condition keys"

        target_cond_ids = []
        for k in self.condition_dict.keys():
            target_cond_ids.append(self.condition_dict[k][target_cond[k]])

        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(input_adata.X.shape[0], dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        # col combine
        total_idx = torch.hstack(total_idx)

        if isinstance(input_adata.X, np.ndarray):
            ary = input_adata.X
        else:
            ary = input_adata.X.toarray()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(ary.astype(np.float32)), total_idx
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        total_recon = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = (
                x.float().to(self.device),
                idx.long().to(self.device),
            )

            z, mu, var = self.model.encoder(x)
            recon_x = self.model.decoder(
                z, idx, col_msk_threshold=-1
            )

            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=input_adata.var,
            obs={k: v + "_transfered" for k, v in target_cond.items()},
        )

    def sampling_noise(self, generate_count: int, mean=0, std=1):
        # sampling noise
        rand_z = torch.randn(
            generate_count,
            self.model.latent_dim,
        ).to(self.device)
        rand_z = rand_z * std + mean
        return rand_z

    def cond_generate(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        mean=0,
        std=1,
        input_noise=None,
    ):
        self.model.to(self.device)
        self.model.eval()
        total_recon = []
        assert (
            target_cond.keys() == self.condition_dict.keys()
        ), "input keys don't match condition keys"

        # input cond to id
        target_cond_ids = []
        for k in self.condition_dict.keys():
            target_cond_ids.append(self.condition_dict[k][target_cond[k]])

        # sampling noise
        if input_noise is not None:
            rand_z = input_noise.to(self.device)
        else:
            print("random sampling noise!")
            rand_z = self.sampling_noise(generate_count, mean, std)

        # construct dataloader
        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(generate_count, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))

        total_idx = torch.hstack(total_idx)
        dataset = torch.utils.data.TensorDataset(rand_z, total_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # generate
        for z, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            recon_x = self.model.decoder(
                z, idx, col_msk_threshold=-1
            )
            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
            obs={k: v + "_gen" for k, v in target_cond.items()},
        )

    def cond_generate_with_guidance(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        mean=0,
        std=1,
        omega=0.5,
        input_noise=None,
    ):
        self.model.to(self.device)
        self.model.eval()
        total_recon = []
        assert (
            target_cond.keys() == self.condition_dict.keys()
        ), "input keys don't match condition keys"

        # input cond to id
        target_cond_ids = []
        for k in self.condition_dict.keys():
            target_cond_ids.append(self.condition_dict[k][target_cond[k]])

        # sampling noise
        if input_noise is not None:
            rand_z = input_noise.to(self.device)
        else:
            print("random sampling noise!")
            rand_z = self.sampling_noise(generate_count, mean, std)

        # construct dataloader
        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(generate_count, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        total_idx = torch.hstack(total_idx)

        dataset = torch.utils.data.TensorDataset(rand_z, total_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # generate
        for z, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            recon_x_condi = self.model.decoder(
                z, idx, col_msk_threshold=-1
            )

            recon_x_uncondi = self.model.decoder(
                z, idx, col_msk_threshold=-1, row_msk_threshold=1
            )

            recon_x = (1 + omega) * recon_x_condi - omega * recon_x_uncondi

            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
            obs={k: v + f"_guide_{omega}" for k, v in target_cond.items()},
        )

    def preprocessing_rna(
        self,
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,
        chunk_size: int = CHUNK_SIZE,
        is_batch_scale: bool = True,
        log=None,
    ):
        # 重新设置一遍预处理的参数
        if min_features is None:
            min_features = 600
        if n_top_features is None:
            n_top_features = 2000
        if target_sum is None:
            target_sum = 10000
        # 如果不是系数矩阵的格式则转换为稀疏矩阵
        if not isinstance(adata.X, csr.csr_matrix):
            tmp = scipy.sparse.csr_matrix(adata.X)
            adata.X = None
            adata.X = tmp

        # 筛选出 不是['ERCC', 'MT-', 'mt-'] 开头的 gene
        adata = adata[
            :,
            [
                gene
                for gene in adata.var_names
                if not str(gene).startswith(tuple(["ERCC", "MT-", "mt-"]))
            ],
        ]  # type: ignore

        # 直接调用 scanpy 来过滤 cell,gene 并且标准化数据
        # 要求细胞中至少有 min_features 个基因表达
        sc.pp.filter_cells(adata, min_genes=min_features)
        # 要求基因至少在 min cell 中表达
        sc.pp.filter_genes(adata, min_cells=min_cells)
        # 把每一行的数据放缩到 10000，但是保持和为 10000
        # 将每个细胞的总测量量标准化为相同的值，有助于消除细胞之间的测量偏差，

        # 因为在细胞总数相等的情况下，每个基因在不同细胞之间的相对表达量会更加可比。
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # 使数据更加符合正态分布
        sc.pp.log1p(adata)

        # 保存一份数据在 adata.raw 中
        adata.raw = adata

        # 此处 n_top_features 是 2000
        if isinstance(n_top_features, int) and n_top_features > 0:
            # 此处是取 2000 个HVG
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_features,
                batch_key="batch",
                inplace=False,
                subset=True,
            )
        elif not isinstance(n_top_features, int):
            adata = reindex(adata, n_top_features)

        # batch 采用 maxabs 归一化,因为数据全部>0， 最后范围为 [0，1],
        # 分 chunck 归一化每次相当于 max 取chunk中的最大值
        if is_batch_scale:
            adata = batch_scale(adata, chunk_size=chunk_size)

        return adata

    def zero_shot_embeding(
        self,
        input_adata,
        is_data_scaled=False,
        embedding_name="SAVE",
        batch_size=1024,
        latent_pos=1,
    ):
        if not is_data_scaled:
            input_adata = input_adata[:, self.adata.var_names]
            print(input_adata.shape)
            input_adata = self.preprocessing_rna(
                input_adata,
                is_batch_scale=True,
                n_top_features=self.num_genes,
                min_features=0,
                min_cells=0,
            )

        if isinstance(input_adata.X, np.ndarray):
            ary = input_adata.X
        else:
            ary = input_adata.X.toarray()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(ary.astype(np.float32))
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        self.model.to(self.device)
        self.model.eval()
        latents = []
        for x in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x = x[0].float().to(self.device)
            z = self.model.encoder(x)[latent_pos]
            latents.append(z.detach().cpu().numpy())

        input_adata.obsm[embedding_name] = np.concatenate(latents, axis=0)
        return input_adata

    def label_predict(
        self,
        input_adata,
        is_data_scaled=False,
        label_key="cell_type",
        embedding_name="SAVE",
        batch_size=1024,
    ):
        test_data = self.zero_shot_embeding(
            input_adata, is_data_scaled=is_data_scaled, embedding_name=embedding_name
        )

        test_latent = test_data.obsm[embedding_name]
        train_latent = self.get_latent()
        train_label = self.adata.obs[label_key].cat.codes.values

        # train svm as classifier
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline

        clf = make_pipeline(SVC(gamma="auto"))
        clf.fit(train_latent, train_label)

        n_test = test_latent.shape[0]
        it = n_test // batch_size
        y_pred = []
        for i in range(it):
            input = test_latent[i * batch_size : (i + 1) * batch_size]
            print(input.shape)
            y_p = clf.predict(input)
            y_pred.append(y_p)
        y_pred.append(clf.predict(test_latent[it * batch_size :]))
        y_pred = np.concatenate(y_pred)

        label_encoder = self.adata.obs[label_key].cat
        pred_labels = label_encoder.categories[y_pred]

        input_adata.obs["pred_labels"] = pred_labels
        return input_adata

    def del_adata(self):
        del self.adata
        print("adata deleted!")