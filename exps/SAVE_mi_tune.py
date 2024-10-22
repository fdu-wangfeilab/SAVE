# %%
import scanpy as sc
import numpy as np
import torch
import wandb
from scib_metrics.benchmark import Benchmarker
import argparse

import os
import sys
import ray
import tracemalloc
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

pwd = "/home/lijiahao/workbench/SAVE/"
sys.path.append(pwd)
os.chdir(pwd)

from model.save_model import SAVE
from model.utils.process_h5ad import batch_scale, preprocessing_rna

# %%
# download the data via link: https://figshare.com/ndownloader/files/24539828


def do_train(config):
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
    # kwargs["lr_milestone"] = config["lr_milestone"]
    # kwargs["lr"] = config["lr"]
    # kwargs["weight_decay"] = config["weight_decay"]
    kwargs["expand_dim"] = 256
    kwargs["lr_milestone"] = 2000
    kwargs["capacity_milestone"] = 2000
    kwargs["enc_dim"] = 8
    kwargs["cls_scale"] = 100
    kwargs["cov_scale"] = 1
    kwargs.update(config)

    from model.save_model import SAVE

    # run = wandb.init(project="SAVE_mi", config=kwargs)
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    save_model.train(loss_monitor=None, session=None, **kwargs)

    save_model.device = torch.device("cpu")
    latent = save_model.get_latent(batch_size=1024, latent_pos=0)
    adata.obsm["SAVE"] = latent

    bm = Benchmarker(
        adata=adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["SAVE"],
    )
    bm.benchmark()
    resdf = bm.get_results(min_max_scale=False)
    scib_score = resdf["Total"][0]
    session.report({"loss": scib_score})


def ray_tune(args):
    ray.init(num_cpus=40, num_gpus=1)
    tracemalloc.start()
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-3),
        # "lr_milestone": tune.choice([500, 1000, 2000]),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        # "expand_dim": tune.choice([256]),
        # "enc_dim": tune.choice([8]),
        "kl_scale": tune.choice([50, 100, 150, 200]),
        "capacity": tune.choice(list(np.arange(25, 301, 25).astype(int))),
        # "cov_scale": tune.choice([1, 2, 5, 10]),
        # "cls_scale": tune.choice([1, 2, 5, 10]),
    }

    scheduler = ASHAScheduler(max_t=5000, grace_period=10, reduction_factor=4)

    optuna_search = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(do_train), resources={"gpu": 0.25}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=500,
            metric="loss",
            mode="max",
        ),
        param_space=search_space,
    )

    tuner.fit()


if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument('--kl_scale', type=float, default=2)
    # args.add_argument('--capacity', type=int, default=15)
    # args = args.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    ray_tune(None)
