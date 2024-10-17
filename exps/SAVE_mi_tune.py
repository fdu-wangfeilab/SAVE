# %%
import scanpy as sc
import numpy as np
import torch
import wandb

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
    kwargs["lr_milestone"] = config["lr_milestone"]

    kwargs["lr"] = config["lr"]
    kwargs["weight_decay"] = config["weight_decay"]

    from model.save_model import SAVE

    run = wandb.init(project="SAVE_mi", config=kwargs)
    save_model = SAVE(
        adata=adata.copy(),
        is_initialized=True,
        condition_cols=["batch"],
        **kwargs,
    )
    save_model.train(loss_monitor=run, session=session, **kwargs)


def ray_tune():
    ray.init(num_cpus=40, num_gpus=1)
    tracemalloc.start()
    search_space = {
        "lr": tune.loguniform(1e-6, 5e-3),
        # "expand_dim": tune.choice([8, 16, 32]),
        # "mi_scale": tune.choice([0.1, 0.01, 0.001, 1e-4]),
        "lr_milestone": tune.choice([500, 1000, 2000]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
    }

    scheduler = ASHAScheduler(max_t=5000, grace_period=10, reduction_factor=4)

    optuna_search = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(do_train), resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=50,
            metric="loss",
            mode="min",
        ),
        param_space=search_space,
    )

    results = tuner.fit()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    ray_tune()
