import logging
import os
import re
from dataclasses import asdict
from glob import glob
from pathlib import Path
import json
import wandb
from omegaconf import open_dict

import draccus
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# from lerobot.common.policies.pretrained import PreTrainedPolicy
# from lerobot.common.utils.utils import get_global_random_state, set_global_random_state
# from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode

# def save_training_state(
#         save_dir: str | Path,
#         train_step: int,
#         optimizer: Optimizer | None = None,
#         scheduler: LRScheduler | None = None,
# ):
#     """Checkpoint the global training_step, optimizer state, scheduler state, and random state.
#
#     All of these are saved as "training_state.pth" under the checkpoint directory.
#     """
#     training_state = {}
#     training_state["step"] = train_step
#     training_state.update(get_global_random_state())
#     if optimizer is not None:
#         training_state["optimizer"] = optimizer.state_dict()
#     if scheduler is not None:
#         training_state["scheduler"] = scheduler.state_dict()
#
#     save_dir = Path(save_dir)
#     os.makedirs(save_dir, exist_ok=True)
#     torch.save(training_state, save_dir / "training_state.pth")

def register_features_types():
    draccus.decode.register(FeatureType, lambda x: FeatureType[x])
    draccus.encode.register(FeatureType, lambda x: x.name)

    draccus.decode.register(NormalizationMode, lambda x: NormalizationMode[x])
    draccus.encode.register(NormalizationMode, lambda x: x.name)


# def load_training_state(checkpoint_dir: str | Path, optimizer: Optimizer, scheduler: LRScheduler | None) -> int:
#     """
#     Given the checkpoint directory, load the optimizer state, scheduler state, and random state, and
#     return the global training step.
#     """
#     # TODO(aliberts): use safetensors instead as weights_only=False is unsafe
#     checkpoint_dir = Path(checkpoint_dir)
#     training_state = torch.load(checkpoint_dir / "training_state.pth", weights_only=False)
#     optimizer.load_state_dict(training_state["optimizer"])
#     if scheduler is not None:
#         scheduler.load_state_dict(training_state["scheduler"])
#     elif "scheduler" in training_state:
#         raise ValueError("The checkpoint contains a scheduler state_dict, but no LRScheduler was provided.")
#     # Small HACK to get the expected keys: use `get_global_random_state`.
#     set_global_random_state({k: training_state[k] for k in get_global_random_state()})
#     return training_state["step"], optimizer, scheduler

def save_wandb(wandb, save_dir):
    wandb_info = {
        "entity": wandb.run.entity,
        "project": wandb.run.project,
        "name": wandb.run.name,
        "id": wandb.run.id
    }
    with open(os.path.join(save_dir, "wandb_run_info.json"), "w") as f:
        json.dump(wandb_info, f, indent=2)


def initialize_wandb(cfg):
    if cfg.resume_ckpt:
        save_dir = Path(cfg.resume_ckpt)
        with open(save_dir.parent / "wandb_run_info.json", "r") as f:
            wandb_info = json.load(f)
        cfg.wandb_entity = wandb_info["entity"]
        cfg.wandb_project = wandb_info["project"]
        cfg.exp_name = wandb_info["name"]
        with open_dict(cfg):
            cfg['wandb_id'] = wandb_info["id"]

    wandb.init(project=cfg.wandb_project,#entity=cfg.wandb_entity,
               name=f"{cfg.exp_name}",
               id=cfg.wandb_id if cfg.resume_ckpt else None,
               mode="online",
               resume="must" if cfg.resume_ckpt else None)

    return cfg