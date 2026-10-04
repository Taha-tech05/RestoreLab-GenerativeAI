"""CLI, device, reproducibility, W&B, and checkpoint helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from .paths import add_common_args, ensure_output_dirs, load_config, resolve_paths


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def wandb_start(**kwargs):
    """W&B reads WANDB_API_KEY from the environment; no key is stored here."""
    if os.getenv("WANDB_API_KEY"):
        wandb.login()
    return wandb.init(**kwargs)


def save_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


__all__ = ["add_common_args", "ensure_output_dirs", "load_config", "resolve_paths",
           "seed_everything", "get_device", "wandb_start", "save_checkpoint"]
