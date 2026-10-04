"""Portable project paths and CLI configuration shared by every script."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_CONFIG = ROOT / "configs" / "paths.yaml"


def add_common_args(parser: argparse.ArgumentParser, task_config: str) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / task_config)
    return parser


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | dict[str, Any]]:
    cfg = load_config(args.config)
    path_cfg = load_config(DEFAULT_PATHS_CONFIG)
    data_root = args.data_root or Path(os.getenv("RESTORELAB_DATA_ROOT", path_cfg.get("data_root", "data")))
    out_dir = args.out_dir or Path(os.getenv("RESTORELAB_OUT_DIR", path_cfg.get("out_dir", "experiments")))
    ckpt_dir = args.ckpt_dir or Path(os.getenv("RESTORELAB_CKPT_DIR", path_cfg.get("ckpt_dir", "checkpoints")))
    resolved = {
        "root": ROOT,
        "data_root": data_root if data_root.is_absolute() else ROOT / data_root,
        "out_dir": out_dir if out_dir.is_absolute() else ROOT / out_dir,
        "ckpt_dir": ckpt_dir if ckpt_dir.is_absolute() else ROOT / ckpt_dir,
        "config_path": args.config.resolve(),
        "config": cfg,
    }
    print("Resolved paths:")
    for key in ("root", "data_root", "out_dir", "ckpt_dir", "config_path"):
        print(f"  {key}: {resolved[key]}")
    return resolved


def ensure_output_dirs(paths: dict[str, Path | dict[str, Any]]) -> None:
    Path(paths["out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
