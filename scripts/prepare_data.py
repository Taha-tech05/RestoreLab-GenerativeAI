"""Prepare deterministic Oxford-IIIT Pet metadata or the FS2K split."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path

from common.data_fs2k import load_fs2k_splits
from common.data_pets import prepare_pet_metadata
from common.paths import add_common_args, ensure_output_dirs, resolve_paths


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, "paths.yaml")
    parser.add_argument("--dataset", choices=("pets", "fs2k"), default="pets")
    args = parser.parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths)
    cfg = paths["config"]
    if args.dataset == "pets":
        split, val, test = prepare_pet_metadata(Path(paths["data_root"]), Path(paths["out_dir"]),
                                                cfg.get("seed", 42), download=True)
        print(f"Pet split: {len(split['train_indices'])} train, {len(split['val_indices'])} val; {len(test)} fixed test rows")
    else:
        fs2k_root = Path(paths["data_root"]) / cfg.get("fs2k_subdir", "FS2K")
        train, val, test = load_fs2k_splits(fs2k_root, Path(paths["out_dir"]) / "fs2k_split.json", 42)
        print(f"FS2K split: {len(train)} train, {len(val)} val, {len(test)} official test")


if __name__ == "__main__":
    main()
