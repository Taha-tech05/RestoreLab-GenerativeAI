"""FS2K paired photo/sketch data pipeline from Task 4 notebook cell 4."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

S = 128


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_img(folder: Path, stem: str) -> Path:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{folder}/{stem}.*")


class FS2KDataset(Dataset):
    def __init__(self, root: Path, records, train=False):
        self.root = Path(root)
        self.records = records
        self.data = [self.load_pair(record) for record in records]
        self.train = train

    def __len__(self):
        return len(self.data)

    def sketch_path(self, image_name: str) -> Path:
        source, base = image_name.split("/")
        number = base.replace("image", "")
        for suffix in (source[-1], "1", "2", "3"):
            try:
                return find_img(self.root / "sketch" / f"sketch{suffix}", f"sketch{number}")
            except FileNotFoundError:
                pass
        raise FileNotFoundError(image_name)

    def load_pair(self, record):
        directory, basename = record["image_name"].split("/")
        photo = Image.open(find_img(self.root / "photo" / directory, basename)).convert("RGB").resize((S, S), Image.BICUBIC)
        sketch = Image.open(self.sketch_path(record["image_name"])).convert("RGB").resize((S, S), Image.BICUBIC)
        to_tensor = lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
        return to_tensor(photo), to_tensor(sketch), record["style"]

    def __getitem__(self, index):
        photo, sketch, style = self.data[index]
        if self.train and random.random() < 0.5:
            photo, sketch = photo.flip(-1), sketch.flip(-1)
        return photo, sketch, torch.tensor(style)


def load_fs2k_splits(data_root: Path, split_path: Path | None = None, seed=42):
    root = Path(data_root)
    train_all = json.loads((root / "anno_train.json").read_text(encoding="utf-8"))
    test_records = json.loads((root / "anno_test.json").read_text(encoding="utf-8"))
    styles = [record["style"] for record in train_all]
    train_ids, val_ids = train_test_split(range(len(train_all)), test_size=0.15,
                                          stratify=styles, random_state=seed)
    train_records = [train_all[i] for i in train_ids]
    val_records = [train_all[i] for i in val_ids]
    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps({"seed": seed, "train_indices": list(train_ids),
                                          "val_indices": list(val_ids)}, indent=2), encoding="utf-8")
    return train_records, val_records, test_records


def build_fs2k_datasets(data_root: Path, split_path: Path | None = None, seed=42):
    train_records, val_records, test_records = load_fs2k_splits(data_root, split_path, seed)
    return {
        "train": FS2KDataset(data_root, train_records, train=True),
        "val": FS2KDataset(data_root, val_records),
        "test": FS2KDataset(data_root, test_records),
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
    }
