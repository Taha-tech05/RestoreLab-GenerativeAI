"""Oxford-IIIT Pet transforms, split loading, and manifest datasets."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF

from .corruptions import (CORRUPTION_TYPES, apply_gaussian_blur, apply_occlusion,
                          apply_salt_and_pepper, generate_test_manifest, generate_val_manifest)

PET_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])


def prepare_pet_metadata(data_root: Path, out_dir: Path, seed: int = 42, download: bool = False):
    dataset = OxfordIIITPet(root=str(data_root), split="trainval", download=download)
    total_samples = len(dataset)
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = torch.randperm(total_samples).tolist()
    train_size = int(0.8 * total_samples)
    split_data = {"seed": seed, "total_samples": total_samples,
                  "train_indices": indices[:train_size], "val_indices": indices[train_size:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split_indices.json"
    split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    full = OxfordIIITPet(root=str(data_root), split="trainval", download=download, transform=PET_TRANSFORM)
    train_set = Subset(full, split_data["train_indices"])
    val_set = Subset(full, split_data["val_indices"])
    test_set = OxfordIIITPet(root=str(data_root), split="test", download=download, transform=PET_TRANSFORM)
    val_manifest = generate_val_manifest(len(val_set), seed)
    test_manifest = generate_test_manifest(len(test_set), seed)
    (out_dir / "val_manifest.json").write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")
    (out_dir / "test_manifest.json").write_text(json.dumps(test_manifest, indent=2), encoding="utf-8")
    return split_data, val_manifest, test_manifest


class CorruptedPetTrainDataset(Dataset):
    def __init__(self, base_subset):
        self.base_subset = base_subset

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, idx):
        clean_img, _ = self.base_subset[idx]
        if not torch.is_tensor(clean_img):
            clean_img = TF.to_image(clean_img.convert("RGB"))
            clean_img = TF.resize(clean_img, [128, 128])
            clean_img = TF.to_dtype(clean_img, torch.float32, scale=True)
        label = random.randint(0, 3)
        condition = CORRUPTION_TYPES[label]
        if condition == "clean":
            corrupted_img = clean_img.clone()
        elif condition == "salt":
            corrupted_img, _ = apply_salt_and_pepper(clean_img)
        elif condition == "blur":
            corrupted_img, _ = apply_gaussian_blur(clean_img)
        else:
            corrupted_img, _ = apply_occlusion(clean_img)
        return corrupted_img, clean_img, label


class ManifestPetDataset(Dataset):
    def __init__(self, base_dataset, manifest):
        self.base, self.manifest = base_dataset, manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        clean, _ = self.base[entry["index"]]
        condition = entry["condition"]
        if condition == "clean":
            x = clean.clone()
        elif condition == "salt":
            x, _ = apply_salt_and_pepper(clean, p=entry["p"], seed=entry["seed"])
        elif condition == "blur":
            x, _ = apply_gaussian_blur(clean, entry["kernel_size"], entry["sigma"])
        else:
            x, _ = apply_occlusion(clean, rect_coords=entry["rect_coords"])
        return x, clean, entry["label"]


class LabeledTrainDataset(Dataset):
    def __init__(self, base, allowed=(0, 1, 2, 3)):
        self.base, self.allowed = base, list(allowed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        idx, label = key if isinstance(key, tuple) else (key, random.choice(self.allowed))
        clean, _ = self.base[idx]
        condition = CORRUPTION_TYPES[label]
        if condition == "clean":
            x = clean.clone()
        elif condition == "salt":
            x, _ = apply_salt_and_pepper(clean)
        elif condition == "blur":
            x, _ = apply_gaussian_blur(clean)
        else:
            x, _ = apply_occlusion(clean)
        return x, clean, label


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, n, batch_size):
        assert batch_size % 4 == 0
        self.n, self.bs = n, batch_size

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        perm = torch.randperm(self.n).tolist()
        for b in range(len(self)):
            labels = [0, 1, 2, 3] * (self.bs // 4)
            random.shuffle(labels)
            yield list(zip(perm[b*self.bs:(b+1)*self.bs], labels))


class CondSubset(Dataset):
    def __init__(self, dataset, manifest, conditions):
        self.dataset = dataset
        self.ids = [i for i, entry in enumerate(manifest) if entry["condition"] in conditions]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.dataset[self.ids[i]]


def load_pet_bundle(data_root: Path, out_dir: Path, download: bool = False):
    split_path = out_dir / "split_indices.json"
    val_path, test_path = out_dir / "val_manifest.json", out_dir / "test_manifest.json"
    if not (split_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError("Pet split/manifests are missing; run scripts/prepare_data.py --dataset pets first")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    val_manifest = json.loads(val_path.read_text(encoding="utf-8"))
    test_manifest = json.loads(test_path.read_text(encoding="utf-8"))
    full = OxfordIIITPet(root=str(data_root), split="trainval", download=download, transform=PET_TRANSFORM)
    train_base = Subset(full, splits["train_indices"])
    val_base = Subset(full, splits["val_indices"])
    test_base = OxfordIIITPet(root=str(data_root), split="test", download=download, transform=PET_TRANSFORM)
    return {
        "train_base": train_base,
        "val_base": val_base,
        "train": CorruptedPetTrainDataset(train_base),
        "val": ManifestPetDataset(val_base, val_manifest),
        "test": ManifestPetDataset(test_base, test_manifest),
        "test_base": test_base,
        "val_manifest": val_manifest,
        "test_manifest": test_manifest,
        "splits": splits,
    }
