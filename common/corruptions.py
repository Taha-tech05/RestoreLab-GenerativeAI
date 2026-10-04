"""Oxford-IIIT Pet corruptions copied from notebook source cells 3 and 6."""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

CORRUPTION_TYPES = ["clean", "salt", "blur", "occlusion"]
TEST_SEVERITIES = {
    "salt": [0.03, 0.08, 0.15],
    "blur": [(3, 0.7), (5, 1.5), (7, 2.5)],
    "occlusion": [(1, 0.10), (2, 0.20), (3, 0.35)],
}
SEV_NAMES = ["low", "medium", "high"]


def apply_salt_and_pepper(img_tensor, p=None, seed=None):
    corrupted = img_tensor.clone()
    if p is None:
        p = random.uniform(0.02, 0.15)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    _, H, W = corrupted.shape
    n = int(p * H * W)
    ys = torch.randint(0, H, (n,), generator=g)
    xs = torch.randint(0, W, (n,), generator=g)
    vals = (torch.rand(n, generator=g) < 0.5).float()
    for c in range(3):
        corrupted[c, ys, xs] = vals
    return corrupted, {"p": p, "seed": seed}


def sample_occlusion_coords(rng, num_rects, target_coverage, H=128, W=128, tol=0.02, max_tries=50):
    """Sample rectangles until their union coverage is within tolerance."""
    best, best_err = None, 1e9
    for _ in range(max_tries):
        area_per = target_coverage * H * W / num_rects
        mask = torch.zeros(H, W, dtype=torch.bool)
        coords = []
        for _ in range(num_rects):
            ar = rng.uniform(0.5, 2.0)
            h = max(1, min(H, int(round(math.sqrt(area_per / ar)))))
            w = max(1, min(W, int(round(area_per / h))))
            top, left = rng.randint(0, H - h), rng.randint(0, W - w)
            mask[top:top+h, left:left+w] = True
            coords.append([top, left, h, w])
        cov = mask.float().mean().item()
        if abs(cov - target_coverage) < best_err:
            best, best_err = (coords, cov), abs(cov - target_coverage)
        if best_err <= tol:
            break
    return best


def apply_occlusion(img_tensor, num_rects=None, target_coverage=None, rect_coords=None):
    corrupted = img_tensor.clone()
    if rect_coords is None:
        num_rects = num_rects or random.randint(1, 3)
        target_coverage = target_coverage or random.uniform(0.10, 0.35)
        rect_coords, _ = sample_occlusion_coords(random, num_rects, target_coverage)
    for top, left, h, w in rect_coords:
        corrupted[:, top:top+h, left:left+w] = 0.0
    return corrupted, {"num_rects": len(rect_coords), "rect_coords": rect_coords}


def apply_gaussian_blur(img_tensor, kernel_size=None, sigma=None):
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    if sigma is None:
        sigma = random.uniform(0.5, 2.5)
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    img_batch = img_tensor.unsqueeze(0)
    img_batch = F.pad(img_batch, [radius] * 4, mode="reflect")
    blurred_batch = F.conv2d(img_batch, kernel_2d, padding=0, groups=3)
    return blurred_batch.squeeze(0), {"kernel_size": kernel_size, "sigma": sigma}


def make_entry(i, cond, rng, seed, severity=None, sev_params=None):
    entry = {"index": i, "label": CORRUPTION_TYPES.index(cond), "condition": cond,
             "severity": severity, "seed": seed}
    if cond == "salt":
        entry["p"] = sev_params if sev_params is not None else rng.uniform(0.02, 0.15)
    elif cond == "blur":
        if sev_params is not None:
            entry["kernel_size"], entry["sigma"] = sev_params
        else:
            entry["kernel_size"], entry["sigma"] = rng.choice([3, 5, 7]), rng.uniform(0.5, 2.5)
    elif cond == "occlusion":
        if sev_params is not None:
            n, cov = sev_params
        else:
            n, cov = rng.randint(1, 3), rng.uniform(0.10, 0.35)
        coords, actual = sample_occlusion_coords(rng, n, cov)
        entry.update(num_rects=n, target_coverage=cov, actual_coverage=actual, rect_coords=coords)
    return entry


def generate_val_manifest(n_images, base_seed):
    manifest = []
    for i in range(n_images):
        seed = base_seed * 10_000_000 + i
        rng = random.Random(seed)
        cond = CORRUPTION_TYPES[rng.randint(0, 3)]
        manifest.append(make_entry(i, cond, rng, seed, severity="random"))
    return manifest


def generate_test_manifest(n_images, base_seed):
    manifest = []
    for i in range(n_images):
        k = 0
        seed = base_seed * 10_000_000 + i * 10 + k
        manifest.append(make_entry(i, "clean", random.Random(seed), seed, severity="none"))
        for cond, params in TEST_SEVERITIES.items():
            for s, sev_params in enumerate(params):
                k += 1
                seed = base_seed * 10_000_000 + i * 10 + k
                manifest.append(make_entry(i, cond, random.Random(seed), seed,
                                           severity=SEV_NAMES[s], sev_params=sev_params))
    return manifest
