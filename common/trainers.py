"""Task training loops transcribed from the original notebooks."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import wandb
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torchvision.utils as vu

from .data_pets import (BalancedBatchSampler, CondSubset, LabeledTrainDataset,
                        load_pet_bundle)
from .data_fs2k import build_fs2k_datasets, seed_all
from .losses import moe_loss, restoration_loss, ssim
from .metrics import ssim_per_image
from .models import (CH_CFGS, SPEC_NAMES, CorruptionCNN, Generator, PatchD,
                     SoftMoE, build_classifier, build_moe, build_restorer, winit)
from .utils import get_device, seed_everything, save_checkpoint, wandb_start


def _pets(paths):
    pet_root = Path(paths["data_root"]) / Path(paths.get("config", {}).get("paths", {}).get("pet_subdir", "."))
    return load_pet_bundle(pet_root, Path(paths["out_dir"]))


@torch.no_grad()
def evaluate_restorer(model, loader, device):
    model.eval()
    l1 = ss = ps = n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x); batch = x.size(0)
        l1 += F.l1_loss(pred, y).item()*batch
        ss += ssim(pred, y).item()*batch
        mse = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2, 3))
        ps += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()*batch
        n += batch
    return {"val_l1": l1/n, "val_ssim": ss/n, "val_psnr": ps/n,
            "val_score": l1/n + (1 - ss/n)}


def train_restorer(cfg, paths, epochs, trial=None, log=True, train_data=None, val_data=None):
    device = get_device()
    if train_data is None:
        bundle = _pets(paths); train_data, val_data = bundle["train"], bundle["val"]
    model = build_restorer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)
    best, best_state = 1e9, None
    for epoch in range(epochs):
        model.train(); running = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = restoration_loss(model(x), y, cfg["alpha"])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); running += loss.item()
        scheduler.step()
        metrics = evaluate_restorer(model, val_loader, device)
        metrics.update(train_loss=running/len(train_loader), epoch=epoch)
        if log:
            wandb.log(metrics)
        if metrics["val_score"] < best:
            best = metrics["val_score"]
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        if trial:
            trial.report(metrics["val_score"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return model, best, best_state


@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval(); predictions, labels = [], []
    for x, _, y in loader:
        predictions.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    pred, target = torch.cat(predictions).numpy(), torch.cat(labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average="macro", zero_division=0)
    return {"val_acc": accuracy_score(target, pred), "val_prec": precision,
            "val_rec": recall, "val_f1": f1}, pred, target


def train_classifier(cfg, paths, epochs, trial=None, log=True):
    device = get_device(); bundle = _pets(paths)
    train_data, val_data = LabeledTrainDataset(bundle["train_base"]), bundle["val"]
    model = build_classifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_data, batch_sampler=BalancedBatchSampler(len(train_data), cfg["batch_size"]),
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=2)
    best, best_state = -1, None
    for epoch in range(epochs):
        model.train(); running = 0
        for x, _, y in train_loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); running += loss.item()
        scheduler.step()
        metrics, _, _ = eval_classifier(model, val_loader, device)
        metrics.update(train_loss=running/len(train_loader), epoch=epoch)
        if log: wandb.log(metrics)
        if metrics["val_f1"] > best:
            best = metrics["val_f1"]
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        if trial:
            trial.report(metrics["val_f1"], epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
    return model, best, best_state


def train_specialist(cfg, paths, condition, epochs, trial=None, log=True, train_data=None, val_data=None):
    bundle = _pets(paths)
    label = {"salt": 1, "blur": 2, "occlusion": 3}[condition]
    if train_data is None:
        train_data = LabeledTrainDataset(bundle["train_base"], allowed=(1, 2, 3))
    if val_data is None:
        val_data = CondSubset(bundle["val"], bundle["val_manifest"], {condition})
    return train_restorer({**cfg, "arch": "spatial"}, paths, epochs, trial, log, train_data, val_data)


def train_specialists(cfg, paths, epochs=40):
    device = get_device(); bundle = _pets(paths)

    class SingleCondTrain(torch.utils.data.Dataset):
        def __init__(self, base, label): self.dataset, self.label = LabeledTrainDataset(base), label
        def __len__(self): return len(self.dataset)
        def __getitem__(self, i): return self.dataset[(i, self.label)]

    result = {}
    for label, name in SPEC_NAMES.items():
        run = wandb_start(project="genai-ass1", group="task2-final", name=f"t2-spec-{name}",
                          config={**cfg, "epochs": epochs, "cond": name}, reinit=True)
        train_data = SingleCondTrain(bundle["train_base"], label)
        val_data = CondSubset(bundle["val"], bundle["val_manifest"], {name})
        try:
            _, best, state = train_restorer({**cfg, "arch": "spatial"}, paths, epochs,
                                            train_data=train_data, val_data=val_data)
        finally:
            run.finish()
        result[name] = {"cfg": {**cfg, "arch": "spatial"}, "state_dict": state, "best": best}
    return result


def train_moe(hp, paths, warm_epochs, joint_epochs, trial=None, log=True):
    device = get_device(); bundle = _pets(paths)
    classifier_ckpt = torch.load(Path(paths["ckpt_dir"]) / "task2_classifier.pth", map_location=device)
    gate = build_classifier(classifier_ckpt["cfg"]).to(device)
    gate.load_state_dict(classifier_ckpt["state_dict"])
    specialists = {}
    spec_cfg = None
    for label, name in SPEC_NAMES.items():
        checkpoint = torch.load(Path(paths["ckpt_dir"]) / f"task2_spec_{name}.pth", map_location=device)
        spec_cfg = checkpoint["cfg"]
        specialists[label] = build_restorer(checkpoint["cfg"]).to(device)
        specialists[label].load_state_dict(checkpoint["state_dict"])
    model = build_moe(gate, specialists, hp["tau"]).to(device)
    train_data = LabeledTrainDataset(bundle["train_base"])
    train_loader = DataLoader(train_data, batch_sampler=BalancedBatchSampler(len(train_data), hp["batch_size"]),
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(bundle["val"], batch_size=64, num_workers=2)
    best, best_state, step = 1e9, None, 0

    @torch.no_grad()
    def evaluate_moe():
        model.eval(); l1 = ss = n = 0; weights_all, labels_all = [], []
        for x, y, labels in val_loader:
            x, y = x.to(device), y.to(device)
            pred, weights = model(x, True); batch = x.size(0)
            l1 += F.l1_loss(pred, y).item()*batch; ss += ssim(pred, y).item()*batch; n += batch
            weights_all.append(weights.cpu()); labels_all.append(labels)
        weights_all, labels_all = torch.cat(weights_all), torch.cat(labels_all)
        avg = weights_all.mean(0)
        return {"val_l1": l1/n, "val_ssim": ss/n, "val_score": l1/n + 1 - ss/n,
                "w_max": avg.max().item(), "w_min": avg.min().item(),
                "route_acc": (weights_all.argmax(1) == labels_all).float().mean().item()}

    def run_epoch(optimizer, train_experts):
        model.train()
        if not train_experts: model.experts.eval()
        total = 0
        for x, y, label in train_loader:
            x, y, label = x.to(device), y.to(device), label.to(device)
            pred, weights = model(x, True)
            loss = moe_loss(pred, y, weights, label, hp)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
        return total / len(train_loader)

    for parameter in model.experts.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=hp["warm_lr"])
    for _ in range(warm_epochs):
        train_loss = run_epoch(optimizer, False); metrics = evaluate_moe()
        metrics.update(stage=0, train_loss=train_loss, epoch=step)
        if log: wandb.log(metrics)
        step += 1
    for parameter in model.experts.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(joint_epochs, 1))
    for _ in range(joint_epochs):
        train_loss = run_epoch(optimizer, True); scheduler.step(); metrics = evaluate_moe()
        metrics.update(stage=1, train_loss=train_loss, epoch=step)
        if log: wandb.log(metrics)
        if metrics["val_score"] < best:
            best = metrics["val_score"]
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        if trial:
            trial.report(metrics["val_score"], step)
            if metrics["w_max"] > 0.7: raise optuna.TrialPruned()
            if trial.should_prune(): raise optuna.TrialPruned()
        step += 1
    return model, best, best_state, classifier_ckpt["cfg"], spec_cfg


def evaluate_generator(generator, loader, device, image_size=128):
    generator.eval(); l1_sum = ss_sum = n = 0
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = generator(photo, style); batch = photo.size(0)
        l1_sum += F.l1_loss(fake, sketch, reduction="sum").item() / (3*image_size*image_size)
        ss_sum += ssim_fn((fake+1)/2, (sketch+1)/2, data_range=1.0, reduction="sum").item()
        n += batch
    generator.train()
    return l1_sum/n, ss_sum/n


def train_cgan(cfg, data_root, out_dir, epochs, log=False, trial=None, log_every=10, save=None):
    device = get_device(); seed_all(42)
    bundle = build_fs2k_datasets(Path(data_root), Path(out_dir) / "fs2k_split.json", seed=42)
    train_data, val_data = bundle["train"], bundle["val"]
    fixed = next(iter(DataLoader(val_data, batch_size=9, shuffle=False)))
    generator = Generator(cfg["c"], cfg["emb"], cfg["drop"]).to(device)
    discriminator = PatchD(cfg["c"], cfg["emb"]).to(device)
    generator.apply(winit); discriminator.apply(winit)
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), cfg["lr_d"], betas=(0.5, 0.999))
    train_loader = DataLoader(train_data, cfg["bs"], shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_data, 32); best = -1
    for epoch in range(1, epochs+1):
        aggregate = np.zeros(4); batches = 0
        for photo, sketch, style in train_loader:
            photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
            fake = generator(photo, style)
            pred_real, pred_fake = discriminator(photo, sketch, style), discriminator(photo, fake.detach(), style)
            d_real = bce(pred_real, torch.ones_like(pred_real)); d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            opt_d.zero_grad(); (0.5*(d_real+d_fake)).backward(); opt_d.step()
            pred_fake = discriminator(photo, fake, style)
            g_adv = bce(pred_fake, torch.ones_like(pred_fake)); g_l1 = F.l1_loss(fake, sketch)
            opt_g.zero_grad(); (g_adv + cfg["lam"]*g_l1).backward(); opt_g.step()
            aggregate += [d_real.item(), d_fake.item(), g_adv.item(), g_l1.item()]; batches += 1
        aggregate /= batches; val_l1, val_ssim = evaluate_generator(generator, val_loader, device)
        if log:
            wandb.log({"D_real": aggregate[0], "D_fake": aggregate[1], "G_adv": aggregate[2],
                       "G_L1": aggregate[3], "val_L1": val_l1, "val_SSIM": val_ssim}, step=epoch)
            if epoch % log_every == 0 or epoch == 1:
                with torch.no_grad():
                    generator.eval(); fake = generator(fixed[0].to(device), fixed[2].to(device)); generator.train()
                grid = vu.make_grid(torch.cat([fixed[0].to(device), fixed[1].to(device), fake]),
                                    nrow=9, normalize=True, value_range=(-1, 1))
                wandb.log({"samples (photo/GT/gen)": wandb.Image(grid)}, step=epoch)
        score = val_ssim - val_l1
        if save and score > best:
            best = score; Path(save).parent.mkdir(parents=True, exist_ok=True); torch.save(generator.state_dict(), save)
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
    return generator, val_l1, val_ssim, bundle
