"""Metric helpers copied from notebook evaluation cells."""
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .losses import gauss_win
import torch.nn.functional as F


def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean()


def ssim_per_image(x, y, C1=0.01**2, C2=0.03**2):
    window = gauss_win(device=x.device)
    conv = lambda tensor: F.conv2d(tensor, window, padding=5, groups=3)
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x*x) - mx**2, conv(y*y) - my**2, conv(x*y) - mx*my
    score = ((2*mx*my + C1) * (2*sxy + C2)) / ((mx**2 + my**2 + C1) * (sxx + syy + C2))
    return score.mean(dim=(1, 2, 3))


def per_image(pred, target):
    l1 = (pred - target).abs().mean(dim=(1, 2, 3))
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    p = (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).clamp(max=50)
    return l1, p, ssim_per_image(pred, target)


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1}, confusion_matrix(y_true, y_pred, normalize="true")
