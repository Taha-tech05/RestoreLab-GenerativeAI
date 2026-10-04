"""Notebook losses shared across restoration tasks."""
import torch
import torch.nn.functional as F


def gauss_win(size=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.exp(-0.5 * ((torch.arange(size) - size//2) / sigma) ** 2)
    g = (g / g.sum()).to(device)
    return torch.outer(g, g).expand(ch, 1, size, size).contiguous()


def ssim(x, y, C1=0.01**2, C2=0.03**2):
    window = gauss_win(device=x.device)
    conv = lambda tensor: F.conv2d(tensor, window, padding=5, groups=3)
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x*x) - mx**2, conv(y*y) - my**2, conv(x*y) - mx*my
    score = ((2*mx*my + C1) * (2*sxy + C2)) / ((mx**2 + my**2 + C1) * (sxx + syy + C2))
    return score.mean()


def restoration_loss(pred, target, alpha):
    return alpha * F.l1_loss(pred, target) + (1 - alpha) * (1 - ssim(pred, target))


def balance_loss(weights):
    return ((weights.mean(0) - 0.25) ** 2).sum()


def moe_loss(pred, target, weights, label, hp):
    l1, structural = F.l1_loss(pred, target), 1 - ssim(pred, target)
    ce = F.nll_loss(torch.log(weights.clamp_min(1e-8)), label)
    balance = balance_loss(weights)
    return hp["w_l1"]*l1 + (1 - hp["w_l1"])*structural + hp["w_ce"]*ce + hp["w_bal"]*balance
