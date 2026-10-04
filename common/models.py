"""Neural architectures from the original task notebooks."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalDAE(nn.Module):
    """128→64→32→16→8 encoder, linear latent bottleneck, mirrored decoder."""
    def __init__(self, base_ch=64, latent_dim=256, dropout=0.1):
        super().__init__()
        c = base_ch
        chs = [3, c, c*2, c*4, c*8]
        enc = []
        for i in range(4):
            enc += [nn.Conv2d(chs[i], chs[i+1], 4, 2, 1),
                    nn.BatchNorm2d(chs[i+1]), nn.LeakyReLU(0.2, True)]
        self.encoder = nn.Sequential(*enc)
        self.flat = c*8*8*8
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.flat, latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, self.flat), nn.LeakyReLU(0.2, True))
        dec = []
        for i in range(4, 0, -1):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(chs[i], chs[i-1] if i > 1 else c, 3, 1, 1),
                    nn.BatchNorm2d(chs[i-1] if i > 1 else c), nn.ReLU(True)]
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Sequential(nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())
        self.c8 = c*8

    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        h = self.from_latent(z).view(-1, self.c8, 8, 8)
        return self.out(self.decoder(h))


class SpatialDAE(UniversalDAE):
    def __init__(self, base_ch=48, latent_ch=64, dropout=0.07):
        super().__init__(base_ch, 128, dropout)
        c8 = base_ch * 8
        self.to_latent = nn.Sequential(nn.Conv2d(c8, latent_ch, 1), nn.Dropout2d(dropout))
        self.from_latent = nn.Sequential(nn.Conv2d(latent_ch, c8, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        h = self.from_latent(self.to_latent(self.encoder(x)))
        return self.out(self.decoder(h))


def build_restorer(cfg):
    if cfg.get("arch") == "spatial":
        return SpatialDAE(cfg["base_ch"], cfg["latent_ch"], cfg["dropout"])
    return UniversalDAE(cfg["base_ch"], cfg["latent_dim"], cfg["dropout"])


class CorruptionCNN(nn.Module):
    def __init__(self, chs=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, cin = [], 3
        for c in chs:
            layers += [nn.Conv2d(cin, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(True),
                       nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, 4))

    def forward(self, x):
        return self.head(self.features(x))


CH_CFGS = {"s": (16, 32, 64, 128), "m": (32, 64, 128, 256),
           "l": (64, 128, 256, 512), "deep": (32, 64, 128, 256, 512)}
SPEC_NAMES = {1: "salt", 2: "blur", 3: "occlusion"}


def build_classifier(cfg):
    return CorruptionCNN(CH_CFGS[cfg["ch"]], cfg["dropout"])


class SoftMoE(nn.Module):
    def __init__(self, gate, experts, tau=1.0):
        super().__init__()
        self.gate, self.experts, self.tau = gate, nn.ModuleList(experts), tau

    def forward(self, x, full=False):
        weights = F.softmax(self.gate(x) / self.tau, 1)
        outputs = [x] + [expert(x) for expert in self.experts]
        restored = sum(weights[:, k].view(-1, 1, 1, 1) * outputs[k] for k in range(4))
        return (restored, weights) if full else restored


def build_moe(gate, specialists, tau):
    return SoftMoE(copy.deepcopy(gate), [copy.deepcopy(specialists[i]) for i in (1, 2, 3)], tau)


def down(i, o, bn=True):
    layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(o))
    return nn.Sequential(*layers, nn.LeakyReLU(0.2, True))


def up(i, o, drop):
    layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o)]
    if drop > 0:
        layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers, nn.ReLU(True))


class Generator(nn.Module):
    def __init__(self, c=64, emb=16, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        self.d1 = down(3 + emb, c, False)
        self.d2 = down(c, c*2)
        self.d3 = down(c*2, c*4)
        self.d4 = down(c*4, c*8)
        self.d5 = down(c*8, c*8)
        self.d6 = down(c*8, c*8)
        self.u1 = up(c*8 + emb, c*8, drop)
        self.u2 = up(c*16, c*8, drop)
        self.u3 = up(c*16, c*4, 0)
        self.u4 = up(c*8, c*2, 0)
        self.u5 = up(c*4, c, 0)
        self.out = nn.Sequential(nn.ConvTranspose2d(c*2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        embedding = self.emb(style)
        full_map = embedding[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        d1 = self.d1(torch.cat([x, full_map], 1)); d2 = self.d2(d1); d3 = self.d3(d2)
        d4 = self.d4(d3); d5 = self.d5(d4); d6 = self.d6(d5)
        bottleneck_map = embedding[:, :, None, None].expand(-1, -1, d6.size(2), d6.size(3))
        u1 = self.u1(torch.cat([d6, bottleneck_map], 1))
        u2 = self.u2(torch.cat([u1, d5], 1)); u3 = self.u3(torch.cat([u2, d4], 1))
        u4 = self.u4(torch.cat([u3, d3], 1)); u5 = self.u5(torch.cat([u4, d2], 1))
        return self.out(torch.cat([u5, d1], 1))


class PatchD(nn.Module):
    def __init__(self, c=64, emb=16):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        self.net = nn.Sequential(
            down(6 + emb, c, False), down(c, c*2), down(c*2, c*4),
            nn.Conv2d(c*4, c*8, 4, 1, 1, bias=False), nn.BatchNorm2d(c*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c*8, 1, 4, 1, 1))

    def forward(self, x, y, style):
        style_map = self.emb(style)[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        return self.net(torch.cat([x, y, style_map], 1))


def winit(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
