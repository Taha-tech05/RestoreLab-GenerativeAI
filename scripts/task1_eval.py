"""Task 1 per-corruption/severity metrics, error maps, and failures."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common.data_pets import load_pet_bundle
from common.metrics import per_image
from common.models import build_restorer
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.utils import get_device


def main():
    args = add_common_args(argparse.ArgumentParser(), "task1.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); device = get_device()
    bundle = load_pet_bundle(Path(paths["data_root"]), Path(paths["out_dir"]))
    ckpt = torch.load(Path(paths["ckpt_dir"]) / "task1_spatial_best.pth", map_location=device)
    model = build_restorer(ckpt["cfg"]).to(device); model.load_state_dict(ckpt["state_dict"]); model.eval()
    loader = DataLoader(bundle["test"], batch_size=128, shuffle=False, num_workers=2)
    rows, index = [], 0
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)); l1, psnr, ssim = per_image(pred, y.to(device))
            l1_in, psnr_in, ssim_in = per_image(x.to(device), y.to(device))
            for k in range(x.size(0)):
                entry = bundle["test_manifest"][index]; index += 1
                rows.append(dict(condition=entry["condition"], severity=entry["severity"],
                                 l1=l1[k].item(), psnr=psnr[k].item(), ssim=ssim[k].item(),
                                 l1_in=l1_in[k].item(), psnr_in=psnr_in[k].item(), ssim_in=ssim_in[k].item()))
    df = pd.DataFrame(rows); df.loc[df.condition == "clean", "psnr_in"] = float("nan")
    condition_order = {"clean": 0, "salt": 1, "blur": 2, "occlusion": 3}
    severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    table = df.groupby(["condition", "severity"]).mean(numeric_only=True).round(4).reset_index()
    table["c"] = table.condition.map(condition_order); table["s"] = table.severity.map(severity_order)
    table = table.sort_values(["c", "s"]).drop(columns=["c", "s"]).set_index(["condition", "severity"])
    table.to_csv(Path(paths["out_dir"]) / "task1_test_results.csv")
    df.to_csv(Path(paths["out_dir"]) / "task1_test_per_image.csv", index=False)
    def draw(indices, filename):
        fig, axes = plt.subplots(len(indices), 4, figsize=(8, 2*len(indices)))
        for row, j in enumerate(indices):
            x, y, _ = bundle["test"][j]
            with torch.no_grad(): pred = model(x.unsqueeze(0).to(device)).cpu()[0].clamp(0, 1)
            err = (pred-y).abs().mean(0); entry = bundle["test_manifest"][j]
            items = [(y.permute(1,2,0), "clean target", None), (x.permute(1,2,0), f"{entry['condition']}/{entry['severity']}", None),
                     (pred.permute(1,2,0), f"restored {df.psnr[j]:.1f} dB", None), (err, "abs error", "inferno")]
            for col, (im, title, cmap) in enumerate(items):
                axes[row, col].imshow(im, cmap=cmap); axes[row, col].set_title(title, fontsize=8); axes[row, col].axis("off")
        plt.tight_layout(); plt.savefig(Path(paths["out_dir"]) / filename, dpi=200); plt.close(fig)
    combos = [("clean", "none")] + [(c, s) for c in ("salt", "blur", "occlusion") for s in ("low", "medium", "high")]
    selected = []
    for cond, sev in combos:
        group = df[(df.condition == cond) & (df.severity == sev)]
        selected.append((group.psnr-group.psnr.median()).abs().sort_values().index[0])
    draw(selected[:6], "task1_examples_1.png"); draw(selected[6:], "task1_examples_2.png")
    failures = [df[(df.condition == "occlusion") & (df.severity == "high")].psnr.idxmin(),
                df[(df.condition == "salt") & (df.severity == "high")].psnr.idxmin(),
                (df[df.condition == "blur"].psnr-df[df.condition == "blur"].psnr_in).idxmin(),
                df[df.condition == "clean"].psnr.idxmin()]
    draw(failures, "task1_failures.png")
    print(table)


if __name__ == "__main__": main()
