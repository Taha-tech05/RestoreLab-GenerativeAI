"""Measure FS2K validation/test quality and save fixed-photo sample grids."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common.data_fs2k import build_fs2k_datasets
from common.models import Generator
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import evaluate_generator
from common.utils import get_device


def main():
    args=add_common_args(argparse.ArgumentParser(),"task4.yaml").parse_args(); paths=resolve_paths(args); ensure_output_dirs(paths); device=get_device()
    cfg=paths["config"]["final"]; bundle=build_fs2k_datasets(Path(paths["data_root"]),Path(paths["out_dir"])/"fs2k_split.json")
    model=Generator(cfg["c"],cfg["emb"],cfg["drop"]).to(device)
    model.load_state_dict(torch.load(Path(paths["ckpt_dir"])/"face2sketch_generator.pth",map_location=device)); model.eval()
    for split in ("val","test"):
        loader=DataLoader(bundle[split],batch_size=32,shuffle=False)
        l1,ssim=evaluate_generator(model,loader,device)
        print(split,"L1:",l1,"SSIM:",ssim)
        by_style={}
        for style in range(3):
            subset=[i for i,record in enumerate(bundle[split].records) if record["style"]==style]
            if subset:
                from torch.utils.data import Subset
                by_style[f"Style {style+1}"]=evaluate_generator(model,DataLoader(Subset(bundle[split],subset),batch_size=32),device)
        print(split,"per-style (L1, SSIM):",by_style)
    photos,sketches,styles=next(iter(DataLoader(bundle["test"],batch_size=9,shuffle=False)))
    with torch.no_grad(): generated=model(photos.to(device),styles.to(device)).cpu()
    fig,axes=plt.subplots(9,3,figsize=(9,22))
    for i in range(9):
        for j,tensor in enumerate((photos[i],sketches[i],generated[i])):
            axes[i,j].imshow(((tensor.permute(1,2,0)+1)/2).clamp(0,1)); axes[i,j].axis("off")
            if i==0: axes[i,j].set_title(("Photo","Target","Generated")[j])
    plt.tight_layout(); plt.savefig(Path(paths["out_dir"])/"task4_fixed_test_grid.png",dpi=180); plt.close(fig)


if __name__=="__main__": main()
