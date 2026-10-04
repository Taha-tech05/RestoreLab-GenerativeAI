"""Report SoftMoE restoration metrics and gate usage by corruption and severity."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common.data_pets import load_pet_bundle
from common.metrics import per_image
from common.models import build_classifier, build_moe, build_restorer, SPEC_NAMES
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.utils import get_device


def main():
    args = add_common_args(argparse.ArgumentParser(), "task3.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); device = get_device()
    bundle = load_pet_bundle(Path(paths["data_root"]), Path(paths["out_dir"]))
    ck = torch.load(Path(paths["ckpt_dir"]) / "task3_soft_moe.pth", map_location=device)
    gate = build_classifier(ck["gate_cfg"]).to(device); gate.load_state_dict({k.removeprefix("gate."):v for k,v in ck["state_dict"].items() if k.startswith("gate.")})
    experts = {}
    for label, name in SPEC_NAMES.items():
        sc = torch.load(Path(paths["ckpt_dir"]) / f"task2_spec_{name}.pth", map_location=device)
        experts[label] = build_restorer(sc["cfg"]).to(device); experts[label].load_state_dict(sc["state_dict"])
    model = build_moe(gate, experts, ck["cfg"]["tau"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    loader = DataLoader(bundle["test"], batch_size=64, shuffle=False, num_workers=2)
    rows=[]; index=0
    with torch.no_grad():
        for x,y,labels in loader:
            pred, weights = model(x.to(device), True); l1, psnr, ssim = per_image(pred, y.to(device))
            for i in range(x.size(0)):
                e=bundle["test_manifest"][index]; index+=1
                rows.append({"condition":e["condition"],"severity":e["severity"],"l1":l1[i].item(),"psnr":psnr[i].item(),"ssim":ssim[i].item(),
                    **{f"weight_{n}":weights[i,j].item() for j,n in enumerate(("identity","salt","blur","occlusion"))}})
    df=pd.DataFrame(rows); df.to_csv(Path(paths["out_dir"])/"task3_test_per_image.csv",index=False)
    summary=df.groupby(["condition","severity"]).mean(numeric_only=True); summary.to_csv(Path(paths["out_dir"])/"task3_gate_by_condition_severity.csv")
    cols=[f"weight_{n}" for n in ("identity","salt","blur","occlusion")]
    plt.figure(figsize=(10,6)); plt.imshow(summary[cols].to_numpy(),aspect="auto",vmin=0,vmax=1,cmap="viridis"); plt.colorbar(label="mean gate weight")
    plt.yticks(range(len(summary)),[f"{a}/{b}" for a,b in summary.index]); plt.xticks(range(4),["Identity","Salt","Blur","Occlusion"]); plt.tight_layout()
    plt.savefig(Path(paths["out_dir"])/"task3_routing_heatmap.png",dpi=180); plt.close()
    mean=df[cols].mean(); top=mean.idxmax(); inactive=mean[mean<0.01].index.tolist()
    print("mean weights:",mean.to_dict(),"dominant:",top,"inactive (<0.01):",inactive)


if __name__ == "__main__": main()
