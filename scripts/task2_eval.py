"""Evaluate classifier and hard expert routing on the fixed Pet test manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.data_pets import load_pet_bundle
from common.metrics import classification_metrics
from common.models import SPEC_NAMES, build_classifier, build_restorer
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.utils import get_device


def main():
    args = add_common_args(argparse.ArgumentParser(), "task2.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); device = get_device()
    bundle = load_pet_bundle(Path(paths["data_root"]), Path(paths["out_dir"]))
    cls_ck = torch.load(Path(paths["ckpt_dir"]) / "task2_classifier.pth", map_location=device)
    classifier = build_classifier(cls_ck["cfg"]).to(device)
    classifier.load_state_dict(cls_ck["state_dict"]); classifier.eval()
    experts = {}
    for label, name in SPEC_NAMES.items():
        ck = torch.load(Path(paths["ckpt_dir"]) / f"task2_spec_{name}.pth", map_location=device)
        experts[label] = build_restorer(ck["cfg"]).to(device)
        experts[label].load_state_dict(ck["state_dict"]); experts[label].eval()
    loader = DataLoader(bundle["test"], batch_size=64, shuffle=False, num_workers=2)
    names = ["clean", "salt", "blur", "occlusion"]
    rows, probs_all, labels_all, pred_all = [], [], [], []
    with torch.no_grad():
        for x, y, label in loader:
            x, y, label = x.to(device), y.to(device), label.to(device)
            probs = classifier(x).softmax(1); pred = probs.argmax(1)
            probs_all.extend(probs.cpu().numpy()); labels_all.extend(label.cpu().numpy()); pred_all.extend(pred.cpu().numpy())
            for i in range(x.size(0)):
                target = int(label[i]); chosen = int(pred[i])
                oracle_out = x[i:i+1] if target == 0 else experts[target](x[i:i+1])
                predicted_out = x[i:i+1] if chosen == 0 else experts[chosen](x[i:i+1])
                rows.append({"true": names[target], "predicted": names[chosen],
                             "oracle_expert": "identity" if target == 0 else names[target],
                             "predicted_expert": "identity" if chosen == 0 else names[chosen],
                             "oracle_l1": (oracle_out-y[i:i+1]).abs().mean().item(),
                             "predicted_l1": (predicted_out-y[i:i+1]).abs().mean().item()})
    metrics, matrix = classification_metrics(np.asarray(labels_all), np.asarray(pred_all))
    (Path(paths["out_dir"]) / "task2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=names, columns=names).to_csv(Path(paths["out_dir"]) / "task2_confusion_normalized.csv")
    pd.DataFrame(probs_all, columns=[f"prob_{n}" for n in names]).assign(
        true=np.asarray(labels_all), predicted=np.asarray(pred_all)).to_csv(Path(paths["out_dir"]) / "task2_probabilities.csv", index=False)
    pd.DataFrame(rows).groupby("true").mean(numeric_only=True).to_csv(Path(paths["out_dir"]) / "task2_routing_oracle_vs_predicted.csv")
    print(metrics); print("normalized confusion matrix:\n", matrix)


if __name__ == "__main__": main()
