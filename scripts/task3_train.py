"""Train the initialized SoftMoE using frozen warm-up then joint fine-tuning."""
import argparse
import json
from pathlib import Path
import torch
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_moe
from common.utils import save_checkpoint, seed_everything, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task3.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]; hp = cfg["final"].copy()
    study_name = hp.pop("from_study", None)
    if study_name:
        import optuna
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{paths['out_dir']}/{study_name.replace('task3_moe','task3')}_optuna.db")
        hp.update(study.best_params)
    warm, joint = hp.pop("warm_epochs", 3), hp.pop("joint_epochs", 20)
    seed_everything(cfg.get("seed", 42))
    run = wandb_start(project="genai-ass1", group="task3-final", name="t3-soft-moe", config={**hp, "warm_epochs": warm, "joint_epochs": joint}, reinit=True)
    try: model, score, state, gate_cfg, expert_cfg = train_moe(hp, paths, warm, joint)
    finally: run.finish()
    payload = {"cfg": hp, "gate_cfg": gate_cfg, "expert_cfg": expert_cfg, "state_dict": state, "best": score}
    save_checkpoint(Path(paths["ckpt_dir"]) / "task3_soft_moe.pth", payload)
    (Path(paths["ckpt_dir"]) / "task3_soft_moe.json").write_text(json.dumps({"cfg": hp, "gate_cfg": gate_cfg, "expert_cfg": expert_cfg}, indent=2), encoding="utf-8")
    print("best val_score:", score)


if __name__ == "__main__": main()
