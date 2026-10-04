"""Train salt, blur, and occlusion restoration specialists independently."""
import argparse
from pathlib import Path
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_specialist
from common.utils import seed_everything, save_checkpoint, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task2.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    final = cfg["final"]["specialists"]
    if "from_study" in final:
        study = optuna.load_study(study_name=final["from_study"], storage=f"sqlite:///{paths['out_dir']}/task2_optuna.db")
        hp = {**study.best_params, "arch": "spatial"}
    else: hp = dict(final)
    epochs = final.get("epochs", 40); seed_everything(cfg.get("seed", 42))
    for name in ("salt", "blur", "occlusion"):
        run = wandb_start(project="genai-ass1", group="task2-final", name=f"t2-spec-{name}",
                          config={**hp, "epochs": epochs, "cond": name}, reinit=True)
        try: _, best, state = train_specialist(hp, paths, name, epochs)
        finally: run.finish()
        save_checkpoint(Path(paths["ckpt_dir"]) / f"task2_spec_{name}.pth", {"cfg": hp, "state_dict": state})
        print(name, "best val_score:", best)


if __name__ == "__main__": main()
