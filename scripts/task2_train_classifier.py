"""Train the 4-class balanced-batch corruption classifier."""
import argparse
from pathlib import Path
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_classifier
from common.utils import seed_everything, save_checkpoint, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task2.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    final = cfg["final"]["classifier"]
    if "from_study" in final:
        study = optuna.load_study(study_name=final["from_study"], storage=f"sqlite:///{paths['out_dir']}/task2_optuna.db")
        hp = study.best_params
    else: hp = dict(final)
    epochs = final.get("epochs", 30); seed_everything(cfg.get("seed", 42))
    run = wandb_start(project="genai-ass1", group="task2-final", name="t2-classifier",
                      config={**hp, "epochs": epochs}, reinit=True)
    try: _, best, state = train_classifier(hp, paths, epochs=epochs)
    finally: run.finish()
    save_checkpoint(Path(paths["ckpt_dir"]) / "task2_classifier.pth", {"cfg": hp, "state_dict": state})
    print("best val macro-F1:", best)


if __name__ == "__main__": main()
