"""Train the selected Task 1 spatial-bottleneck restoration model."""
import argparse
from pathlib import Path
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_restorer
from common.utils import seed_everything, save_checkpoint, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task1.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    hp = cfg["final"].copy(); epochs = hp.pop("epochs", 60)
    seed_everything(cfg.get("seed", 42))
    run = wandb_start(project="genai-ass1", group="task1-final", name="t1-final-spatial",
                      config={**hp, "epochs": epochs}, reinit=True)
    try: _, best, state = train_restorer(hp, paths, epochs=epochs)
    finally: run.finish()
    save_checkpoint(Path(paths["ckpt_dir"]) / "task1_spatial_best.pth", {"cfg": hp, "state_dict": state})
    print("best val_score:", best)


if __name__ == "__main__": main()
