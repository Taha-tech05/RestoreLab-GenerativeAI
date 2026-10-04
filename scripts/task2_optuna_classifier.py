"""Task 2 corruption classifier Optuna search."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_classifier
from common.utils import seed_everything, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task2.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    seed_everything(cfg.get("seed", 42)); search = cfg["classifier_search"]
    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
              "ch": trial.suggest_categorical("ch", ["s", "m", "l", "deep"]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.5),
              "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True)}
        run = wandb_start(project="genai-ass1", group="task2-cls-optuna",
                          name=f"t2c-trial{trial.number}", config=hp, reinit=True)
        try:
            _, best, _ = train_classifier(hp, paths, epochs=search["epochs"], trial=trial)
            run.log({"best_val_f1": best}); return best
        finally: run.finish()
    study = optuna.create_study(direction=search["direction"], study_name=search["study_name"],
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=f"sqlite:///{paths['out_dir']}/task2_optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=search["trials"])
    study.trials_dataframe().to_csv(paths["out_dir"] / "task2_cls_optuna_trials.csv", index=False)
    print(study.best_trial.number, study.best_value, study.best_params)


if __name__ == "__main__": main()
