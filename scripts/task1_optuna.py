"""Task 1 spatial-bottleneck restoration Optuna search."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_restorer
from common.utils import seed_everything, wandb_start


def main():
    parser = add_common_args(argparse.ArgumentParser(), "task1.yaml")
    args = parser.parse_args(); paths = resolve_paths(args); ensure_output_dirs(paths)
    cfg = paths["config"]; seed_everything(cfg.get("seed", 42))
    def objective(trial):
        hp = {"arch": "spatial", "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
              "latent_ch": trial.suggest_categorical("latent_ch", [16, 32, 64, 128]),
              "base_ch": trial.suggest_categorical("base_ch", [32, 48, 64]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.3),
              "alpha": trial.suggest_float("alpha", 0.5, 0.95)}
        run = wandb_start(project="genai-ass1", group="task1-optuna-spatial",
                          name=f"t1sp-trial{trial.number}", config=hp, reinit=True)
        try:
            _, best, _ = train_restorer(hp, paths, epochs=cfg["search"]["epochs"], trial=trial)
            run.log({"best_val_score": best}); return best
        finally: run.finish()
    search = cfg["search"]
    study = optuna.create_study(direction=search["direction"], study_name=search["study_name"],
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=search["pruner_startup_trials"],
                                           n_warmup_steps=search["pruner_warmup_steps"]),
        storage=f"sqlite:///{paths['out_dir']}/task1_optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=search["trials"])
    study.trials_dataframe().to_csv(paths["out_dir"] / "task1_optuna_trials.csv", index=False)
    print(study.best_trial.number, study.best_value, study.best_params)


if __name__ == "__main__": main()
