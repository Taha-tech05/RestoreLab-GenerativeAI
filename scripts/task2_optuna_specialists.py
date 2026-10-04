"""Shared Task 2 specialist hyperparameter study."""
import argparse
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_specialist
from common.utils import seed_everything, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task2.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    search = cfg["specialist_search"]; seed_everything(cfg.get("seed", 42))
    def objective(trial):
        hp = {"arch": "spatial", "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
              "latent_ch": trial.suggest_categorical("latent_ch", [16, 32, 64, 128]),
              "base_ch": trial.suggest_categorical("base_ch", [32, 48, 64]),
              "dropout": trial.suggest_float("dropout", 0.0, 0.3),
              "alpha": trial.suggest_float("alpha", 0.5, 0.95)}
        run = wandb_start(project="genai-ass1", group="task2-spec-optuna",
                          name=f"t2s-trial{trial.number}", config=hp, reinit=True)
        try:
            _, best, _ = train_specialist(hp, paths, "salt", search["epochs"], trial=trial)
            run.log({"best_val_score": best}); return best
        finally: run.finish()
    study = optuna.create_study(direction=search["direction"], study_name=search["study_name"],
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=f"sqlite:///{paths['out_dir']}/task2_optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=search["trials"])
    study.trials_dataframe().to_csv(paths["out_dir"] / "task2_spec_optuna_trials.csv", index=False)
    print(study.best_trial.number, study.best_value, study.best_params)


if __name__ == "__main__": main()
