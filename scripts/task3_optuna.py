"""Search the soft-MoE gate, balance, reconstruction, and temperature settings."""
import argparse
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_moe
from common.utils import seed_everything, wandb_start


def main():
    args = add_common_args(argparse.ArgumentParser(), "task3.yaml").parse_args()
    paths = resolve_paths(args); ensure_output_dirs(paths); cfg = paths["config"]
    seed_everything(cfg.get("seed", 42))
    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
              "tau": trial.suggest_float("tau", 0.5, 3.0),
              "w_ce": trial.suggest_float("w_ce", 0.01, 1.0, log=True),
              "w_bal": trial.suggest_float("w_bal", 0.001, 1.0, log=True),
              "w_l1": trial.suggest_float("w_l1", 0.5, 0.95), "batch_size": 32, "warm_lr": 1e-4}
        run = wandb_start(project="genai-ass1", group="task3-optuna", name=f"t3-trial{trial.number}", config=hp, reinit=True)
        try:
            _, score, _state, _gate_cfg, _expert_cfg = train_moe(hp, paths, cfg["search"]["epochs_warm"], cfg["search"]["epochs_joint"], trial)
            run.log({"best_val_score": score}); return score
        finally: run.finish()
    s = cfg["search"]
    study = optuna.create_study(direction=s["direction"], study_name=s["study_name"],
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        storage=f"sqlite:///{paths['out_dir']}/task3_optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=s["trials"])
    study.trials_dataframe().to_csv(paths["out_dir"] / "task3_optuna_trials.csv", index=False)
    print(study.best_trial.number, study.best_value, study.best_params)


if __name__ == "__main__": main()
