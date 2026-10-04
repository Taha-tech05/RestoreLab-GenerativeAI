"""Optuna search for the conditional FS2K cGAN."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import optuna
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_cgan
from common.utils import wandb_start


def main():
    args=add_common_args(argparse.ArgumentParser(),"task4.yaml").parse_args(); paths=resolve_paths(args); ensure_output_dirs(paths); cfg=paths["config"]
    def objective(trial):
        hp={"lr_g":trial.suggest_float("lr_g",5e-5,1e-3,log=True),"lr_d":trial.suggest_float("lr_d",5e-5,1e-3,log=True),
            "bs":trial.suggest_categorical("bs",[8,16,32]),"c":trial.suggest_categorical("c",[32,48,64]),
            "drop":trial.suggest_float("drop",0.0,0.5),"emb":trial.suggest_categorical("emb",[8,16,32,64]),"lam":trial.suggest_float("lam",10,200,log=True)}
        run=wandb_start(project="genai-ass1-task4",group="optuna",name=f"optuna-t{trial.number}",config=hp,reinit=True)
        try:
            _g,_l1,ssim,_bundle=train_cgan(hp,paths["data_root"],paths["out_dir"],cfg["search"]["epochs"],trial=trial)
            run.log({"val_ssim":ssim}); return ssim
        finally: run.finish()
    s=cfg["search"]; study=optuna.create_study(direction=s["direction"],study_name=s["study_name"],sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{paths['out_dir']}/task4_optuna.db",load_if_exists=True)
    study.optimize(objective,n_trials=s["trials"]); study.trials_dataframe().to_csv(paths["out_dir"] / "task4_optuna_trials.csv",index=False)
    print(study.best_trial.number,study.best_value,study.best_params)


if __name__=="__main__": main()
