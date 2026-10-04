"""Train final style-conditioned FS2K cGAN."""
import argparse
from pathlib import Path
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.trainers import train_cgan
from common.utils import wandb_start


def main():
    args=add_common_args(argparse.ArgumentParser(),"task4.yaml").parse_args(); paths=resolve_paths(args); ensure_output_dirs(paths)
    cfg=paths["config"]; hp=cfg["final"].copy(); epochs=hp.pop("epochs",100)
    run=wandb_start(project="genai-ass1-task4",group="final",name="final",config={**hp,"epochs":epochs},reinit=True)
    try:
        model,l1,ssim,bundle=train_cgan(hp,paths["data_root"],paths["out_dir"],epochs,log=True,save=Path(paths["ckpt_dir"])/"face2sketch_generator.pth")
    finally: run.finish()
    # NOTE: The notebook selects best params from an in-memory Optuna study cell; configured values record its saved output.
    print("val_l1:",l1,"val_ssim:",ssim)


if __name__=="__main__": main()
