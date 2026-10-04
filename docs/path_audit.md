# Static path and Colab audit

The original notebooks are preserved unchanged. Their Colab-only references are documented here; new source under `common/`, `scripts/`, and `configs/` uses the shared path resolver and contains none of these environment-specific values.

| Original reference | Notebook location | Replacement |
|---|---|---|
| `/content/data` | Pet notebook cells 1–2 and test setup | `--data-root`, `RESTORELAB_DATA_ROOT`, or `configs/paths.yaml` (`data_root`). |
| `/content/drive/MyDrive/pet_restoration_project` | Pet notebook cells 0, 2, 14–26 and later task cells | `--out-dir`, `--ckpt-dir`, `RESTORELAB_OUT_DIR`, `RESTORELAB_CKPT_DIR`, or `configs/paths.yaml`. |
| `from google.colab import drive` and `drive.mount(...)` | Pet cell 0; FS2K cell 0 | Removed from scripts; host paths are passed through CLI/configuration. |
| `/content/drive/MyDrive/FS2K.zip` | FS2K cell 0 | Download and extract the FS2K archive manually into the configured data root; see README. |
| `/content/data/FS2K` | FS2K cells 1 and 4 | `--data-root` / configured `fs2k_root`. |
| `/content/G_best.pt`, `/content/face2sketch.onnx`, `/content/task4.db`, `/content/task4_optuna.csv` | FS2K cells 9, 11, 16 | Checkpoints go under `--ckpt-dir`; studies/results go under `--out-dir`; exports go under the configured model output directory. |
| `/home/...`, `C:\Users\...`, `Desktop`, `genai1`, `pet_restoration_project` | Saved outputs and extracted source/artifact context | Not copied into new runtime code. Notebook/project data paths resolve from CLI, environment, and YAML. |
| `!unzip`, `!ls`, `!pip`, `!mkdir`, `!cp` | FS2K cells 0–1, 13, 15–16; pet cells 27–29, 39, 49, 59–60 | Dataset archive handling is documented in README; package installation is via `training/requirements.txt`; output copy is a normal filesystem/README step. |

The pet corruptions, seeds, 128×128 RGB preprocessing, official test split, and FS2K paired transforms are ported without algorithm changes. Colab GPU selection is replaced by a CUDA-if-available device choice so the same code can run on CPU hosts.
