# RestoreLab

RestoreLab is a two-service image restoration suite. The React frontend is served by Nginx, which proxies `/api/` requests to the FastAPI backend. ONNX files are mounted read-only from `./models`; they are not copied into either container.

Experiment tracking is done with Weights & Biases (W&B) Cloud. There is no MLflow service.

## Prerequisites

- Docker Desktop with Docker Compose v2, or Docker Engine and the Compose plugin.
- The seven trained ONNX model files listed below.
- About 2 GB of free disk space for images and build layers; model files need roughly 320 MB.

## Get the ONNX models

The supplied training artifacts include the ONNX files. Download them from the links below, or copy them from the [pet restoration training artifacts folder](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/) into `./models`. For a separately hosted bundle, replace these source links with your team's shared download URL before publishing the repository.

- [task1_universal.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task1_universal.onnx)
- [task2_classifier.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task2_classifier.onnx)
- [task2_specialist_salt.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task2_specialist_salt.onnx)
- [task2_specialist_blur.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task2_specialist_blur.onnx)
- [task2_specialist_occlusion.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task2_specialist_occlusion.onnx)
- [task3_soft_moe.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task3_soft_moe.onnx)
- [face2sketch.onnx](./pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/face2sketch.onnx)

The filenames in `./models` must be exactly:

```text
task1_universal.onnx
task2_classifier.onnx
task2_specialist_salt.onnx
task2_specialist_blur.onnx
task2_specialist_occlusion.onnx
task3_soft_moe.onnx
face2sketch.onnx
```

The model files are ignored by Git. The backend still starts if one or more files are missing; open `http://localhost:8000/api/health` to see which models loaded. Workspaces whose required models are unavailable stay disabled until the files are present.

If you are using the supplied training artifact folder, copy its exported ONNX files into the mount before startup:

```powershell
Copy-Item pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/*.onnx models/
```

```sh
cp pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/*.onnx models/
```

## Run with Docker Compose

Create a local `.env` from the example, then start both services:

```sh
cp .env.example .env
docker compose up --build
```

On PowerShell, use `Copy-Item .env.example .env` instead of `cp` if needed.

- Frontend: [http://localhost:3000](http://localhost:3000)
- API health: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- `MOE_TAU` defaults to `1.26`; it is displayed by the soft-MoE UI. The trained model already includes its temperature.

Stop the services with `docker compose down`. Model files remain in `./models` on the host.

The project report, including the original Stitch screenshot evidence, is in [REPORT.md](REPORT.md).

## Troubleshooting

- **Model unavailable in a workspace:** confirm every required filename is directly inside `./models` (not nested in an extracted subfolder), then inspect `/api/health`. Restart after adding files with `docker compose restart backend`.
- **Backend is unhealthy:** run `docker compose logs backend`. The health check calls `/api/health`; a missing model by itself is reported as degraded and does not prevent the API from starting.
- **Frontend shows Backend offline:** run `docker compose ps` and `docker compose logs backend frontend`. Wait for the backend health check to pass; Nginx routes `/api/` over the Compose network.
- **Port 3000 or 8000 is already in use:** change the host side of the corresponding mapping in `docker-compose.yml`, for example `3001:80`.
- **Upload rejected:** only JPEG, PNG, and WEBP images up to 10 MB are accepted. Nginx allows 12 MB to leave room for multipart form overhead.
- **Permission or mount issue:** make sure `./models` exists and is readable on the host. The container mounts it read-only at `/app/models`.

## Weights & Biases projects

Replace the placeholders with the team's W&B Cloud project URLs in `frontend/src/config.ts` and below:

- [Task 1 · Universal restoration](https://wandb.ai/your-team/your-universal-project)
- [Task 2 · Hard-routed experts](https://wandb.ai/your-team/your-hard-routing-project)
- [Task 3 · Soft mixture of experts](https://wandb.ai/your-team/your-soft-moe-project)
- [Task 4 · Face to sketch](https://wandb.ai/your-team/your-sketch-project)

## Training, evaluation and export

The original Colab notebooks are retained in `notebooks/`. Their cell-to-script map and static path cleanup notes are in [docs/notebook_map.md](docs/notebook_map.md) and [docs/path_audit.md](docs/path_audit.md). Shared loaders, model definitions, losses, metrics, corruption logic, and utilities live under `common/`; command line entry points are in `scripts/`; final settings and portable paths are in `configs/`; Optuna databases, plots, and result tables go in `experiments/`. Checkpoints default to `checkpoints/`, and ONNX exports default to `experiments/onnx/`.

Every script accepts `--data-root`, `--out-dir`, `--ckpt-dir`, and `--config`. Defaults are in `configs/paths.yaml`; equivalent overrides are `RESTORELAB_DATA_ROOT`, `RESTORELAB_OUT_DIR`, and `RESTORELAB_CKPT_DIR`. The scripts print resolved locations before work begins. Install unpinned training dependencies with `pip install -r training/requirements.txt` in a separate training environment.

### Datasets

- **Oxford-IIIT Pet:** `prepare_data.py` downloads the official dataset using `torchvision.datasets.OxfordIIITPet` into `data/`. Official `trainval` is divided into train/validation with seed 42; the official test split remains untouched.
- **FS2K:** download from the course-provided or official distribution and extract into `data/FS2K/`, with `anno_train.json`, `anno_test.json`, `photo/`, and `sketch/` directly inside. Link placeholder: [FS2K dataset](https://your-dataset-download-link).
- **Model/checkpoint download placeholder:** [trained checkpoints and ONNX bundle](https://your-model-download-link). The Docker app needs the seven exact ONNX filenames listed above in `models/`.

### Script order

Run from the repository root. `prepare_data.py` supports `--dataset pets` and `--dataset fs2k`.

```sh
python scripts/prepare_data.py --dataset pets --data-root data
python scripts/task1_optuna.py
python scripts/task1_train.py
python scripts/task1_eval.py
python scripts/task2_optuna_classifier.py
python scripts/task2_train_classifier.py
python scripts/task2_optuna_specialists.py
python scripts/task2_train_specialists.py
python scripts/task2_eval.py
python scripts/task3_optuna.py
python scripts/task3_train.py
python scripts/task3_eval.py
python scripts/prepare_data.py --dataset fs2k --data-root data
python scripts/task4_optuna.py --data-root data/FS2K
python scripts/task4_train.py --data-root data/FS2K
python scripts/task4_eval.py --data-root data/FS2K
python scripts/export_onnx.py --config configs/task4.yaml --data-root data/FS2K
python scripts/verify_onnx.py --config configs/task4.yaml --data-root data/FS2K
```

Tasks 1–3 use the Pet root. For Task 4, pass the FS2K directory as `--data-root`. Task 2 uses its classifier and specialist studies; Task 3 uses the trained Task 2 checkpoints. Export writes embedded-weight ONNX files under `experiments/onnx/`; copy them into `models/` for the Docker app. The check script prints each ONNX/PyTorch maximum absolute difference (comparison tolerance `1e-4`) and records results in `experiments/onnx_consistency.json`. It was not run as part of this static notebook reorganization.

SQLite studies, trial CSVs, metrics, and plots go under `experiments/`, which starts with only `.gitkeep` and is ignored for generated outputs.

### Weights & Biases

Set `WANDB_API_KEY` in the shell before training to authenticate. The scripts call `wandb.login()` and contain no stored credentials. Project placeholders (replace with the team URLs): [Tasks 1–3: genai-ass1](https://wandb.ai/your-team/genai-ass1) and [Task 4: genai-ass1-task4](https://wandb.ai/your-team/genai-ass1-task4).

### Folder map

```text
common/       shared datasets, corruptions, models, losses, metrics, utilities, loops
scripts/      preparation, task searches/training/evaluation, ONNX export/check
configs/      paths.yaml and task1.yaml through task4.yaml
training/     training-only Python dependencies
experiments/  runtime studies and outputs (ignored; starts with .gitkeep)
docs/         notebook map and static path audit
```

AI-use note: AI assistance was used to reorganize notebook code into shared modules and command-line scripts. The original notebooks are retained for comparison; review and validate generated artifacts against them before relying on results.
