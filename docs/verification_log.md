# Verification log

Verification was performed on Windows PowerShell from the repository root. No training, Optuna study, full evaluation, package install, Docker command, or notebook cell was run. The environment did not match the attachment’s claim that dependencies were installed: `.venv` is missing core training and backend packages; system Python has Torch and ONNX Runtime but not torchvision/FastAPI. Node, npm, Docker, and the Git executable are not on `PATH`.

## Stage summary

| Stage | Check | Status | Evidence |
|---|---|---|---|
| 1 | compileall `common scripts backend/scripts` | PASS | Output below. |
| 1 | common imports | BLOCKED | `.venv` lacks Torch; system Python lacks torchvision. |
| 1 | `--help` for every training script | BLOCKED | All 17 fail in `.venv` because required packages such as Torch, NumPy, Matplotlib, Optuna, and PyYAML are absent. `export_onnx.py --help` and `verify_onnx.py --help` work in system Python after the bootstrap fix. |
| 2 | seven model files, sizes, ONNX sidecar | PASS | All seven expected files exist; face2sketch is 118,175,155 bytes; no `.onnx.data` sidecar. |
| 2 | model I/O names/shapes/dtypes | PASS | `backend/scripts/check_onnx.py` output below. I/O matches the supplied model facts. |
| 2 | ONNX/PyTorch parity | PASS | All seven max absolute differences are below `1e-4`; output saved to `verification_outputs/onnx_consistency.json`. |
| 3 | direct Uvicorn/API tests, negative tests, output pixel analysis, pytest | BLOCKED | Uvicorn and pytest are absent from `.venv`; FastAPI is absent from system Python. No API process was left running. |
| 4 | npm install/build/dev-server/browser checks | BLOCKED | `npm` is not installed/on `PATH`; static route/API checks passed by reading files. |
| 5 | Docker Compose/runtime | BLOCKED | `docker` is not installed/on `PATH`; per instructions, Compose checks were skipped. |
| 6 | ignore/path/secret/README static checks | PARTIAL | Ignore rules and no-secret/path scans are clean. `.git` exists, but Git is unavailable, so tracked status cannot be determined. |

## Commands and actual output

### Stage 1

Command: `python -m compileall common scripts backend\scripts -q`

```text
compileall common scripts backend/scripts: PASS
```

Command: `& .\.venv\Scripts\python.exe -c "import common.models, common.corruptions, common.data_pets, common.data_fs2k, common.trainers, common.losses"`

```text
ModuleNotFoundError: No module named 'torch'
```

System Python import attempt:

```text
ModuleNotFoundError: No module named 'torchvision'
```

Help checks: `& .\.venv\Scripts\python.exe <script> --help` for each `scripts/*.py`.

```text
FAIL export_onnx.py: No module named 'torch'
FAIL prepare_data.py: No module named 'numpy'
FAIL task1_eval.py: No module named 'matplotlib'
FAIL task1_optuna.py: No module named 'optuna'
FAIL task1_train.py: No module named 'yaml'
FAIL task2_eval.py: No module named 'numpy'
FAIL task2_optuna_classifier.py: No module named 'optuna'
FAIL task2_optuna_specialists.py: No module named 'optuna'
FAIL task2_train_classifier.py: No module named 'optuna'
FAIL task2_train_specialists.py: No module named 'optuna'
FAIL task3_eval.py: No module named 'matplotlib'
FAIL task3_optuna.py: No module named 'optuna'
FAIL task3_train.py: No module named 'torch'
FAIL task4_eval.py: No module named 'matplotlib'
FAIL task4_optuna.py: No module named 'optuna'
FAIL task4_train.py: No module named 'yaml'
FAIL verify_onnx.py: No module named 'numpy'
```

After the script-root bootstrap change, system-Python help checks succeeded for `export_onnx.py` and `verify_onnx.py`; other entry points still require missing dependencies. `verify_onnx.py --help` printed the options including `--onnx-dir`.

### Stage 2

Command: `$env:MODEL_DIR='models'; python backend/scripts/check_onnx.py`

```text
task1_universal.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: output ['batch', 3, 128, 128] tensor(float)
task2_classifier.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: probs ['batch', 4] tensor(float)
task2_specialist_salt.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: output ['batch', 3, 128, 128] tensor(float)
task2_specialist_blur.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: output ['batch', 3, 128, 128] tensor(float)
task2_specialist_occlusion.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: output ['batch', 3, 128, 128] tensor(float)
task3_soft_moe.onnx:
  inputs: input ['batch', 3, 128, 128] tensor(float)
  outputs: restored ['batch', 3, 128, 128] tensor(float); weights ['batch', 4] tensor(float)
face2sketch.onnx:
  inputs: photo ['b', 3, 128, 128] tensor(float); style ['b'] tensor(int64)
  outputs: sketch ['b', 3, 128, 128] tensor(float)
```

The seven local file sizes were: task1 17,449,468; classifier 18,863,066; specialists 17,515,070 each; soft MoE 71,414,370; face2sketch 118,175,155 bytes. `Test-Path models\face2sketch.onnx.data` returned `False`.

Classifier/MoE sample inference and saved checkpoint metadata check:

```text
checkpoint tau: 1.2615344229334267
probs sum: [1.0]
weights sum: [1.0]
```

This matches the stated display value 1.26 to two decimals; the trained model’s saved temperature is baked into its graph.

Parity command:
`python scripts/verify_onnx.py --ckpt-dir pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project --onnx-dir models --out-dir verification_outputs --config configs/task4.yaml`

```text
task1_universal.onnx max_abs_diff= 6.556510925292969e-07
task2_classifier.onnx max_abs_diff= 1.3969838619232178e-09
task2_specialist_salt.onnx max_abs_diff= 1.0132789611816406e-06
task2_specialist_blur.onnx max_abs_diff= 1.2516975402832031e-06
task2_specialist_occlusion.onnx max_abs_diff= 2.1457672119140625e-06
task3_soft_moe.onnx max_abs_diff= 4.76837158203125e-07
face2sketch.onnx max_abs_diff= 2.5033950805664062e-06
```

### Stage 3

Command from `backend/`: `..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`

```text
No module named uvicorn
```

The environment also returned `ModuleNotFoundError: No module named 'fastapi'` for system Python and `No module named pytest` for `.venv\Scripts\python.exe -m pytest backend\tests\test_api.py -q`. No health, sample, restore, sketch, negative-upload, or image-stat requests could be made. There is no live app URL from this stage.

### Stage 4

Command from `frontend/`: `npm install`

```text
npm : The term 'npm' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

Static inspection confirmed `src/api.ts` uses `/api/...` relative paths; `vite.config.ts` proxies `/api` to `http://localhost:8000`; `src/App.tsx` declares `/universal`, `/hard`, `/soft`, `/sketch`, the backend-offline banner, webcam `getUserMedia`, and capture logic. The dependency install/build/dev server and browser console checks were not run.

### Stage 5

Command: `docker version`

```text
docker : The term 'docker' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

Compose config/build/health checks were skipped. No containers are running from this verification.

### Stage 6

The following large files were found; each matches a visible `.gitignore` rule (`models/`, `*.onnx`, `*.pt`, or `*.pth`):

```text
models/face2sketch.onnx 112.7 MB
models/task3_soft_moe.onnx 68.1 MB
pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/face2sketch.onnx 112.7 MB
pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/G_best.pt 112.7 MB
pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task1_best.pth 201.5 MB
pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task3_soft_moe.onnx 68.1 MB
pet_restoration_project-20261004T094832Z-1-001/pet_restoration_project/task3_soft_moe.pth 68.3 MB
```

`.env` filename scan returned only `.env.example`. Secret-value regex search outside notebooks returned no files. Environment-path search over `common/`, `scripts/`, `configs/`, and `backend/` returned no matches. README still contains placeholders for the FS2K dataset link, model bundle link, and W&B project URLs; its `docker compose up --build` and `http://localhost:3000` instructions match the static Compose configuration. `.git` exists, but `git` is unavailable on `PATH`, so `git status`/tracked-file/LFS checks could not be performed. No `git init` was needed.

## Fixes made

| File | Change | Reason and recheck |
|---|---|---|
| `backend/scripts/check_onnx.py` | Print tensor dtype as well as input/output name and shape. | Needed to compare all metadata with the supplied facts. Reran the utility successfully on all seven models. |
| `scripts/*.py` | Add repository-root path bootstrap before local imports. | Direct script invocation could not locate `common`. `python scripts/verify_onnx.py --help` then printed usage; compileall passed. Other help commands remain blocked by missing packages. |
| `scripts/verify_onnx.py` | Add `--onnx-dir`, fallback to the provided `G_best.pt`, and support Task 3 checkpoint metadata under either `cfg` or `hp`. | Matched the supplied artifact directory and checkpoint layout. Reran parity successfully for all seven files. |
| `scripts/export_onnx.py` | Fallback to `G_best.pt` and support Task 3 metadata under `cfg` or `hp`. | Keeps export compatible with the supplied checkpoint naming/layout. Compiled; export itself was not run. |

## ONNX parity

| Model | Max absolute difference | Tolerance | Result |
|---|---:|---:|---|
| task1_universal | 6.56e-7 | 1e-4 | PASS |
| task2_classifier | 1.40e-9 | 1e-4 | PASS |
| task2_specialist_salt | 1.01e-6 | 1e-4 | PASS |
| task2_specialist_blur | 1.25e-6 | 1e-4 | PASS |
| task2_specialist_occlusion | 2.15e-6 | 1e-4 | PASS |
| task3_soft_moe | 4.77e-7 | 1e-4 | PASS |
| face2sketch | 2.50e-6 | 1e-4 | PASS |

## API response image sanity

| Endpoint | Min | Max | Mean | Verdict |
|---|---:|---:|---:|---|
| Universal restoration | — | — | — | BLOCKED: backend dependencies unavailable; endpoint was not called. |
| Hard routing | — | — | — | BLOCKED: backend dependencies unavailable; endpoint was not called. |
| Soft MoE | — | — | — | BLOCKED: backend dependencies unavailable; endpoint was not called. |
| Face-to-sketch | — | — | — | BLOCKED: backend dependencies unavailable; endpoint was not called. |

## Remaining problems, highest impact first

1. Install/provide the declared Python, Node/npm, Git, and Docker toolchains in the verification environment; current dependencies contradict the attachment’s environment assumption. The live backend/frontend/Compose evaluation and API image analysis remain blocked.
2. Replace the README dataset/model/W&B URL placeholders before distribution.
3. Git is unavailable, so large-file tracked status and LFS coverage are unknown. Large local files are ignored by visible patterns, but ignored status is not equivalent to verified Git tracking status.

No application URL is currently running. The intended Compose URL remains `http://localhost:3000` after Docker is available.
