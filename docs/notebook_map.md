# Notebook cell map

Cell numbers below are zero-based indices from the original notebooks. The cell numbers identify source cells; saved outputs are not treated as executable source.

## `notebooks/genaiass1taskprep,1,2,3.ipynb`

| Cell range | Purpose |
|---|---|
| 0–2 | Colab/Drive setup, Oxford-IIIT Pet trainval download, seed 42, 80/20 index split, RGB 128×128 preprocessing. |
| 3–7 | Salt-and-pepper, Gaussian blur, rectangle occlusion; runtime random-condition training dataset; deterministic validation/test manifest generation and manifest-backed dataset. |
| 8–10 | Corruption preview, repeatability/coverage checks, and fixed test-severity visualization. |
| 11–15 | Flat-bottleneck `UniversalDAE`, SSIM/L1 loss and PSNR, evaluation/training loop, Task 1 Optuna search and final flat model. |
| 16–26 | Spatial-bottleneck `SpatialDAE`, final spatial search/training, per-image test metrics and example/failure plots, Task 1 ONNX export and numerical comparison. |
| 27–33 | Colab package-install/login cells and post-hoc black-pixel diagnostics; install magics are replaced by `training/requirements.txt`, login uses the environment. |
| 34–37 | Task 2 labeled corruption dataset, balanced batch sampler, classifier CNN, classifier Optuna and final training. |
| 38–41 | Specialist restoration training/search and independent salt, blur, and occlusion checkpoints. |
| 42–46 | Classification report, normalized confusion matrix, oracle/predicted routing evaluation, routing failure/example grids, Optuna plots. |
| 47–49 | Classifier/specialist reconstruction, hard-routing ONNX exports and ONNX-vs-PyTorch checks; install magic removed. |
| 50–53 | Task 3 soft-MoE model, gate/expert initialization, balance/reconstruction/classification loss, frozen warm-up, joint fine-tuning, Optuna and final training. |
| 54–58 | Per-image Task 3 evaluation, gate-weight CSV/heatmap, dominant/inactive expert diagnostics, sample grids, `tau` metadata, ONNX export and parity checks. |
| 59–61 | Colab installs and empty trailing cell; dependencies are moved to `training/requirements.txt`. |

## `notebooks/genaiass1task4.ipynb`

| Cell range | Purpose |
|---|---|
| 0–1 | Drive mount/FS2K zip extraction and shell-based directory inspection; replaced with README download/extract instructions and `prepare_data.py`. |
| 2–4 | Imports, FS2K annotations and paired photo/sketch loading, official test split, stratified 15% validation split (seed 42), identical train flip. |
| 5 | Style-embedded U-Net generator and PatchGAN discriminator. |
| 6 | GAN losses, separate D-real/D-fake/G-adversarial/G-L1 logs, validation L1/SSIM and fixed-photo progress grids. |
| 7–9 | Task 4 Optuna search, study export, W&B final run and generator checkpoint. |
| 10–12 | Generator reload, ONNX export preparation, full test metrics. |
| 13 | ONNX exporter install magic; dependency moved to `training/requirements.txt`. |
| 14 | Per-style test metrics. |
| 15–16 | Drive mkdir/copy shell magics; replaced by configured output directory and README copy instructions. |
| 17 | Empty trailing cell. |

## Port notes

- The first notebook's filename is `genaiass1taskprep,1,2,3.ipynb` in this repository; it was not renamed or edited.
- The notebooks contain saved cell outputs from earlier Colab runs. These are retained in the originals and are not replayed by the scripts.
- Task 1 includes both a flat-bottleneck baseline and a later spatial-bottleneck variant. The exported `task1_universal.onnx` in the notebook is built from the spatial checkpoint; both model classes are retained.
- Task 4 cell 9 assigns `nbest = study.best_params` but then builds `cfg` using `best[...]`. This pre-existing notebook-state ambiguity is retained as a `# NOTE:` in the corresponding training script/config flow; the explicit best-parameter YAML is used by the script.
