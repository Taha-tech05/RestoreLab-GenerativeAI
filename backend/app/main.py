"""FastAPI service for the image restoration ONNX models."""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import convolve, gaussian_filter

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
SAMPLE_DIR = Path(__file__).parent.parent / "samples"
MOE_TAU = float(os.getenv("MOE_TAU", "1.26"))
MODEL_FILES = {
    "universal": "task1_universal.onnx",
    "classifier": "task2_classifier.onnx",
    "salt": "task2_specialist_salt.onnx",
    "blur": "task2_specialist_blur.onnx",
    "occlusion": "task2_specialist_occlusion.onnx",
    "soft": "task3_soft_moe.onnx",
    "sketch": "face2sketch.onnx",
}
CLASSES = ["clean", "salt", "blur", "occlusion"]
SALT_P = {"low": .03, "medium": .08, "high": .15}
BLUR = {"low": (3, .7), "medium": (5, 1.5), "high": (7, 2.5)}
OCCLUSION = {"low": (1, .10), "medium": (2, .20), "high": (3, .35)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sessions = {}
    app.state.model_errors = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            app.state.model_errors[name] = f"missing: {path}"
            continue
        try:
            app.state.sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            app.state.model_errors[name] = f"failed to load: {exc}"
    yield


app = FastAPI(title="RestoreLab API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


def require_model(name: str):
    session = app.state.sessions.get(name)
    if session is None:
        raise HTTPException(503, detail={"error": "model_unavailable", "model": name, "reason": app.state.model_errors.get(name, "not loaded")})
    return session


async def read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, detail="Image exceeds 10 MB")
    if not data:
        raise HTTPException(400, detail="Empty image")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if (im.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, detail="Only JPEG, PNG, and WEBP images are supported")
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, detail=f"Invalid image: {exc}") from exc


def resized_array(image: Image.Image, resample=Image.Resampling.BICUBIC) -> np.ndarray:
    image = image.resize((128, 128), resample)
    return np.asarray(image, dtype=np.float32) / 255.0


def nchw(rgb: np.ndarray) -> np.ndarray:
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32, copy=False)


def encode_png(rgb: np.ndarray) -> str:
    arr = np.clip(rgb, 0, 1)
    if arr.ndim == 4:
        arr = arr[0].transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    buffer = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def corrupt(clean: np.ndarray, kind: str, severity: str, seed: int | None):
    rng = np.random.default_rng(seed)
    out = clean.copy()
    if kind == "clean":
        return out, {"type": kind, "severity": severity, "params": {}}
    if kind == "salt":
        p = SALT_P[severity]
        n = int(p * 128 * 128)
        ys, xs = rng.integers(0, 128, size=n), rng.integers(0, 128, size=n)
        vals = rng.integers(0, 2, size=n).astype(np.float32)
        out[ys, xs, :] = vals[:, None]
        params = {"probability": p, "pixels": n}
    elif kind == "blur":
        k, sigma = BLUR[severity]
        coords = np.arange(k, dtype=np.float64) - k // 2
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        # scipy mirror maps to cv2 BORDER_REFLECT_101 (edge pixel excluded).
        out = np.stack([convolve(clean[:, :, c], kernel, mode="mirror") for c in range(3)], axis=2).astype(np.float32)
        params = {"kernel": k, "sigma": sigma, "padding": "reflect"}
    elif kind == "occlusion":
        nr, coverage = OCCLUSION[severity]
        target = coverage * 128 * 128
        best = None
        best_diff = float("inf")
        best_coverage = 0.0
        for _ in range(50):
            candidate = clean.copy()
            covered = np.zeros((128, 128), dtype=bool)
            for _rect in range(nr):
                area_per = target / nr
                ar = rng.uniform(.5, 2.0)
                h = max(1, int(round(np.sqrt(area_per / ar))))
                w = max(1, int(round(area_per / h)))
                h, w = min(h, 128), min(w, 128)
                top, left = int(rng.integers(0, 129 - h)), int(rng.integers(0, 129 - w))
                candidate[top:top+h, left:left+w, :] = 0.0
                covered[top:top+h, left:left+w] = True
            measured = covered.mean()
            diff = abs(measured - coverage)
            if diff < best_diff:
                best, best_diff, best_coverage = candidate, diff, measured
            if diff <= .02:
                break
        out = best
        params = {"rectangles": nr, "coverage_target": coverage, "coverage_actual": float(best_coverage)}
    else:
        raise HTTPException(400, detail="corruption must be clean, salt, blur, or occlusion")
    return out, {"type": kind, "severity": severity, "params": params}


def run(session, feeds: dict[str, np.ndarray]):
    start = time.perf_counter()
    result = session.run(None, feeds)
    return result, (time.perf_counter() - start) * 1000.0


def session_run(session, input_name: str, tensor: np.ndarray):
    start = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    elapsed = (time.perf_counter() - start) * 1000.0
    return outputs, elapsed


def run_named(session, feeds: dict[str, np.ndarray]):
    """Return session outputs mapped by the model's declared output names."""
    start = time.perf_counter()
    values = session.run(None, feeds)
    elapsed = (time.perf_counter() - start) * 1000.0
    return dict(zip([item.name for item in session.get_outputs()], values)), elapsed


def psnr_ssim(clean: np.ndarray, restored: np.ndarray):
    # ONNX restoration outputs are batched NCHW; metric inputs are HWC images.
    if restored.ndim == 4:
        restored = restored[0].transpose(1, 2, 0)
    elif restored.ndim == 3 and restored.shape[0] == 3:
        restored = restored.transpose(1, 2, 0)
    mse = float(np.mean((clean - restored) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    # Global luminance/channel averaged SSIM, stabilized for data range 1.
    mu_x, mu_y = gaussian_filter(clean, (1.5, 1.5, 0)), gaussian_filter(restored, (1.5, 1.5, 0))
    var_x = gaussian_filter(clean * clean, (1.5, 1.5, 0)) - mu_x * mu_x
    var_y = gaussian_filter(restored * restored, (1.5, 1.5, 0)) - mu_y * mu_y
    cov = gaussian_filter(clean * restored, (1.5, 1.5, 0)) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + .01**2) * (2 * cov + .03**2)) / ((mu_x**2 + mu_y**2 + .01**2) * (var_x + var_y + .03**2))
    return psnr, float(ssim_map.mean())


def valid_choice(value: str, choices: set[str], label: str):
    if value not in choices:
        raise HTTPException(400, detail=f"{label} must be one of {', '.join(sorted(choices))}")


async def restore_request(image: UploadFile, mode: str, corruption: str, severity: str, seed: int | None, model: str):
    valid_choice(mode, {"apply", "already_corrupted"}, "mode")
    valid_choice(corruption, set(CLASSES), "corruption")
    valid_choice(severity, {"low", "medium", "high"}, "severity")
    original_image = await read_image(image)
    clean = resized_array(original_image)
    if mode == "apply":
        incoming, settings = corrupt(clean, corruption, severity, seed)
    else:
        incoming = clean
        settings = {"type": corruption, "severity": severity, "params": {"applied": False}}
    return original_image, clean, incoming, settings


def base_result(incoming, restored, settings, inference_ms):
    return {"input_b64": encode_png(incoming), "output_b64": encode_png(restored),
            "corruption_settings": settings, "inference_ms": inference_ms}


@app.get("/api/health")
@app.get("/health")
def health():
    loaded = sorted(app.state.sessions.keys())
    return {"status": "ok" if not app.state.model_errors else "degraded", "models": {name: {"loaded": name in app.state.sessions, **({"error": app.state.model_errors[name]} if name in app.state.model_errors else {})} for name in MODEL_FILES}, "loaded_models": loaded}


def sample_path(name: str) -> Path:
    path = (SAMPLE_DIR / name).resolve()
    if path.parent != SAMPLE_DIR.resolve() or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(404, detail="Sample not found")
    return path


@app.get("/api/samples")
@app.get("/samples")
def samples():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    return {"samples": [{"name": p.name, "url": f"/api/samples/{p.name}"} for p in sorted(SAMPLE_DIR.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]}


@app.get("/api/samples/{name}")
@app.get("/samples/{name}")
def get_sample(name: str):
    return {"name": name, "image_b64": base64.b64encode(sample_path(name).read_bytes()).decode("ascii")}


@app.post("/api/restore/universal")
@app.post("/restore/universal")
async def restore_universal(image: UploadFile = File(...), mode: str = Form("apply"), corruption: str = Form("clean"), severity: str = Form("medium"), seed: int | None = Form(None)):
    _, clean, incoming, settings = await restore_request(image, mode, corruption, severity, seed, "universal")
    session = require_model("universal")
    outputs, elapsed = run_named(session, {"input": nchw(incoming)})
    restored = np.clip(outputs["output"], 0, 1)
    result = base_result(incoming, restored, settings, elapsed)
    if mode == "apply":
        result.update(original_b64=encode_png(clean), psnr=psnr_ssim(clean, restored)[0], ssim=psnr_ssim(clean, restored)[1])
    return result


@app.post("/api/restore/hard")
@app.post("/restore/hard")
async def restore_hard(image: UploadFile = File(...), mode: str = Form("apply"), corruption: str = Form("clean"), severity: str = Form("medium"), seed: int | None = Form(None), routing: str = Form("predicted")):
    valid_choice(routing, {"predicted", "oracle"}, "routing")
    _, clean, incoming, settings = await restore_request(image, mode, corruption, severity, seed, "classifier")
    if routing == "oracle" and mode == "already_corrupted":
        raise HTTPException(400, detail="oracle routing requires mode=apply")
    classifier = require_model("classifier")
    probs_out, classifier_ms = run_named(classifier, {"input": nchw(incoming)})
    probs = np.asarray(probs_out["probs"]).reshape(-1)[:4]
    predicted = int(np.argmax(probs))
    chosen = CLASSES.index(corruption) if routing == "oracle" else predicted
    expert_name = CLASSES[chosen]
    expert_ms = 0.0
    if expert_name == "clean":
        restored = incoming.copy()
        selected = "identity"
    else:
        expert = require_model(expert_name)
        expert_out, expert_ms = run_named(expert, {"input": nchw(incoming)})
        restored, selected = np.clip(expert_out["output"], 0, 1), expert_name
    total = classifier_ms + expert_ms
    result = base_result(incoming, restored, settings, total)
    result.update(probs={k: float(v) for k, v in zip(CLASSES, probs)}, predicted_class=CLASSES[predicted], selected_expert=selected, routing_mode=routing, classifier_ms=classifier_ms, expert_ms=expert_ms, total_ms=total)
    if mode == "apply":
        result.update(original_b64=encode_png(clean), psnr=psnr_ssim(clean, restored)[0], ssim=psnr_ssim(clean, restored)[1])
    return result


@app.post("/api/restore/soft")
@app.post("/restore/soft")
async def restore_soft(image: UploadFile = File(...), mode: str = Form("apply"), corruption: str = Form("clean"), severity: str = Form("medium"), seed: int | None = Form(None)):
    _, clean, incoming, settings = await restore_request(image, mode, corruption, severity, seed, "soft")
    session = require_model("soft")
    outputs, elapsed = run_named(session, {"input": nchw(incoming)})
    restored = np.clip(outputs["restored"], 0, 1)
    weights = np.asarray(outputs["weights"]).reshape(-1)[:4]
    result = base_result(incoming, restored, settings, elapsed)
    result.update(weights={k: float(v) for k, v in zip(["identity", "salt", "blur", "occlusion"], weights)}, dominant_branch=["identity", "salt", "blur", "occlusion"][int(np.argmax(weights))], tau=MOE_TAU)
    if mode == "apply":
        result.update(original_b64=encode_png(clean), psnr=psnr_ssim(clean, restored)[0], ssim=psnr_ssim(clean, restored)[1])
    return result


@app.post("/api/sketch")
@app.post("/sketch")
async def sketch(image: UploadFile = File(...), style: int = Form(...)):
    if style not in (1, 2, 3):
        raise HTTPException(400, detail="style must be 1, 2, or 3")
    original = await read_image(image)
    photo = resized_array(original, Image.Resampling.BICUBIC)
    tensor = (nchw(photo) * 2.0 - 1.0).astype(np.float32)
    session = require_model("sketch")
    outputs, elapsed = run_named(session, {"photo": tensor, "style": np.asarray([style - 1], dtype=np.int64)})
    output = np.clip((outputs["sketch"][0] + 1.0) / 2.0, 0, 1)
    return {"input_b64": encode_png(photo), "input_original_b64": encode_png(resized_array(original, Image.Resampling.BICUBIC)), "sketch_b64": encode_png(output), "output_b64": encode_png(output), "style": style, "inference_ms": elapsed, "corruption_settings": {"type": "clean", "severity": "none", "params": {}}}
