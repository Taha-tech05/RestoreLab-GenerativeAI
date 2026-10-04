#!/usr/bin/env python3
"""Print every configured ONNX model's input and output names and shapes."""
import os
from pathlib import Path

import onnxruntime as ort

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODELS = [
    "task1_universal.onnx",
    "task2_classifier.onnx",
    "task2_specialist_salt.onnx",
    "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx",
    "task3_soft_moe.onnx",
    "face2sketch.onnx",
]


def shape(meta):
    return [dim if isinstance(dim, (int, str, type(None))) else str(dim) for dim in meta.shape]


def main():
    for filename in MODELS:
        path = MODEL_DIR / filename
        print(f"{filename}:")
        if not path.is_file():
            print(f"  MISSING ({path})")
            continue
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        for heading, metas in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
            print(f"  {heading}:")
            for meta in metas:
                print(f"    {meta.name}: {shape(meta)}")


if __name__ == "__main__":
    main()
