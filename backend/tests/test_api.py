import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app, MODEL_FILES


class FakeMeta:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, kind):
        self.kind = kind

    def get_outputs(self):
        names = {"universal": ["output"], "classifier": ["probs"], "salt": ["output"], "blur": ["output"], "occlusion": ["output"], "soft": ["restored", "weights"], "sketch": ["sketch"]}[self.kind]
        return [FakeMeta(name) for name in names]

    def run(self, _, feeds):
        x = feeds.get("input", feeds.get("photo"))
        if self.kind == "classifier":
            return [np.array([[0.01, 0.96, 0.02, 0.01]], np.float32)]
        if self.kind == "soft":
            return [np.zeros((1, 3, 128, 128), np.float32), np.array([[.1, .7, .1, .1]], np.float32)]
        if self.kind == "sketch":
            return [np.zeros((1, 3, 128, 128), np.float32)]
        return [np.clip(x, 0, 1)]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.sessions = {name: FakeSession(name) for name in MODEL_FILES}
        app.state.model_errors = {}
        yield test_client


def upload():
    buffer = io.BytesIO()
    Image.new("RGB", (24, 20), (130, 75, 20)).save(buffer, format="PNG")
    return {"image": ("test.png", buffer.getvalue(), "image/png")}


def test_health_reports_models(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert all(item["loaded"] for item in response.json()["models"].values())


def test_samples_endpoint(client):
    assert client.get("/api/samples").status_code == 200
    assert client.get("/api/samples/missing.png").status_code == 404


@pytest.mark.parametrize("path", ["universal", "hard", "soft"])
def test_restore_endpoints(client, path):
    response = client.post(f"/api/restore/{path}", files=upload(), data={"mode": "apply", "corruption": "salt", "severity": "low", "seed": "42"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["input_b64"] and result["output_b64"]
    assert result["corruption_settings"]["type"] == "salt"
    assert "inference_ms" in result
    assert "psnr" in result and "ssim" in result


def test_hard_oracle_rejects_already_corrupted(client):
    response = client.post("/api/restore/hard", files=upload(), data={"mode": "already_corrupted", "routing": "oracle", "corruption": "salt"})
    assert response.status_code == 400


def test_sketch_endpoint(client):
    response = client.post("/api/sketch", files=upload(), data={"style": "2"})
    assert response.status_code == 200, response.text
    assert response.json()["style"] == 2
    assert response.json()["sketch_b64"]


def test_invalid_media_is_rejected(client):
    response = client.post("/api/sketch", files={"image": ("x.gif", b"bad", "image/gif")}, data={"style": "1"})
    assert response.status_code in (400, 415)
