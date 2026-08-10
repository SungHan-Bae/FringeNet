"""GPU 학습 경로(src/train_gpu.py) 단위 테스트 — 대회 데이터·GPU 없이 전부 돈다.

train_one_model_gpu는 device 인자만 다를 뿐 CPU에서도 동작해야 한다 (device="cpu"로
스모크). CUDA가 있으면 같은 테스트가 cuda로도 돈다 — Colab에서 pytest로 확인 가능.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate import load_model_checkpoint, predict
from src.train_gpu import predict_on_device, resolve_device, run_config, train_one_model_gpu

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tiny_cnn_cfg() -> dict:
    return {
        "model": {
            "name": "cnn",
            "channels": [8, 16],
            "strides": [1, 2],
            "output_bound": False,
        },
        "train": {
            "epochs": 2,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "lr_schedule": "cosine",
            "warmup_steps": 2,
        },
    }


@pytest.mark.parametrize("device", _DEVICES)
def test_train_one_model_gpu_smoke_and_checkpoint_roundtrip(device: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)

    result = train_one_model_gpu(
        x[:192],
        y[:192],
        x[192:],
        y[192:],
        _tiny_cnn_cfg(),
        seed=0,
        run_dir=tmp_path,
        tag="model",
        device=torch.device(device),
    )

    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["val_mae"])
    assert result["val_pred"].shape == (64, 4)
    assert (tmp_path / "train.log").read_text().count("epoch") == 2

    # 체크포인트는 device와 무관하게 CPU 텐서여야 한다 — 로컬 CPU 분석 호환성의 핵심
    ckpt = torch.load(tmp_path / "model.pt", map_location=None, weights_only=True)
    assert all(v.device.type == "cpu" for v in ckpt["state_dict"].values())

    # 왕복: CPU 파이프라인의 load_model_checkpoint + predict로 best 예측이 재현돼야 한다
    model = load_model_checkpoint(tmp_path / "model.pt")
    again = predict(model, x[192:])
    assert np.allclose(again, result["val_pred"], atol=1e-4)


def test_predict_on_device_matches_cpu_predict() -> None:
    from src.models import build_model

    torch.manual_seed(0)
    model = build_model({"name": "cnn", "channels": [8], "strides": [1]})
    x = torch.rand(50, 226)
    out = predict_on_device(model, x, batch_size=16)
    assert out.shape == (50, 4)
    assert np.allclose(out, predict(model, x.numpy()), atol=1e-6)


def test_resolve_device_defaults_and_rejects_missing_cuda() -> None:
    assert resolve_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(None).type == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_run_config_rejects_kfold(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "run",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3, "num_folds": 5},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="holdout"):
        run_config(path, device="cpu")
