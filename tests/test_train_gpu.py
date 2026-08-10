"""GPU 학습 경로(src/train_gpu.py) 단위 테스트 — 대회 데이터·GPU 없이 전부 돈다.

train_one_model_gpu는 device 인자만 다를 뿐 CPU에서도 동작해야 한다 (device="cpu"로
스모크). CUDA가 있으면 같은 테스트가 cuda로도 돈다 — Colab에서 pytest로 확인 가능.
"""

from __future__ import annotations

import json
import shutil
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


# ---------------------------------------------------------------------------
# 세션 유실 대비 — resume / best 즉시 저장 / 미러 / 완료 run 스킵
# ---------------------------------------------------------------------------
def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)
    return x, y


def _train(run_dir: Path, cfg: dict | None = None, **kwargs: object) -> dict:
    x, y = _synthetic()
    return train_one_model_gpu(
        x[:192],
        y[:192],
        x[192:],
        y[192:],
        cfg or _tiny_cnn_cfg(),
        seed=0,
        run_dir=run_dir,
        tag="model",
        device=torch.device("cpu"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_resume_after_interrupt_matches_uninterrupted(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 3

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    # epoch 1 직후 세션 중단을 흉내 낸 뒤 재개 — RNG까지 복원되므로 결과가 같아야 한다
    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, _abort_after_epoch=1)
    assert (part_dir / "resume.pt").exists()
    assert (part_dir / "model.pt").exists()  # best는 갱신 즉시 저장 — 중단 시점에도 존재
    resumed = _train(part_dir, cfg)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)
    assert not (part_dir / "resume.pt").exists()  # 완료 시 재개 상태는 정리된다
    # 로그에 재개 기록이 남는다
    assert "resume" in (part_dir / "train.log").read_text()


def test_resume_restores_from_mirror_on_fresh_vm(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 3

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    run_dir = tmp_path / "run"
    mirror = tmp_path / "mirror"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(run_dir, cfg, mirror_dir=mirror, _abort_after_epoch=2)
    # 미러에 에폭 단위 백업이 남아 있어야 한다
    assert (mirror / "resume.pt").exists()
    assert (mirror / "train.log").exists()
    assert (mirror / "model.pt").exists()

    # 세션 유실로 VM 디스크가 날아간 상황: run_dir를 비우고 미러만으로 재개
    shutil.rmtree(run_dir)
    run_dir.mkdir()
    resumed = _train(run_dir, cfg, mirror_dir=mirror)

    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)
    assert not (mirror / "resume.pt").exists()  # 완료 시 미러의 재개 상태도 정리
    assert (mirror / "model.pt").exists()  # best 체크포인트·로그는 미러에 남는다


def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 2
    with pytest.raises(RuntimeError, match="중단"):
        _train(tmp_path, cfg, _abort_after_epoch=1)
    changed = _tiny_cnn_cfg()
    changed["train"]["epochs"] = 2
    changed["train"]["lr"] = 5e-4  # 설정이 달라졌으면 이어받으면 안 된다
    with pytest.raises(ValueError, match="config"):
        _train(tmp_path, changed)


def test_run_config_skips_completed_run(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "done",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    completed = {"experiment": "exp", "run_name": "done", "model": {"val_mae": 1.23}}
    run_dir = tmp_path / "runs" / "exp" / "done"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(completed))

    out = run_config(cfg_path, device="cpu", runs_root=tmp_path / "runs")
    assert out == completed  # 데이터 로드·학습 없이 기존 결과를 그대로 돌려준다


def test_run_config_restores_completed_run_from_mirror(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "done",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    completed = {"experiment": "exp", "run_name": "done", "model": {"val_mae": 1.23}}
    mirror_run = tmp_path / "mirror" / "exp" / "done"
    mirror_run.mkdir(parents=True)
    (mirror_run / "metrics.json").write_text(json.dumps(completed))
    (mirror_run / "model.pt").write_bytes(b"dummy")
    (mirror_run / "train.log").write_text("log")

    out = run_config(
        cfg_path, device="cpu", runs_root=tmp_path / "runs", mirror_dir=tmp_path / "mirror"
    )
    assert out == completed
    # 미러의 산출물이 로컬 runs/로 되돌아온다 (push 대상 복원)
    run_dir = tmp_path / "runs" / "exp" / "done"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "train.log").exists()
