"""Stage B 물리 손실 경로 단위 테스트 — 동결 TMM 디코더 + train_gpu physics 플래그.

대회 데이터·GPU 없이 전부 돈다 (CPU device 스모크, CUDA가 있으면 같은 테스트가
cuda로도 돈다 — test_train_gpu.py와 동일한 방침).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.calibrate import CalibratedStack
from src.physics.decoder import TMMDecoder, load_tmm_decoder
from src.train_gpu import physics_beta_at, train_one_model_gpu

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _save_calib_ckpt(run_dir: Path, n_channels: int = 226) -> CalibratedStack:
    """calibrate.py의 save_best와 같은 포맷으로 캘리브레이션 체크포인트를 만든다."""
    stack = CalibratedStack(n_channels=n_channels, n_si_knots=4)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": stack.model_cfg,
            "state_dict": {k: v.detach().cpu() for k, v in stack.state_dict().items()},
            "step": 7,
            "val_rmse": 0.0123,
            "fingerprint": "test",
        },
        run_dir / "model.pt",
    )
    return stack


# ---------------------------------------------------------------------------
# TMMDecoder — CalibratedStack과의 일치, 동결성, 미분가능성
# ---------------------------------------------------------------------------
def test_decoder_matches_calibrated_stack_float64() -> None:
    # complex128 디코더는 CalibratedStack forward와 수치까지 일치해야 한다
    stack = CalibratedStack(n_channels=32, n_si_knots=4)
    with torch.no_grad():
        lam, n_layers, ns = stack.spectra()
    decoder = TMMDecoder(lam, n_layers, ns, dtype=torch.complex128)
    d = torch.rand(8, 4, dtype=torch.float64) * 290.0 + 10.0
    with torch.no_grad():
        r_stack = stack(d)
        r_dec = decoder(d)
    assert torch.allclose(r_dec, r_stack, atol=1e-12)


def test_decoder_complex64_close_to_complex128() -> None:
    # 학습 기본 dtype(complex64)은 complex128 대비 float32 수치오차 수준이어야 한다
    stack = CalibratedStack(n_channels=226, n_si_knots=4)
    with torch.no_grad():
        lam, n_layers, ns = stack.spectra()
    dec64 = TMMDecoder(lam, n_layers, ns, dtype=torch.complex64)
    dec128 = TMMDecoder(lam, n_layers, ns, dtype=torch.complex128)
    d = torch.rand(16, 4, dtype=torch.float64) * 290.0 + 10.0
    with torch.no_grad():
        diff = (dec64(d.float()).double() - dec128(d)).abs()
    assert float(diff.max()) < 1e-4  # R은 O(0.1) — 노이즈 σ(0.0087)보다 두 자릿수 작다


def test_decoder_frozen_and_differentiable_wrt_d() -> None:
    stack = CalibratedStack(n_channels=32, n_si_knots=4)
    with torch.no_grad():
        lam, n_layers, ns = stack.spectra()
    decoder = TMMDecoder(lam, n_layers, ns)
    assert list(decoder.parameters()) == []  # 옵티마이저에 잡힐 파라미터가 없어야 한다
    d = (torch.rand(4, 4) * 290.0 + 10.0).requires_grad_()
    decoder(d).sum().backward()
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert d.grad.abs().sum() > 0


def test_decoder_rejects_bad_dtype_and_shape() -> None:
    stack = CalibratedStack(n_channels=16, n_si_knots=4)
    with torch.no_grad():
        lam, n_layers, ns = stack.spectra()
    with pytest.raises(TypeError, match="complex"):
        TMMDecoder(lam, n_layers, ns, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        TMMDecoder(lam[:-1], n_layers, ns)


def test_load_tmm_decoder_roundtrip(tmp_path: Path) -> None:
    stack = _save_calib_ckpt(tmp_path, n_channels=32)
    decoder, meta = load_tmm_decoder(tmp_path, dtype=torch.complex128)
    assert meta == {"run_dir": str(tmp_path), "step": 7, "val_rmse": 0.0123}
    d = torch.rand(4, 4, dtype=torch.float64) * 290.0 + 10.0
    with torch.no_grad():
        assert torch.allclose(decoder(d), stack(d), atol=1e-12)


# ---------------------------------------------------------------------------
# β 워밍업 스케줄
# ---------------------------------------------------------------------------
def test_physics_beta_warmup_schedule() -> None:
    assert physics_beta_at(1, 100.0, 0) == 100.0  # 워밍업 없음 — 즉시 β
    assert physics_beta_at(5, 100.0, 10) == pytest.approx(50.0)
    assert physics_beta_at(10, 100.0, 10) == pytest.approx(100.0)
    assert physics_beta_at(999, 100.0, 10) == 100.0  # 워밍업 후 상수


# ---------------------------------------------------------------------------
# train_gpu physics 경로 — 스모크 / 대조군 등가성 / resume 계약
# ---------------------------------------------------------------------------
def _tiny_cnn_cfg(epochs: int = 2) -> dict:
    return {
        "model": {"name": "cnn", "channels": [8, 16], "strides": [1, 2], "output_bound": False},
        "train": {
            "epochs": epochs,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "lr_schedule": "cosine",
            "warmup_steps": 2,
        },
    }


def _physics_cfg(decoder_run: Path, beta: float, epochs: int = 2) -> dict:
    cfg = _tiny_cnn_cfg(epochs)
    cfg["train"]["physics"] = {
        "decoder_run": str(decoder_run),
        "beta": beta,
        "beta_warmup_steps": 4,
    }
    return cfg


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)
    return x, y


def _train(run_dir: Path, cfg: dict, device: str = "cpu", **kwargs: object) -> dict:
    x, y = _synthetic()
    return train_one_model_gpu(
        x[:192],
        y[:192],
        x[192:],
        y[192:],
        cfg,
        seed=0,
        run_dir=run_dir,
        tag="model",
        device=torch.device(device),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_train_physics_smoke(device: str, tmp_path: Path) -> None:
    calib_run = tmp_path / "calib"
    _save_calib_ckpt(calib_run)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = _train(run_dir, _physics_cfg(calib_run, beta=50.0), device)

    assert np.isfinite(result["val_mae"])
    assert result["val_phys_l1"] is not None
    assert np.isfinite(result["val_phys_l1"])
    log = (run_dir / "train.log").read_text()
    assert "물리 디코더" in log  # 디코더 출처가 로그에 남는다
    assert "phys_l1" in log and "val_phys" in log


def test_beta_zero_equals_no_physics_block(tmp_path: Path) -> None:
    # beta=0이면 디코더를 로드하지 않는 동일 경로 — 대조군(블록 생략)과 결과가 같아야 한다
    calib_run = tmp_path / "calib"
    _save_calib_ckpt(calib_run)

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    plain = _train(plain_dir, _tiny_cnn_cfg())

    zero_dir = tmp_path / "zero"
    zero_dir.mkdir()
    zero = _train(zero_dir, _physics_cfg(calib_run, beta=0.0))

    assert zero["val_phys_l1"] is None
    assert np.array_equal(zero["val_pred"], plain["val_pred"])
    assert zero["val_mae"] == plain["val_mae"]


def test_negative_beta_raises(tmp_path: Path) -> None:
    calib_run = tmp_path / "calib"
    _save_calib_ckpt(calib_run)
    with pytest.raises(ValueError, match="beta"):
        _train(tmp_path, _physics_cfg(calib_run, beta=-1.0))


@pytest.mark.parametrize("device", _DEVICES)
def test_physics_resume_matches_uninterrupted(device: str, tmp_path: Path) -> None:
    # 재개 = 무중단 계약이 physics 경로(β 워밍업 포함)에서도 성립해야 한다.
    # β는 전역 스텝(에폭에서 결정)으로 계산되므로 에폭 경계 resume에 불변이다.
    calib_run = tmp_path / "calib"
    _save_calib_ckpt(calib_run)
    cfg = _physics_cfg(calib_run, beta=50.0, epochs=3)

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg, device)

    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, device, _abort_after_epoch=1)
    resumed = _train(part_dir, cfg, device)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)
    assert resumed["val_phys_l1"] == pytest.approx(full["val_phys_l1"], abs=1e-6)
