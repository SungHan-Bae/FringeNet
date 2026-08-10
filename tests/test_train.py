"""학습·평가 파이프라인 단위 테스트 — 대회 데이터 파일 없이 전부 돈다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluate import (
    build_submission_frame,
    load_model_checkpoint,
    mae_per_layer,
    predict,
    snap_to_grid,
)
from src.train import build_lr_scheduler, train_one_model


def _tiny_cfg() -> dict:
    return {
        "model": {"name": "mlp", "hidden_dims": [8], "output_bound": True},
        "train": {
            "epochs": 2,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "lr_schedule": "cosine",
            "warmup_steps": 2,
        },
    }


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def test_mae_per_layer_matches_manual() -> None:
    target = np.zeros((5, 4), dtype=np.float32)
    pred = target + np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    m = mae_per_layer(pred, target)
    assert m["layer_1"] == pytest.approx(1.0)
    assert m["layer_2"] == pytest.approx(2.0)
    assert m["layer_3"] == pytest.approx(3.0)
    assert m["layer_4"] == pytest.approx(4.0)
    assert m["overall"] == pytest.approx(2.5)


def test_snap_to_grid_rounds_and_clips() -> None:
    pred = np.array([[7.0, 154.0, 156.0, 320.0]])
    assert np.array_equal(snap_to_grid(pred), np.array([[10.0, 150.0, 160.0, 300.0]]))


# ---------------------------------------------------------------------------
# 제출 파일 — sample_submission의 id·컬럼 순서를 따라야 한다
# ---------------------------------------------------------------------------
def test_build_submission_frame_aligns_ids(tmp_path: Path) -> None:
    layer_zeros = dict.fromkeys([f"layer_{i}" for i in range(1, 5)], 0.0)
    sample = pd.DataFrame({"id": [30, 10, 20], **layer_zeros})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)

    ids = np.array([10, 20, 30])
    pred = np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4])  # id 10 -> 1, 20 -> 2, 30 -> 3
    out = build_submission_frame(ids, pred, sample_path)

    assert list(out.columns) == ["id", "layer_1", "layer_2", "layer_3", "layer_4"]
    assert out["id"].tolist() == [30, 10, 20]  # sample의 id 순서 유지
    assert out["layer_1"].tolist() == [3.0, 1.0, 2.0]  # id에 맞춰 재정렬됨


def test_build_submission_frame_rejects_missing_id(tmp_path: Path) -> None:
    layer_zeros = dict.fromkeys([f"layer_{i}" for i in range(1, 5)], 0.0)
    sample = pd.DataFrame({"id": [1, 2], **layer_zeros})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    with pytest.raises(ValueError):
        build_submission_frame(np.array([1]), np.ones((1, 4)), sample_path)


# ---------------------------------------------------------------------------
# LR 스케줄러 — linear warmup 후 cosine 감쇠 (스텝 단위)
# ---------------------------------------------------------------------------
def test_lr_scheduler_warmup_then_cosine_decay() -> None:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_lr_scheduler(opt, "cosine", warmup_steps=4, total_steps=10)
    assert sched is not None
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # warmup 4스텝: (step+1)/4 로 선형 상승 — 첫 스텝부터 lr > 0
    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    # 이후 cosine 단조 감소, 전체 스텝을 다 돌면 0에 도달
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:], strict=False))
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)


def test_lr_scheduler_none_and_invalid() -> None:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    assert build_lr_scheduler(opt, "none", warmup_steps=0, total_steps=10) is None
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, "step", warmup_steps=0, total_steps=10)
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, "cosine", warmup_steps=10, total_steps=10)


# ---------------------------------------------------------------------------
# 학습 스모크 — 합성 데이터 2 epoch + 체크포인트 왕복 결정성
# ---------------------------------------------------------------------------
def test_train_one_model_smoke_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)

    result = train_one_model(
        x[:192], y[:192], x[192:], y[192:], _tiny_cfg(), seed=0, run_dir=tmp_path, tag="model"
    )

    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["val_mae"])
    assert result["val_pred"].shape == (64, 4)
    assert (tmp_path / "model.pt").exists()
    history = pd.read_csv(tmp_path / "history_model.csv")
    assert len(history) == 2

    # 체크포인트 왕복: 복원한 모델이 best 시점 예측을 그대로 재현해야 한다
    model = load_model_checkpoint(tmp_path / "model.pt")
    again = predict(model, x[192:])
    assert np.allclose(again, result["val_pred"], atol=1e-5)
