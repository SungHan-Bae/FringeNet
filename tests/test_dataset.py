"""데이터 로더 테스트.

순수 로직(split, Dataset)은 데이터 없이 돌고, 실제 파일이 필요한 테스트는
`data/raw/`가 없으면 skip 한다 — 저장소에 데이터가 없어도 `pytest -q`가 green이어야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import (
    CHANNEL_COLS,
    LAYER_COLS,
    N_CHANNELS,
    RAW_DIR,
    FringeDataset,
    kfold_indices,
    load_frame,
    prepare_train_arrays,
    random_split_indices,
)

requires_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "train.csv").exists(),
    reason="data/raw/train.csv 없음 (대회 데이터는 저장소에 포함하지 않는다)",
)


# ---------------------------------------------------------------------------
# 컬럼 규약
# ---------------------------------------------------------------------------
def test_column_constants_are_consistent() -> None:
    assert len(CHANNEL_COLS) == N_CHANNELS == 226
    assert CHANNEL_COLS[0] == "0"
    assert CHANNEL_COLS[-1] == "225"
    assert LAYER_COLS == ["layer_1", "layer_2", "layer_3", "layer_4"]


# ---------------------------------------------------------------------------
# split — 겹치지 않고, 전부 덮고, 시드에 대해 재현 가능해야 한다
# ---------------------------------------------------------------------------
def test_random_split_is_a_partition_and_reproducible() -> None:
    n = 1000
    train_idx, val_idx = random_split_indices(n, val_frac=0.1, seed=42)

    assert len(val_idx) == 100
    assert len(train_idx) == 900
    assert set(train_idx).isdisjoint(val_idx)
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(n))

    again = random_split_indices(n, val_frac=0.1, seed=42)
    assert np.array_equal(train_idx, again[0])
    assert np.array_equal(val_idx, again[1])

    other = random_split_indices(n, val_frac=0.1, seed=7)
    assert not np.array_equal(val_idx, other[1])


def test_random_split_rejects_invalid_fraction() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            random_split_indices(100, val_frac=bad)


def test_kfold_indices_partitions_given_indices() -> None:
    base = np.arange(100, 200)  # holdout을 뺀 나머지 인덱스를 흉내낸다
    folds = kfold_indices(base, 5, seed=0)

    assert len(folds) == 5
    # OOF 조각들이 base를 정확히 한 번씩 덮는다
    all_val = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(all_val), base)
    for train_part, val_part in folds:
        assert set(train_part).isdisjoint(val_part)
        assert np.array_equal(np.sort(np.concatenate([train_part, val_part])), base)

    again = kfold_indices(base, 5, seed=0)
    for (t1, v1), (t2, v2) in zip(folds, again, strict=True):
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)


def test_kfold_indices_rejects_bad_fold_count() -> None:
    with pytest.raises(ValueError):
        kfold_indices(np.arange(10), 1)
    with pytest.raises(ValueError):
        kfold_indices(np.arange(3), 5)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_yields_spectrum_and_thickness() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((17, N_CHANNELS)).astype(np.float64)  # 일부러 float64로 준다
    y = rng.integers(10, 301, size=(17, 4)).astype(np.int64)

    ds = FringeDataset(x, y)
    assert len(ds) == 17

    spectrum, thickness = ds[3]
    assert spectrum.shape == (N_CHANNELS,)
    assert thickness.shape == (4,)
    assert spectrum.dtype == torch.float32  # 학습 dtype으로 캐스팅되어야 한다
    assert thickness.dtype == torch.float32
    assert torch.allclose(spectrum, torch.tensor(x[3], dtype=torch.float32))


def test_dataset_without_targets_is_inference_mode() -> None:
    x = np.zeros((5, N_CHANNELS), dtype=np.float32)
    ds = FringeDataset(x)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (N_CHANNELS,)


def test_dataset_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        FringeDataset(np.zeros((5, N_CHANNELS)), np.zeros((4, 4)))


def test_load_frame_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        load_frame("valid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 실제 파일이 있을 때만 — 검증 스크립트가 확정한 데이터 계약을 로더가 지키는지
# ---------------------------------------------------------------------------
@requires_raw_data
def test_prepare_train_arrays_subset_is_random_and_reproducible() -> None:
    x1, y1, tr1, va1 = prepare_train_arrays(val_frac=0.1, seed=42, subset=1000)
    x2, _, tr2, va2 = prepare_train_arrays(val_frac=0.1, seed=42, subset=1000)

    assert x1.shape == (1000, N_CHANNELS)
    assert y1.shape == (1000, 4)
    assert len(va1) == 100
    # 같은 (seed, subset)이면 같은 분할 — evaluate.py가 holdout을 재현하는 근거
    assert np.array_equal(x1, x2)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    # 사전식 앞머리(x[:N])가 아님을 간접 확인 — layer_1이 구석 값에 몰려 있으면 안 된다
    assert len(np.unique(y1[:, 0])) > 5


@requires_raw_data
def test_train_frame_matches_verified_contract() -> None:
    frame = load_frame("train")
    assert frame.shape == (810_000, 230)
    assert list(frame.columns) == LAYER_COLS + CHANNEL_COLS

    head = frame.head(1000)
    thickness = head[LAYER_COLS].to_numpy()
    assert ((thickness >= 10) & (thickness <= 300)).all()
    assert (thickness % 10 == 0).all()  # 10 nm 격자
