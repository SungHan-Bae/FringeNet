"""Stage B 물리 손실 테스트 — 동결 계약 · dtype 캐스팅 충실도 · beta 워밍업 · 누수.

물리 항이 켜졌는지는 학습 결과로는 확인이 안 된다 (동결이 깨져도 손실은 내려간다).
그래서 계약을 여기서 못 박는다: 디코더가 파라미터를 보유하지 않는다 · float32 캐스팅
오차가 노이즈보다 두 자릿수 작다 · beta=0 대조군의 gradient가 물리 항 도입 전과
**비트 단위로 같다** · 캘리브레이션에 쓴 행이 평가 holdout에 없다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.functional import l1_loss

from src.calibrate import (
    _SPLIT_FIT_ROWS,
    _SPLIT_SEED,
    NOISE_BOUND,
    NOISE_SIGMA,
    load_physical_stack,
    load_split,
)
from src.data.dataset import RAW_DIR, REPO_ROOT, prepare_train_arrays
from src.losses import (
    DEFAULT_DECODER,
    FrozenDecoder,
    PhysicsLoss,
    beta_at,
    build_physics_loss,
)

requires_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "train.csv").exists(),
    reason="data/raw/train.csv 없음 (대회 데이터는 저장소에 포함하지 않는다)",
)


def _random_thickness(n: int, *, seed: int = 0, layers: int = 4) -> torch.Tensor:
    """물리 범위 [10, 300] nm 무작위 두께 (B, L) float64."""
    gen = torch.Generator().manual_seed(seed)
    return 10.0 + 290.0 * torch.rand(n, layers, generator=gen, dtype=torch.float64)


def _grid_keys(d: np.ndarray) -> np.ndarray:
    """두께 조합 → 유일 정수 키. train은 30⁴ 전수 조합이라 행마다 키가 다르다."""
    idx = np.rint(d / 10.0).astype(np.int64) - 1
    assert idx.min() >= 0 and idx.max() <= 29, "두께가 10 nm 격자 밖이다"
    return ((idx[:, 0] * 30 + idx[:, 1]) * 30 + idx[:, 2]) * 30 + idx[:, 3]


# ---------------------------------------------------------------------------
# 동결 계약
# ---------------------------------------------------------------------------
def test_decoder_holds_no_parameters() -> None:
    """옵티마이저에 넘길 파라미터가 아예 없어야 동결이 구조적으로 보장된다."""
    dec = FrozenDecoder()
    assert list(dec.parameters()) == []
    assert set(dec.state_dict()) == {"lam", "n_layers", "ns"}
    assert all(not buf.requires_grad for buf in dec.buffers())


def test_decoder_matches_physical_stack_in_float64() -> None:
    """래퍼가 물리를 바꾸지 않는다 — Stage A 모델과 같은 값."""
    stack, _ = load_physical_stack(REPO_ROOT / DEFAULT_DECODER)
    dec = FrozenDecoder(dtype=torch.complex128)
    d = _random_thickness(16)
    with torch.no_grad():
        assert torch.equal(dec(d), stack(d))


def test_float32_cast_error_is_negligible() -> None:
    """학습 dtype(complex64) 캐스팅 오차가 노이즈 σ보다 두 자릿수 이상 작다."""
    d = _random_thickness(256, seed=7)
    with torch.no_grad():
        r64 = FrozenDecoder(dtype=torch.complex128)(d)
        r32 = FrozenDecoder(dtype=torch.complex64)(d.to(torch.float32))
    err = (r32.to(torch.float64) - r64).abs().max().item()
    assert err < NOISE_SIGMA / 100.0, f"캐스팅 오차 {err:.3e}가 너무 크다"
    assert err < NOISE_BOUND / 100.0


def test_gradient_flows_to_thickness() -> None:
    dec = FrozenDecoder()
    d = _random_thickness(8, seed=1).to(torch.float32).requires_grad_(True)
    dec(d).sum().backward()
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert (d.grad.abs() > 0).any()


def test_reconstruct_matches_forward_across_batches() -> None:
    dec = FrozenDecoder()
    d = _random_thickness(37, seed=3).to(torch.float32)
    assert torch.equal(dec.reconstruct(d, batch_size=8), dec(d))
    assert dec.reconstruct(d[:0]).shape == (0, dec.n_channels)


# ---------------------------------------------------------------------------
# beta 스케줄
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (750, 25.0), (1500, 50.0), (3000, 100.0), (30_000, 100.0)],
)
def test_beta_warmup_is_linear_then_flat(step: int, expected: float) -> None:
    assert beta_at(step, 100.0, 3000) == pytest.approx(expected)


def test_beta_at_edge_cases() -> None:
    assert beta_at(0, 100.0, 0) == 100.0  # 워밍업 없음
    assert beta_at(5000, 0.0, 3000) == 0.0  # 대조군은 언제나 0
    with pytest.raises(ValueError, match="step"):
        beta_at(-1, 100.0, 3000)


# ---------------------------------------------------------------------------
# 손실 조립
# ---------------------------------------------------------------------------
def test_beta_zero_leaves_supervised_gradient_bitwise_identical() -> None:
    """대조군의 학습 경로가 물리 항 도입 전과 정확히 같아야 ablation 차이를 귀속할 수 있다."""
    torch.manual_seed(0)
    model = nn.Linear(226, 4)
    x = torch.rand(32, 226)
    d_true = 10.0 + 290.0 * torch.rand(32, 4)
    params = list(model.parameters())

    g_plain = torch.autograd.grad(l1_loss(model(x), d_true), params)
    loss = PhysicsLoss(FrozenDecoder(), beta=0.0)
    parts = loss(model(x), d_true, x, step=10)
    g_phys = torch.autograd.grad(parts.total, params)

    for a, b in zip(g_plain, g_phys, strict=True):
        assert torch.equal(a, b)
    assert parts.beta == 0.0
    assert not parts.phys.requires_grad  # 진단값으로만 기록된다
    assert torch.equal(parts.total, parts.sup)


def test_physics_term_is_minimized_at_true_thickness() -> None:
    """물리 항의 스케일과 방향 — 참 두께에서 E|ε|=0.0075, 벗어나면 커진다."""
    dec = FrozenDecoder()
    d_true = _random_thickness(64, seed=5).to(torch.float32)
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        noise = (torch.rand(64, dec.n_channels, generator=gen) * 2.0 - 1.0) * 0.015
        r_obs = dec(d_true) + noise

    loss = PhysicsLoss(dec, beta=100.0)
    at_true = loss(d_true, d_true, r_obs, step=0).phys.item()
    at_off = loss(d_true + 5.0, d_true, r_obs, step=0).phys.item()

    assert at_true == pytest.approx(0.0075, abs=0.002)  # 균등 ±0.015의 평균 절대값
    assert at_true < at_off


def test_total_is_supervised_plus_weighted_physics() -> None:
    dec = FrozenDecoder()
    d_true = _random_thickness(16, seed=9).to(torch.float32)
    d_hat = d_true + 3.0
    r_obs = dec(d_true)
    loss = PhysicsLoss(dec, beta=100.0, warmup_steps=1000)
    parts = loss(d_hat, d_true, r_obs, step=500)  # 워밍업 절반 → beta 50
    assert parts.beta == pytest.approx(50.0)
    assert parts.sup.item() == pytest.approx(3.0, abs=1e-4)
    assert parts.total.item() == pytest.approx(parts.sup.item() + 50.0 * parts.phys.item())


# ---------------------------------------------------------------------------
# config 조립
# ---------------------------------------------------------------------------
def test_build_physics_loss_returns_none_without_block() -> None:
    assert build_physics_loss({"epochs": 1, "batch_size": 512}) is None


def test_build_physics_loss_rejects_typo_and_missing_beta() -> None:
    with pytest.raises(ValueError, match="모르는 physics 키"):
        build_physics_loss({"physics": {"beta": 1.0, "warmup": 10}})
    with pytest.raises(ValueError, match="beta"):
        build_physics_loss({"physics": {"warmup_steps": 10}})


def test_build_physics_loss_defaults_to_adopted_decoder() -> None:
    loss = build_physics_loss({"physics": {"beta": 30.0, "warmup_steps": 100}})
    assert loss is not None
    assert loss.beta == 30.0 and loss.warmup_steps == 100
    cfg = loss.config
    assert cfg["decoder"] == DEFAULT_DECODER
    assert cfg["si_source"] == "Si_nk_Schinke.yml"
    assert len(cfg["free"]) == 7  # Stage A 확정 자유도
    assert cfg["stage_a_rmse"] == pytest.approx(0.009573, abs=5e-6)


# ---------------------------------------------------------------------------
# 누수 (사전등록 항목 4)
# ---------------------------------------------------------------------------
@requires_raw_data
def test_calibration_rows_are_disjoint_from_holdout() -> None:
    """캘리브레이션 표집(fit + diag) ∩ 평가 holdout = 0행.

    전수 조합이라 두께 4개 조합이 행의 유일 키가 된다 — 표본이 아니라 전수 대조다.
    """
    split = load_split(_SPLIT_FIT_ROWS)
    used_keys = _grid_keys(np.concatenate([split["d_fit"], split["d_diag"]]))

    arrays = prepare_train_arrays(val_frac=0.1, seed=_SPLIT_SEED)
    y, holdout_idx = arrays[1], arrays[3]
    del arrays
    holdout_keys = _grid_keys(y[holdout_idx])

    assert len(np.unique(holdout_keys)) == len(holdout_keys)
    assert len(np.unique(used_keys)) == len(used_keys)
    assert np.intersect1d(holdout_keys, used_keys).size == 0
