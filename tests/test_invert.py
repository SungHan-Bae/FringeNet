"""두께 역해 LM 테스트 — 출발점 계약 · 청크 불변성 · 상자 제약.

이 최적화는 두 실험이 공유하고 **출발점만 다르다** (게이트 (d)는 d_true, 역산 refinement는
d_hat). 그래서 여기서 못 박는 것은 정확도가 아니라 계약이다: 출발점이 결과를 바꾼다 ·
행 감쇠는 청크 구성에 불변이다 · 배치 감쇠로 청크를 쓰면 조용히 달라지는 대신 에러가 난다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.losses import FrozenDecoder
from src.physics.invert import (
    DEFAULT_BOX_NM,
    inversion_stats,
    lm_invert,
    residual_l1_rows,
)


@pytest.fixture(scope="module")
def decoder() -> FrozenDecoder:
    """Stage A 확정 디코더 (git 추적, 44 KB) — complex128로 검증한다."""
    return FrozenDecoder(dtype=torch.complex128)


def _truth(n: int = 8, *, seed: int = 3) -> np.ndarray:
    """물리 범위 안 무작위 두께 (n, 4) float64."""
    rng = np.random.default_rng(seed)
    return rng.uniform(30.0, 280.0, size=(n, 4))


def _observe(decoder: FrozenDecoder, d: np.ndarray) -> np.ndarray:
    """노이즈 없는 합성 관측 R(d) — 역해가 되찾아야 할 정답이 d 자신이 된다."""
    with torch.no_grad():
        return decoder(torch.from_numpy(d)).numpy()


def test_recovers_thickness_from_noiseless_observation(decoder: FrozenDecoder) -> None:
    """디코더 자신이 만든 R이면 근처에서 출발한 LM이 두께를 되찾는다."""
    d_true = _truth()
    x = _observe(decoder, d_true)
    d_init = d_true + np.array([1.5, -2.0, 1.0, -1.5])
    d_hat = lm_invert(decoder, x, d_init, iters=30, damping="row")
    assert np.abs(d_hat - d_true).max() < 0.05


def test_starting_point_changes_the_answer(decoder: FrozenDecoder) -> None:
    """출발점이 결과를 바꾼다 — 이 함수의 계약 자체다 (게이트 (d) vs refinement).

    d_true 근처에서 출발하면 되찾고, 멀리서 출발하면 다른 분지에 갇힌다.
    """
    d_true = _truth()
    x = _observe(decoder, d_true)
    near = lm_invert(decoder, x, d_true + 1.0, iters=30, damping="row")
    far = lm_invert(decoder, x, np.full_like(d_true, 155.0), iters=30, damping="row")
    assert np.abs(near - d_true).mean() < 0.05
    assert np.abs(far - d_true).mean() > 1.0


def test_row_damping_is_chunk_invariant(decoder: FrozenDecoder) -> None:
    """행별 감쇠면 행이 독립이라 청크를 어떻게 끊어도 같은 답이 나온다."""
    d_true = _truth(n=7)
    x = _observe(decoder, d_true)
    d_init = d_true + 2.0
    whole = lm_invert(decoder, x, d_init, iters=8, damping="row")
    chunked = lm_invert(decoder, x, d_init, iters=8, damping="row", chunk=3)
    assert np.allclose(whole, chunked, rtol=0.0, atol=1e-9)


def test_batch_damping_refuses_chunking(decoder: FrozenDecoder) -> None:
    """배치 감쇠는 한 행의 갱신이 배치 구성에 의존한다 — 조용히 달라지는 대신 막는다."""
    d_true = _truth(n=5)
    x = _observe(decoder, d_true)
    with pytest.raises(ValueError, match="damping='row'"):
        lm_invert(decoder, x, d_true, iters=2, damping="batch", chunk=2)
    # 청크가 배치 전체를 덮으면 쪼개지지 않으므로 허용된다 (게이트 (d)의 호출 형태)
    lm_invert(decoder, x, d_true, iters=2, damping="batch", chunk=len(x))


def test_box_constraint_is_respected(decoder: FrozenDecoder) -> None:
    """상자를 좁히면 해가 그 안에 머문다 (기본 상자는 물리 범위보다 의도적으로 넓다)."""
    d_true = _truth(n=6)
    x = _observe(decoder, d_true)
    box = (100.0, 120.0)
    d_hat = lm_invert(decoder, x, np.full_like(d_true, 110.0), iters=10, box=box, damping="row")
    assert d_hat.min() >= box[0] - 1e-9
    assert d_hat.max() <= box[1] + 1e-9
    assert DEFAULT_BOX_NM[0] < 10.0 and DEFAULT_BOX_NM[1] > 300.0


def test_row_count_mismatch_raises(decoder: FrozenDecoder) -> None:
    d_true = _truth(n=4)
    x = _observe(decoder, d_true)
    with pytest.raises(ValueError, match="행이 다르다"):
        lm_invert(decoder, x[:3], d_true, iters=1)


def test_inversion_stats_matches_hand_computation() -> None:
    """통계는 손계산과 맞아야 한다 — 층별 편향의 부호가 뒤집히면 진단이 거짓말을 한다."""
    d_true = np.array([[100.0, 100.0], [200.0, 200.0]])
    d_hat = np.array([[101.0, 99.0], [198.0, 205.0]])
    stats = inversion_stats(d_hat, d_true, box=(1.0, 400.0))
    assert stats["mae"] == pytest.approx((1 + 1 + 2 + 5) / 4)
    assert stats["mae_per_layer"] == pytest.approx([1.5, 3.0])
    assert stats["bias_per_layer"] == pytest.approx([-0.5, 2.0])
    assert stats["abs_err_max"] == pytest.approx(5.0)
    assert stats["out_of_physical"] == 0.0
    assert stats["at_box_boundary"] == 0.0


def test_residual_rows_match_decoder_contract(decoder: FrozenDecoder) -> None:
    """행별 잔차는 FrozenDecoder.residual_l1과 같은 값이어야 한다 (신뢰도 지표 단일 정의)."""
    d = _truth(n=9)
    x = _observe(decoder, d) + 0.001
    mine = residual_l1_rows(decoder, d, x, chunk=4)
    theirs = decoder.residual_l1(torch.from_numpy(d), torch.from_numpy(x)).numpy()
    assert np.allclose(mine, theirs, rtol=0.0, atol=1e-12)
