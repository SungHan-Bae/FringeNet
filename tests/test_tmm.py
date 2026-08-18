"""미분가능 TMM forward 모델의 물리 단위 테스트 (CLAUDE.md §물리 단위 테스트).

전부 complex128/float64로 수행한다. 여기서 검증하는 것은 구현이 아니라 **물리**다 —
해석적으로 답이 알려진 극한과 보존 법칙만 사용하며, 어떤 tolerance도 구현에 맞춰
완화하지 않는다.

단위 규약: d와 lam은 같은 길이 단위(nm)를 쓴다. delta = 2*pi*n*d/lam 는 무차원.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.physics.tmm import tmm_reflectance, tmm_reflectance_jacobian, tmm_rt

DTYPE_R = torch.float64
DTYPE_C = torch.complex128


def _lam_grid(w: int = 16, lo: float = 400.0, hi: float = 800.0) -> torch.Tensor:
    """(W,) 파장 그리드 [nm]."""
    return torch.linspace(lo, hi, w, dtype=DTYPE_R)


def _const_n(values: list[float] | list[complex], w: int) -> torch.Tensor:
    """층별 상수 굴절률을 (L, W) complex 텐서로 편다 (분산 없음)."""
    n = torch.tensor(values, dtype=DTYPE_C).reshape(-1, 1)
    return n.expand(-1, w).contiguous()


# ---------------------------------------------------------------------------
# 1. 무층 극한 — d=0 이면 맨 기판의 프레넬 반사를 회복해야 한다.
# ---------------------------------------------------------------------------
def test_zero_thickness_recovers_bare_fresnel() -> None:
    w = 16
    lam = _lam_grid(w)
    n0, ns = 1.0, 1.5
    n_layers = _const_n([2.0, 1.46, 2.0, 1.46], w)  # 값은 무관해야 한다
    d = torch.zeros(3, 4, dtype=DTYPE_R)

    r_out = tmm_reflectance(d, n_layers, n0, ns, lam)

    r_expected = ((n0 - ns) / (n0 + ns)) ** 2  # = 0.04
    assert r_out.shape == (3, w)
    assert r_out.dtype == DTYPE_R
    assert torch.allclose(r_out, torch.full_like(r_out, r_expected), atol=1e-12)
    assert r_expected == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 2. 에너지 보존 — 무흡수(실수 n) 스택에서 R + T = 1.
# ---------------------------------------------------------------------------
def test_energy_conservation_lossless_stack() -> None:
    torch.manual_seed(0)
    b, ell, w = 8, 4, 32
    lam = _lam_grid(w)
    n0 = 1.0
    ns = 3.8  # 실수 → 무흡수 기판

    # 무작위 스택: n ∈ [1.3, 2.5], d ∈ [10, 300] nm
    n_real = 1.3 + 1.2 * torch.rand(ell, w, dtype=DTYPE_R)
    n_layers = n_real.to(DTYPE_C)
    d = 10.0 + 290.0 * torch.rand(b, ell, dtype=DTYPE_R)

    r_out, t_out = tmm_rt(d, n_layers, n0, ns, lam)

    total = r_out + t_out
    assert torch.isfinite(total).all()
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)

    # 보강: **흡수 기판 + n0 ≠ 1**. 무흡수 층 스택에서는 복소 ns여도 R + T = 1이 그대로
    # 성립한다 (T = 4·n0·Re(ns)/|n0B+C|² 가 기판으로 들어가는 전력이므로). n0를 1에서
    # 떼고 ns에 허수부를 주면 T 식의 n0 계수·Re(ns) 취급·복소 부호 관례를 동시에 건다 —
    # n0 = 1, 실수 ns만 쓰면 이 넷을 잘못 써도 통과한다.
    for n0_abs, ns_abs in ((1.31, complex(3.8, -0.5)), (1.31, complex(4.5, -2.0))):
        r_abs, t_abs = tmm_rt(d, n_layers, n0_abs, ns_abs, lam)
        assert torch.isfinite(r_abs).all() and torch.isfinite(t_abs).all()
        assert (t_abs >= 0).all()
        assert torch.allclose(r_abs + t_abs, torch.ones_like(r_abs), atol=1e-9)


# ---------------------------------------------------------------------------
# 3. λ/4 무반사 — n1 = sqrt(n0*ns), d = λ/(4 n1) 에서 반사가 사라진다.
# ---------------------------------------------------------------------------
def test_quarter_wave_antireflection() -> None:
    n0, ns = 1.0, 2.25
    n1 = 1.5  # = sqrt(1.0 * 2.25)
    assert n1 == pytest.approx(math.sqrt(n0 * ns))

    lam0 = 550.0
    lam = torch.tensor([lam0], dtype=DTYPE_R)
    n_layers = _const_n([n1], 1)  # 단층 (L=1)
    d = torch.tensor([[lam0 / (4.0 * n1)]], dtype=DTYPE_R)

    r_out = tmm_reflectance(d, n_layers, n0, ns, lam)

    assert r_out.shape == (1, 1)
    assert r_out.item() < 1e-8


# ---------------------------------------------------------------------------
# 4. Airy 대조 — 단층 TMM 이 해석해와 일치해야 한다.
#    r = (r01 + r12 e^{-2i delta}) / (1 + r01 r12 e^{-2i delta})
# ---------------------------------------------------------------------------
def test_single_layer_matches_airy_formula() -> None:
    w = 64
    lam = _lam_grid(w, 380.0, 900.0)
    n0 = 1.0
    ns = 3.5
    n1 = 2.05
    n_layers = _const_n([n1], w)
    d = torch.tensor([[120.0], [37.5], [255.0]], dtype=DTYPE_R)  # (B=3, L=1)

    r_out = tmm_reflectance(d, n_layers, n0, ns, lam)

    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - ns) / (n1 + ns)
    delta = 2.0 * math.pi * n1 * d / lam  # (B, W)
    phase = torch.exp(-2j * delta.to(DTYPE_C))
    r_airy = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    r_expected = (r_airy.abs() ** 2).to(DTYPE_R)

    assert torch.allclose(r_out, r_expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. 미분가능성 — autograd dR/dd 가 중심 유한차분과 일치해야 한다.
# ---------------------------------------------------------------------------
def test_gradient_matches_finite_difference() -> None:
    torch.manual_seed(1)
    b, ell, w = 2, 4, 8
    lam = _lam_grid(w)
    n0, ns = 1.0, 3.8
    n_layers = _const_n([2.02, 1.46, 2.02, 1.46], w)
    d0 = torch.tensor([[95.0, 140.0, 60.0, 210.0], [175.0, 25.0, 285.0, 130.0]], dtype=DTYPE_R)
    assert d0.shape == (b, ell)

    def scalar_objective(d: torch.Tensor) -> torch.Tensor:
        # 스칼라로 축약해야 유한차분과 직접 비교할 수 있다.
        weights = torch.linspace(0.5, 1.5, w, dtype=DTYPE_R)
        return (tmm_reflectance(d, n_layers, n0, ns, lam) * weights).sum()

    d = d0.clone().requires_grad_(True)
    scalar_objective(d).backward()
    grad_auto = d.grad
    assert grad_auto is not None
    assert torch.isfinite(grad_auto).all()

    eps = 1e-5
    grad_fd = torch.zeros_like(d0)
    for i in range(b):
        for j in range(ell):
            dp, dm = d0.clone(), d0.clone()
            dp[i, j] += eps
            dm[i, j] -= eps
            with torch.no_grad():
                grad_fd[i, j] = (scalar_objective(dp) - scalar_objective(dm)) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# 6. 흡수 기판 — 복소 ns 에서도 0 <= R < 1 이고 NaN/Inf 가 없어야 한다.
# ---------------------------------------------------------------------------
def test_absorbing_substrate_is_stable() -> None:
    torch.manual_seed(2)
    b, ell, w = 16, 4, 32
    lam = _lam_grid(w)
    n0 = 1.0
    # Si 기판풍 복소 굴절률: 파장에 따라 흡수가 변하도록 (W,) 로 준다.
    # Macleod 관례 n = n' - i*k (k >= 0 이 흡수) 이므로 허수부는 음수다.
    ns = torch.complex(
        torch.linspace(3.5, 5.5, w, dtype=DTYPE_R),
        -torch.linspace(0.02, 1.5, w, dtype=DTYPE_R),
    )
    n_layers = _const_n([2.02, 1.46, 2.02, 1.46], w)
    d = 10.0 + 290.0 * torch.rand(b, ell, dtype=DTYPE_R)

    r_out = tmm_reflectance(d, n_layers, n0, ns, lam)

    assert r_out.shape == (b, w)
    assert torch.isfinite(r_out).all()
    assert (r_out >= 0.0).all()
    assert (r_out < 1.0).all()


# ---------------------------------------------------------------------------
# 7. 적층 순서 — CLAUDE.md 명세 6종에 대한 보강.
#
#    변이 검사 결과 위 6종은 층 순서를 뒤집어도 전부 통과한다(에너지 보존과
#    R<1은 순서 불변이고, 나머지는 단층이거나 d=0이라 순서를 보지 않는다).
#    타깃이 layer_1..layer_4 각각의 두께이므로 순서가 어긋난 forward 모델은
#    "그럴듯하지만 층이 뒤바뀐" 물리 손실을 만든다. 비대칭 2층 스택을
#    행렬식과 독립인 재귀 프레넬 공식으로 대조해 순서를 고정한다.
# ---------------------------------------------------------------------------
def test_layer_order_matches_recursive_fresnel() -> None:
    w = 48
    lam = _lam_grid(w, 400.0, 850.0)
    n0, ns = 1.0, 3.9
    n1, n2 = 2.05, 1.46  # 비대칭: 위가 SiN, 아래가 SiO2
    n_layers = _const_n([n1, n2], w)
    d = torch.tensor([[35.0, 210.0], [180.0, 55.0]], dtype=DTYPE_R)  # (B=2, L=2)

    r_out = tmm_reflectance(d, n_layers, n0, ns, lam)

    # 기판 쪽에서부터 쌓아 올라가는 재귀 프레넬 — 특성행렬과 무관한 유도.
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    r2s = (n2 - ns) / (n2 + ns)
    delta1 = (2.0 * math.pi * n1 * d[:, 0:1] / lam).to(DTYPE_C)
    delta2 = (2.0 * math.pi * n2 * d[:, 1:2] / lam).to(DTYPE_C)
    ph1, ph2 = torch.exp(-2j * delta1), torch.exp(-2j * delta2)

    r_lower = (r12 + r2s * ph2) / (1.0 + r12 * r2s * ph2)
    r_total = (r01 + r_lower * ph1) / (1.0 + r01 * r_lower * ph1)
    r_expected = (r_total.abs() ** 2).to(DTYPE_R)

    assert torch.allclose(r_out, r_expected, rtol=1e-10, atol=1e-12)

    # 순서를 뒤집으면 실제로 달라져야 한다 — 그래야 위 단언이 순서를 고정한다.
    r_swapped = tmm_reflectance(d.flip(1), n_layers.flip(0), n0, ns, lam)
    assert not torch.allclose(r_out, r_swapped, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 8. 해석적 야코비안 — autograd와 같아야 하고, R은 forward와 비트 동일해야 한다.
#    (LM 역해가 두 경로의 R을 같은 잔차 장부에 섞어 쓴다 — invert.lm_invert)
# ---------------------------------------------------------------------------
def test_analytic_jacobian_matches_autograd() -> None:
    torch.manual_seed(4)
    b, ell, w = 2, 4, 6
    lam = _lam_grid(w, 300.0, 800.0)
    n0 = 1.0
    # 분산·흡수를 모두 켠다 — 복소 n의 부호 관례가 도함수에서 어긋나면 여기서 걸린다.
    n_layers = _const_n([2.02, 1.46, 2.02, 1.46], w) - 0.01j * torch.rand(ell, 1, dtype=DTYPE_C)
    ns = torch.full((w,), 3.9 - 0.3j, dtype=DTYPE_C)
    d = torch.tensor([[95.0, 140.0, 60.0, 210.0], [175.0, 25.0, 285.0, 130.0]], dtype=DTYPE_R)

    r_out, jac = tmm_reflectance_jacobian(d, n_layers, n0, ns, lam)
    assert jac.shape == (b, w, ell)

    # R은 반올림까지 forward와 같아야 한다 (수학적 동치로는 부족하다).
    assert torch.equal(r_out, tmm_reflectance(d, n_layers, n0, ns, lam))

    # autograd를 참조값으로 쓴다 — 유한차분과 달리 절단오차가 없다.
    dg = d.clone().requires_grad_(True)
    r_graph = tmm_reflectance(dg, n_layers, n0, ns, lam)
    for i in range(b):
        for k in range(w):
            grad = torch.autograd.grad(r_graph[i, k], dg, retain_graph=True)[0]
            assert torch.allclose(jac[i, k], grad[i], rtol=1e-11, atol=1e-14)


def test_analytic_jacobian_handles_single_layer() -> None:
    """L=1이면 접두 곱이 항등이라 곱 순서 버그가 드러나지 않는다 — 따로 건다."""
    w = 8
    lam = _lam_grid(w)
    n_layers = _const_n([2.02], w)
    d = torch.tensor([[123.0], [45.0]], dtype=DTYPE_R)

    r_out, jac = tmm_reflectance_jacobian(d, n_layers, 1.0, 3.8, lam)
    assert jac.shape == (2, w, 1)
    assert torch.equal(r_out, tmm_reflectance(d, n_layers, 1.0, 3.8, lam))

    dg = d.clone().requires_grad_(True)
    r_graph = tmm_reflectance(dg, n_layers, 1.0, 3.8, lam)
    grad = torch.autograd.grad(r_graph.sum(), dg)[0]
    assert torch.allclose(jac.sum(dim=1), grad, rtol=1e-11, atol=1e-14)
