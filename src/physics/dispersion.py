"""문헌 분산(dispersion) 데이터와 Cauchy 유틸 — Stage A 캘리브레이션의 초기값·게이지 고정용.

수치 출처 (refractiveindex.info 수록 문헌):
  - SiO₂ (fused silica): Malitson 1965 Sellmeier. **게이지 고정(freeze) 대상** —
    delta = 2πnd/λ 가 (n, λ)의 공통 스케일에 불변이라 SiO₂ n(λ)를 문헌값에 고정해야
    λ 그리드가 식별된다 (CLAUDE.md Level 2 게이지 고정).
  - Si₃N₄: Luke et al. 2015 Sellmeier. 학습 파라미터(SiN Cauchy)의 초기값.
  - Si (결정질): Aspnes & Studna 1983 (≲830 nm) + Green 2008 (그 너머) 근사 독취 테이블.
    학습으로 갱신되는 초기값 용도라 소수점 아래 정밀도는 결과에 중요하지 않다.

파장축이 비식별화되어 있으므로 (CLAUDE.md 데이터 계약) 여기의 λ[nm]는 전부
"초기 가정 그리드"에서만 쓰이고, 실제 λ 그리드는 캘리브레이션이 학습한다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "cauchy_n",
    "fit_cauchy",
    "linear_interp_matrix",
    "si3n4_n",
    "si_nk",
    "sio2_n",
    "softplus_inverse",
]

# Sellmeier: n²(λ) = 1 + Σ_i B_i λ² / (λ² − C_i),  λ[μm].
_SIO2_SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)  # Malitson 1965
_SIO2_SELLMEIER_C_UM2 = (0.0684043**2, 0.1162414**2, 9.896161**2)
_SI3N4_SELLMEIER_B = (3.0249, 40314.0)  # Luke et al. 2015
_SI3N4_SELLMEIER_C_UM2 = (0.1353406**2, 1239.842**2)

# 결정질 Si의 n, k — Aspnes & Studna 1983 / Green 2008 근사 독취 (초기값 용도).
_SI_LAM_NM = np.array(
    [380.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 1000.0]
)
_SI_N = np.array([6.06, 5.57, 4.67, 4.29, 4.08, 3.94, 3.85, 3.78, 3.73, 3.69, 3.66, 3.63, 3.59])
_SI_K = np.array(
    [0.63, 0.387, 0.145, 0.071, 0.033, 0.022, 0.016, 0.011, 0.0079, 0.0057, 0.0041, 0.003, 0.0005]
)


def _sellmeier_n(lam_nm: np.ndarray, b: tuple[float, ...], c_um2: tuple[float, ...]) -> np.ndarray:
    """Sellmeier n(λ). lam_nm: (W,) [nm] → n: (W,) float64."""
    lam2 = (np.asarray(lam_nm, dtype=np.float64) * 1e-3) ** 2
    n2 = 1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c_um2, strict=True))
    return np.sqrt(n2)


def sio2_n(lam_nm: np.ndarray) -> np.ndarray:
    """SiO₂(fused silica) 굴절률 — Malitson 1965. lam_nm: (W,) → (W,) float64."""
    return _sellmeier_n(lam_nm, _SIO2_SELLMEIER_B, _SIO2_SELLMEIER_C_UM2)


def si3n4_n(lam_nm: np.ndarray) -> np.ndarray:
    """Si₃N₄ 굴절률 — Luke et al. 2015. lam_nm: (W,) → (W,) float64."""
    return _sellmeier_n(lam_nm, _SI3N4_SELLMEIER_B, _SI3N4_SELLMEIER_C_UM2)


def si_nk(lam_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """결정질 Si의 (n, k) — 테이블 선형 보간, 범위 밖은 끝값 유지. (W,) → ((W,), (W,))."""
    lam = np.asarray(lam_nm, dtype=np.float64)
    return np.interp(lam, _SI_LAM_NM, _SI_N), np.interp(lam, _SI_LAM_NM, _SI_K)


def fit_cauchy(lam_nm: np.ndarray, n: np.ndarray) -> np.ndarray:
    """n(λ) 표본에 Cauchy n = A + B/λ² + C/λ⁴ (λ[μm])를 최소제곱 피팅한다.

    Args:
        lam_nm: (W,) 파장 [nm].
        n: (W,) 굴절률.

    Returns:
        (3,) float64 — [A, B, C] (B: μm², C: μm⁴ 단위).
    """
    inv2 = (np.asarray(lam_nm, dtype=np.float64) * 1e-3) ** -2
    design = np.stack([np.ones_like(inv2), inv2, inv2**2], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, np.asarray(n, dtype=np.float64), rcond=None)
    return coeffs


def cauchy_n(lam_nm: Tensor, coeffs: Tensor) -> Tensor:
    """Cauchy n(λ) = A + B/λ² + C/λ⁴ (λ[μm]) — λ·계수 모두로 미분가능.

    Args:
        lam_nm: (W,) 파장 [nm].
        coeffs: (3,) — [A, B(μm²), C(μm⁴)].

    Returns:
        n: (W,) — lam_nm과 같은 dtype.
    """
    inv2 = (lam_nm * 1e-3) ** -2
    return coeffs[0] + coeffs[1] * inv2 + coeffs[2] * inv2**2


def linear_interp_matrix(n_points: int, n_knots: int) -> Tensor:
    """채널축 균등 knot의 조각별 선형 보간 행렬 P를 만든다: 곡선 = P @ knot값.

    knot는 채널 [0, n_points-1]에 균등 배치된다. 각 행은 볼록 결합(합 1, 음수 없음)
    이므로 knot값이 전부 비음수면 보간 결과도 비음수다.

    Returns:
        P: (n_points, n_knots) float64.
    """
    if n_knots < 2:
        raise ValueError(f"n_knots는 2 이상이어야 한다 (받은 값: {n_knots})")
    if n_points < 2:
        raise ValueError(f"n_points는 2 이상이어야 한다 (받은 값: {n_points})")
    t = torch.linspace(0.0, float(n_knots - 1), n_points, dtype=torch.float64)
    left = t.floor().long().clamp(max=n_knots - 2)
    frac = t - left.to(torch.float64)
    p = torch.zeros(n_points, n_knots, dtype=torch.float64)
    rows = torch.arange(n_points)
    p[rows, left] = 1.0 - frac
    p[rows, left + 1] = frac
    return p


def softplus_inverse(y: Tensor) -> Tensor:
    """softplus의 역함수: softplus(softplus_inverse(y)) == y (y > 0).

    y가 크면 softplus가 항등에 수렴하므로 y를 그대로 돌려 expm1 오버플로를 피한다.
    """
    safe = y.clamp(max=20.0)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(safe)))
