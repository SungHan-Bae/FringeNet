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

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ADACHI_SI_PARAM_NAMES",
    "adachi_si_eps",
    "adachi_si_init",
    "adachi_si_nk",
    "cauchy_n",
    "fit_cauchy",
    "linear_interp_matrix",
    "si3n4_n",
    "si_nk",
    "sio2_n",
    "softplus_inverse",
]

# 광자 에너지 변환: E[eV] = _EV_NM / λ[nm].
_EV_NM = 1239.84198

# Sellmeier: n²(λ) = 1 + Σ_i B_i λ² / (λ² − C_i),  λ[μm].
_SIO2_SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)  # Malitson 1965
_SIO2_SELLMEIER_C_UM2 = (0.0684043**2, 0.1162414**2, 9.896161**2)
_SI3N4_SELLMEIER_B = (3.0249, 40314.0)  # Luke et al. 2015
_SI3N4_SELLMEIER_C_UM2 = (0.1353406**2, 1239.842**2)

# 결정질 Si의 n, k — Aspnes & Studna 1983 / Green 2008 근사 독취 (초기값 용도).
# 380 nm 미만은 E1(~365 nm)·E2(~290 nm) 임계점 구조의 거친 근사다 — 캘리브레이션
# knot이 학습으로 갱신하므로 초기값 정밀도는 결과에 중요하지 않다.
_SI_LAM_NM = np.array(
    [270.0, 290.0, 310.0, 335.0, 355.0, 368.0, 380.0, 400.0, 450.0, 500.0,
     550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 1000.0]
)  # fmt: skip
_SI_N = np.array(
    [2.6, 4.4, 5.0, 5.3, 5.6, 6.5, 6.06, 5.57, 4.67, 4.29,
     4.08, 3.94, 3.85, 3.78, 3.73, 3.69, 3.66, 3.63, 3.59]
)  # fmt: skip
_SI_K = np.array(
    [5.0, 4.4, 3.6, 3.1, 3.0, 2.0, 0.63, 0.387, 0.145, 0.071,
     0.033, 0.022, 0.016, 0.011, 0.0079, 0.0057, 0.0041, 0.003, 0.0005]
)  # fmt: skip


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


# --- Adachi 모델 유전함수(MDF) — Si 기판의 매끈한 물리 파라미터화 ------------------
#
# 함수형 출처: Adachi, Phys. Rev. B 38, 12966 (1988) / J. Appl. Phys. 66, 3224 (1989).
# 유료 원문 대신 식형을 재수록한 Petrik, Physica B 453, 2 (2014) §3.4를 참조했다.
# 대역 284–793 nm(1.56–4.37 eV)에 유효한 항만 취한다:
#   ε(E) = ε∞ + ε_E1(2D M0 임계점) + ε_E1x(여기자, 이산계열 1차항)
#          + ε_E0'(DHO — Adachi의 Si E0' 삼중항, E1 부근의 넓은 흡수 담당)
#          + ε_E2(DHO) + i·ε₂_ind(간접갭 흡수 — 장파장 k의 지배항)
# 근사(문서화된 한계):
#   - 간접갭 항은 ε₂만 모델링 (Adachi 1989와 동일). ε₁ 기여는 대역 내 ε₁(~15–40)
#     대비 미미해 ε∞·임계점 항이 흡수한다 → 이 항만 KK 비일관.
#   - 고에너지 컷오프 없음 — 대역 밖(E > 4.4 eV)에서는 평가하지 않는 계약.
# 진폭·broadening ≥ 0 (softplus)이면 모든 항의 Im ε ≥ 0 → k ≥ 0이 구조적으로 보장.
#
# 계수 초기값(adachi_si_init.json)은 Adachi 논문 테이블 전사가 아니라
# scripts/fit_adachi_init.py 가 위 함수형을 본 파일의 Aspnes & Studna 테이블에
# 결정론적으로 프리핏한 산출물이다 — 원문 테이블 접근 불가 + 수치는 스크립트
# 산출물이어야 한다는 프로젝트 규약(CLAUDE.md) 때문. 어차피 초기값 용도라
# 캘리브레이션이 학습으로 갱신한다.

ADACHI_SI_PARAM_NAMES = (
    "eps_inf",
    "D",
    "Eg",
    "B1",
    "B1x",
    "Gamma1",
    "E1",
    "C0",
    "gamma0",
    "E0p",
    "C",
    "gamma",
    "E2",
)


def adachi_si_eps(e_ev: Tensor, params: Tensor) -> Tensor:
    """Adachi MDF — 결정질 Si의 복소 유전함수 ε(E). 물리 관례 ε = ε₁ + i·ε₂ (ε₂ ≥ 0).

    Args:
        e_ev: (W,) float — 광자 에너지 [eV].
        params: (13,) float — ADACHI_SI_PARAM_NAMES 순서의 value 공간 계수 (전부 > 0).
            에너지(Eg, E1, E0p, E2)·broadening(Gamma1, gamma0, gamma) [eV],
            진폭(B1, B1x[eV], C0, C, D)·ε∞ 무차원.

    Returns:
        eps: (W,) complex — e_ev와 같은 실수 dtype의 복소 대응 dtype.
    """
    eps_inf, d_ind, eg, b1, b1x, gam1, e1, c0, gam0, e0p, c2, gam2, e2 = (
        params[i] for i in range(13)
    )
    # E1 — 2D M0 임계점: −B1·χ⁻²·ln(1−χ²), χ = (E + iΓ)/E1.
    chi1 = (e_ev + 1j * gam1) / e1
    chi1_sq = chi1 * chi1
    eps_e1 = -b1 * torch.log(1.0 - chi1_sq) / chi1_sq
    # E1 여기자 — 이산 계열(∝ n⁻³)의 1차항, E1 임계점과 broadening 공유.
    eps_e1x = b1x / (e1 - e_ev - 1j * gam1)
    # E0' — DHO: E1과 겹치는 삼중항, 넓은 broadening으로 E1 주변 흡수를 담당.
    chi0 = e_ev / e0p
    eps_e0p = c0 / (1.0 - chi0 * chi0 - 1j * chi0 * gam0)
    # E2 — 감쇠 조화 진동자(DHO): C / (1 − χ² − i·χ·γ), χ = E/E2.
    chi2 = e_ev / e2
    eps_e2 = c2 / (1.0 - chi2 * chi2 - 1j * chi2 * gam2)
    # 간접갭 — ε₂ = (D/E²)·(E − Eg)², E > Eg (ε₁ 기여는 무시: 상단 주석).
    eps2_ind = d_ind * torch.clamp(e_ev - eg, min=0.0) ** 2 / e_ev**2
    return eps_inf + eps_e1 + eps_e1x + eps_e0p + eps_e2 + 1j * eps2_ind


def adachi_si_nk(lam_nm: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    """Adachi MDF의 (n, k). λ·계수 모두로 미분가능.

    Args:
        lam_nm: (W,) float — 파장 [nm].
        params: (10,) — adachi_si_eps와 동일.

    Returns:
        (n, k): 각 (W,) float — ε₂ ≥ 0이므로 주가지 sqrt에서 n ≥ 0, k ≥ 0.
    """
    nk = torch.sqrt(adachi_si_eps(_EV_NM / lam_nm, params))
    return nk.real, nk.imag


def adachi_si_init() -> np.ndarray:
    """프리핏된 Adachi Si 초기 계수 (value 공간) — scripts/fit_adachi_init.py 산출물."""
    obj = json.loads((Path(__file__).with_name("adachi_si_init.json")).read_text())
    if obj["names"] != list(ADACHI_SI_PARAM_NAMES):
        raise ValueError("adachi_si_init.json의 계수 순서가 코드와 다르다 — 프리핏 재실행 필요")
    return np.asarray(obj["values"], dtype=np.float64)


def softplus_inverse(y: Tensor) -> Tensor:
    """softplus의 역함수: softplus(softplus_inverse(y)) == y (y > 0).

    y가 크면 softplus가 항등에 수렴하므로 y를 그대로 돌려 expm1 오버플로를 피한다.
    """
    safe = y.clamp(max=20.0)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(safe)))
