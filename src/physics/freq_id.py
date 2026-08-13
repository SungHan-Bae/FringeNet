"""두께축 주파수 식별 — 비식별 파장축 λ(c)의 **결정론적 닫힌형** 복원.

Stage A의 첫 단계로 본 피팅(`src/calibrate.py`)의 λ 초기값을 만든다.

λ축 fringe 정렬은 경사하강에 비볼록이어서 초기값이 틀리면 잘못된 fringe 차수 분지에
안착한다 (λ 범위 후보를 짧게 피팅해 보는 방식은 전부 RMSE ~0.12에서 정체했다). 대신
타깃이 30⁴ **전수 격자**인 점을 쓴다:

1. 층 j로 조건화한 평균 `g_c(d) = E[R_c | d_j = d]` (30점)는 다른 층의 변동이 주변화로
   지워지고 두께축에서 기본 주파수 **f_j(c) = 2·n_j(λ_c)/λ_c** [cycles/nm] 로 진동한다
   (특성행렬이 δ→δ+π에서 부호만 바뀌므로 R은 δ_j에 π 주기).
2. SiO₂ 층(2·4)은 게이지 고정으로 n(λ)를 아니까 `2·n_SiO₂(λ)/λ = f` 를 λ에 대해 푼다.
   **g(λ) = 2n/λ 가 단조 감소**하므로 이분법으로 유일해가 나온다.
3. SiN 층(1·3)은 같은 관계에서 n_SiN(λ_c) = f·λ_c/2 추정을 준다.

경사하강이 개입하지 않으므로 같은 입력이면 항상 같은 출력이다. **자체 검증**은 같은
물리량을 다른 층에서 독립 측정한 일치도로 한다 (`diagnostics`의 `lam24_dev_*`,
`n_sin13_dev_*`).

**주의**: 반환된 채널별 λ에는 ~0.44 nm의 흔들림이 있다. λ(c)의 매끈한 구조가 아니고
정체는 **결정론적 추정량 편향**이다 (주파수 후보 격자 양자화 + 기저 절단 — 노이즈를 0으로
둔 합성 전수격자에서도 같은 크기가 나온다). 그대로 고정하면 R 오차 rms 0.0052가 생기므로
본 피팅은 매끈한 3계수 곡선을 적합해 쓴다 (`calibrate.fit_lam_coefficients`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.physics.dispersion import si3n4_n, sio2_n

__all__ = ["describe_identification", "identify_wavelength_grid"]

# 탐색 주파수 구간 [cycles/nm] — λ 100~2000 nm, n 1.4~2.3을 넉넉히 덮는다.
DEFAULT_F_RANGE = (0.003, 0.022)
# 층 2·4의 독립 λ 추정이 이보다 어긋나면 주파수 추정 실패로 보고 이웃 보간한다 [nm].
LAM_AGREEMENT_TOL = 5.0
# SiN 굴절률의 물리적 하한 — 이보다 낮으면 주파수 추정 실패다 (박막 SiN은 1.9~2.3).
N_SIN_FLOOR = 1.8


def _solve_lam_from_frequency(f: np.ndarray) -> np.ndarray:
    """2·n_SiO₂(λ)/λ = f 를 λ에 대해 이분법으로 푼다 (게이지: 정확한 Malitson).

    g(λ) = 2n(λ)/λ 는 정상 분산 구간에서 강한 단조 감소라 해가 유일하다.

    Args:
        f: (W,) 두께축 기본 주파수 [cycles/nm].

    Returns:
        (W,) λ [nm].
    """
    lo = np.full_like(f, 100.0)
    hi = np.full_like(f, 2000.0)
    for _ in range(60):  # 2000/2^60 → float64 정밀도 한계까지
        mid = 0.5 * (lo + hi)
        above = 2.0 * sio2_n(mid) / mid > f
        lo[above] = mid[above]
        hi[~above] = mid[~above]
    return 0.5 * (lo + hi)


def identify_wavelength_grid(
    x: np.ndarray,
    d: np.ndarray,
    *,
    f_range: tuple[float, float] = DEFAULT_F_RANGE,
    n_candidates: int = 1200,
) -> dict[str, Any]:
    """채널별 λ와 n_SiN을 두께축 주파수에서 식별한다 (닫힌형, 결정론).

    Args:
        x: (N, W) float — R_obs. **표본이 클수록 좋다** — 조건부 평균이 전수 격자의
            정확한 주변화에 가까워야 다른 층의 변동이 지워진다. holdout 제외 train
            전체(~73만 행, bin당 ~2.4만 행)를 넘기는 것이 기본이다 (5만 행 표본은
            bin 노이즈로 주파수 추정이 눈에 띄게 흔들린다 — 합성 실험 확인).
        d: (N, 4) — 두께 [nm] (10 nm 격자, 전수 조합이어야 한다).
        f_range: 탐색할 주파수 구간 [cycles/nm].
        n_candidates: 주파수 격자 분해능.

    Returns:
        {"lam_grid": (W,) 채널 순서 λ [nm],
         "n_sin_samples": ((W,) λ, (W,) n_SiN),
         "diagnostics": {...}} — 자체 검증 수치는 diagnostics 참조.
    """
    n_ch = x.shape[1]
    grid_vals = np.unique(d.reshape(-1))
    d64 = d.astype(np.float64)

    # 층별 조건부 평균 g[j, v, c] = E[R_c | d_j = grid_vals[v]].
    g = np.zeros((4, len(grid_vals), n_ch), dtype=np.float64)
    for j in range(4):
        for v, val in enumerate(grid_vals):
            rows = d64[:, j] == val
            if not rows.any():
                raise ValueError(f"layer_{j + 1} = {val} nm 행이 없다 — 표본이 격자를 못 덮음")
            g[j, v] = x[rows].mean(axis=0, dtype=np.float64)

    # 주파수 추정: 후보 f마다 투영행렬을 미리 만들고(채널·층 공용) 잔차 최소 f를 고른다.
    # 기저 = [1, d, cos, sin, cos2, sin2] — 완만한 배경 + 2배음까지 흡수한다.
    d_c = grid_vals - grid_vals.mean()
    f_grid = np.linspace(f_range[0], f_range[1], n_candidates)
    freqs = np.zeros((4, n_ch))
    for j in range(4):
        sig = g[j] - g[j].mean(axis=0, keepdims=True)  # (V, W)
        best_res = np.full(n_ch, np.inf)
        for f in f_grid:
            w = 2.0 * np.pi * f * d_c
            basis = np.stack(
                [np.ones_like(d_c), d_c, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)],
                axis=1,
            )
            proj = basis @ np.linalg.pinv(basis)
            res = ((sig - proj @ sig) ** 2).sum(axis=0)
            better = res < best_res
            best_res[better] = res[better]
            freqs[j, better] = f

    lam2, lam4 = _solve_lam_from_frequency(freqs[1]), _solve_lam_from_frequency(freqs[3])
    lam_grid = 0.5 * (lam2 + lam4)
    n_sin1 = freqs[0] * lam_grid / 2.0
    n_sin3 = freqs[2] * lam_grid / 2.0
    n_sin = 0.5 * (n_sin1 + n_sin3)

    # 신뢰도 마스크: 층 2·4의 λ 추정이 어긋나거나 n_SiN이 물리 하한 미만인 채널은 추정
    # 실패로 보고 이웃에서 선형 보간한다 (λ가 채널축에서 매끈하다는 물리 가정이 근거).
    # 실데이터에서는 Si E1 근방 fringe 대비 저하로 소수 채널이 크게 튄다. **λ 불일치만으로
    # 판정하면 SiN 주파수 실패를 놓쳐** 물리적으로 불가능한 n_SiN ≈ 1.54가 남는다.
    bad_lam = np.abs(lam2 - lam4) > LAM_AGREEMENT_TOL
    bad_sin = n_sin < N_SIN_FLOOR
    bad = bad_lam | bad_sin
    n_bad = int(bad.sum())
    if 0 < n_bad <= n_ch - 2:
        ch = np.arange(n_ch, dtype=np.float64)
        lam_grid[bad] = np.interp(ch[bad], ch[~bad], lam_grid[~bad])
        n_sin[bad] = np.interp(ch[bad], ch[~bad], n_sin[~bad])

    # 단조 강제: 추정 오차로 국소 요철이 있으면 누적 최솟값/최댓값으로 살짝 보정한다
    # (매끈 파라미터화의 초기값 조건).
    direction = -1.0 if lam_grid[0] > lam_grid[-1] else 1.0
    mono = np.minimum.accumulate(lam_grid) if direction < 0 else np.maximum.accumulate(lam_grid)
    fix = mono != lam_grid
    if np.any(np.diff(mono) * direction <= 0):
        mono = mono + direction * np.arange(n_ch) * 1e-6  # 동률 제거 (강단조)
    lam_grid = mono

    reliable = ~bad
    if not reliable.any():
        raise ValueError(
            "모든 채널의 주파수 식별이 실패했다 — 표본이 전수 격자를 덮는지, "
            f"f_range {f_range} 가 실제 주파수를 포함하는지 확인할 것"
        )
    diagnostics = {
        "lam24_dev_median": float(np.median(np.abs(lam2 - lam4))),
        "lam24_dev_max": float(np.abs(lam2 - lam4).max()),
        "lam_range": [float(lam_grid.min()), float(lam_grid.max())],
        "descending": bool(lam_grid[0] > lam_grid[-1]),
        "n_sin_range": [float(n_sin.min()), float(n_sin.max())],
        "n_sin_vs_luke_reldev_median": float(
            np.median(np.abs(n_sin - si3n4_n(lam_grid)) / si3n4_n(lam_grid))
        ),
        # 층 1 vs 층 3 — 같은 재료를 서로 다른 층에서 독립 측정한 것이므로 일치해야 한다.
        "n_sin13_dev_median": float(np.median(np.abs(n_sin1 - n_sin3)[reliable])),
        "n_sin13_dev_max": float(np.abs(n_sin1 - n_sin3)[reliable].max()),
        "monotone_fixups": int(fix.sum()),
        "unreliable_channels": n_bad,
        "unreliable_by_lam": int(bad_lam.sum()),
        "unreliable_by_n_sin": int(bad_sin.sum()),
    }
    return {"lam_grid": lam_grid, "n_sin_samples": (lam_grid, n_sin), "diagnostics": diagnostics}


def describe_identification(diag: dict[str, Any]) -> str:
    """진단 dict을 train.log 한 줄로 요약한다."""
    return (
        f"[freq-id] λ {diag['lam_range'][0]:.1f}–{diag['lam_range'][1]:.1f} nm "
        f"{'내림' if diag['descending'] else '오름'}차순 / 층2·4 λ 편차 중앙값 "
        f"{diag['lam24_dev_median']:.2f} nm (최대 {diag['lam24_dev_max']:.2f}) / "
        f"층1·3 n_SiN 편차 중앙값 {diag['n_sin13_dev_median']:.4f} "
        f"(최대 {diag['n_sin13_dev_max']:.4f}) / "
        f"n_SiN {diag['n_sin_range'][0]:.3f}–{diag['n_sin_range'][1]:.3f} "
        f"(Luke 대비 중앙 상대편차 {diag['n_sin_vs_luke_reldev_median']:.3%}) / "
        f"불신 채널 보간 {diag['unreliable_channels']} "
        f"(λ불일치 {diag['unreliable_by_lam']} / n_SiN하한 {diag['unreliable_by_n_sin']}) / "
        f"단조 보정 {diag['monotone_fixups']}채널"
    )
