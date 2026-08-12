"""Stage A 캘리브레이션 — train (d_true, R_obs) 서브셋으로 TMM forward 미지수를 피팅한다.

무엇을 학습하나 (CLAUDE.md Level 2 Stage A 스펙):
  - λ 그리드: lam = lam_min + cumsum(softplus(·)) — 채널 순서대로 단조 (방향 자유).
  - SiN(layer 1·3): Cauchy n(λ) = A + B/λ² + C/λ⁴ (λ[μm], k=0 가정) — 학습.
  - SiO₂(layer 2·4): 같은 Cauchy — **문헌값(Malitson 1965 fit)에 freeze (게이지 고정)**.
    delta = 2πnd/λ 가 (n, λ) 공통 스케일에 불변이라 SiO₂를 고정해야 λ가 식별된다.
  - Si 기판: n(λ), k(λ) 곡선 — 채널축 knot 조각별 선형 보간, k ≥ 0 (softplus).
    (si_param: adachi 선택 시 knot 대신 Adachi MDF 13계수 — n·k가 학습된 λ의
    매끈한 물리 함수가 된다. sio2-freeze-adachi 변형 참조.)

초기화 (phase 0): 두께축 주파수 식별 `identify_initial_grid` — λ축 fringe 정렬은
경사하강에 비볼록이라(잘못된 fringe 차수 분지에서 RMSE ~0.12 정체) 그쪽을 우회한다.
전수 격자 데이터의 조건부 평균 E[R|d_j]가 두께축에서 f_j = 2n_j(λ)/λ로 진동하는
성질로 λ 그리드와 n_SiN(λ)을 채널별 닫힌형으로 추정한 뒤, 본 피팅은 그 근방에서
전 파라미터를 공동 미세조정한다.

산출물: runs/stage_a/<run_name>/{model.pt, train.log, metrics.json}
(+ 진행 중 resume.pt — 완료 시 삭제). 판정 게이트 (a) RMSE는 여기서 즉시 보고하고,
(c) 잔차 백색성 진단·플롯은 scripts/diagnose_calibration.py 가 model.pt를 읽어 수행한다.

세션 유실 대비 계약 (CLAUDE.md — train_gpu.py와 동일): best 갱신 즉시 model.pt 저장,
eval 블록마다 resume.pt(+RNG) 저장·미러, 재실행 시 완료 run 스킵 + 진행 run 재개
(무중단 실행과 동일 결과 — 테스트로 검증).

사용:
    python -m src.calibrate --config configs/stage_a/sio2-freeze.yaml
    python -m src.calibrate --config ... --fit-rows 2000 --steps 40 \
        --lam-init 400,800 --run-name smoke   # 스모크 (주파수 식별 생략)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn.functional import mse_loss, softplus

from src.data.dataset import REPO_ROOT, prepare_train_arrays
from src.physics.dispersion import (
    ADACHI_SI_PARAM_NAMES,
    adachi_si_init,
    adachi_si_nk,
    cauchy_n,
    fit_cauchy,
    linear_interp_matrix,
    si3n4_n,
    si_nk,
    sio2_n,
    softplus_inverse,
)
from src.physics.tmm import tmm_reflectance
from src.train import build_lr_scheduler, log_line
from src.train_gpu import RESUME_NAME, _atomic_save, _mirror_copy, resolve_device

RUNS_DIR = REPO_ROOT / "runs"

# 판정 게이트 (a): 재구성 RMSE < 1.2σ ≈ 0.0105 (σ ≈ 0.0087 노이즈 바닥 — CLAUDE.md).
GATE_A_RMSE = 0.0105

# "value = init + scale × raw" 재파라미터화의 scale — λ(수백 nm)와 Cauchy C(~1e-4)처럼
# 스케일이 극단적으로 다른 양을 Adam이 단일 lr·균일 보폭(raw 공간)으로 움직이게 한다.
_LAM_MIN_SCALE = 100.0  # raw 1 ≈ λ_min 100 nm 이동
_DLAM_SCALE = 0.3  # 채널 간격(softplus 전 raw)의 보폭
_SI_N_SCALE = 0.5  # Si n knot 보폭
_SI_K_SCALE = 1.0  # Si k knot(softplus 전 raw) 보폭 — k가 작을 땐 곱셈적 변화
_SIN_KNOT_SCALE = 0.2  # SiN n knot 보폭 (knot 모드일 때)

# Adachi 계수의 raw(softplus 전) 보폭 — 임계점 에너지는 가장 확실한 물성이라 느리게,
# 진폭·broadening은 상대적으로 자유롭게 움직인다.
_ADACHI_SCALE_BY_NAME = {
    "eps_inf": 0.3,
    "D": 0.3,
    "Eg": 0.02,
    "B1": 0.3,
    "B1x": 0.3,
    "Gamma1": 0.1,
    "E1": 0.02,
    "C0": 0.3,
    "gamma0": 0.1,
    "E0p": 0.02,
    "C": 0.3,
    "gamma": 0.1,
    "E2": 0.02,
}


class CalibratedStack(nn.Module):
    """캘리브레이션 대상 forward 모델의 미지수 전부를 담는 모듈.

    구조: 공기(1.0) / SiN / SiO₂ / SiN / SiO₂ / Si 기판, 수직입사 (CLAUDE.md 도메인 계약).
    dtype은 캘리브레이션 계약대로 float64/complex128 고정.

    Args:
        n_channels: 스펙트럼 채널 수 W.
        n_si_knots: Si n·k 곡선의 knot 수 (채널축 균등 배치, 조각별 선형 보간).
        lam_init: 초기 가정 λ 범위 (min, max) [nm] — 실제 그리드는 학습된다.
        descending: True면 채널 0이 λ_max (λ가 채널 순서로 감소).
        si_param: Si 기판 파라미터화 — "knot"(채널축 knot 보간, 기본) 또는
            "adachi"(Adachi MDF ~13계수 — n·k가 학습된 λ의 매끈한 물리 함수가 된다).
    """

    def __init__(
        self,
        n_channels: int = 226,
        n_si_knots: int = 16,
        n_sin_knots: int | None = None,
        lam_init: tuple[float, float] = (400.0, 800.0),
        descending: bool = False,
        lam_grid: np.ndarray | None = None,
        sin_init_samples: tuple[np.ndarray, np.ndarray] | None = None,
        curve_inits: dict[str, np.ndarray] | None = None,
        si_param: str = "knot",
    ) -> None:
        super().__init__()
        curve_inits = curve_inits or {}
        if si_param not in ("knot", "adachi"):
            raise ValueError(f'si_param은 "knot"|"adachi" 중 하나여야 한다 (받은 값: {si_param})')
        if si_param == "adachi" and not {"n_si", "k_si"}.isdisjoint(curve_inits):
            raise ValueError("adachi 모드는 Si curve_inits를 받지 않는다 (계수는 프리핏 JSON에서)")
        if lam_grid is not None:
            # 명시적 초기 그리드 (채널 순서, 예: 주파수 식별 결과) — lam_init/descending 대체.
            lam_ch = np.asarray(lam_grid, dtype=np.float64)
            if lam_ch.shape != (n_channels,):
                raise ValueError(f"lam_grid는 ({n_channels},) 여야 한다: {lam_ch.shape}")
            descending = bool(lam_ch[0] > lam_ch[-1])
            base = lam_ch[::-1].copy() if descending else lam_ch
            if not (base[0] > 0 and np.all(np.diff(base) > 0)):
                raise ValueError("lam_grid는 양수·강단조여야 한다")
            lam_lo, lam_hi = float(base[0]), float(base[-1])
        else:
            lam_lo, lam_hi = float(lam_init[0]), float(lam_init[1])
            if not 0.0 < lam_lo < lam_hi:
                raise ValueError(f"lam_init은 0 < min < max 여야 한다 (받은 값: {lam_init})")
            base = np.linspace(lam_lo, lam_hi, n_channels)
            lam_ch = base[::-1].copy() if descending else base
        if n_channels < 2:
            raise ValueError(f"n_channels는 2 이상이어야 한다 (받은 값: {n_channels})")
        self.n_channels = int(n_channels)
        self.n_si_knots = int(n_si_knots)
        self.n_sin_knots = None if n_sin_knots is None else int(n_sin_knots)
        self.lam_init = (lam_lo, lam_hi)
        self.descending = bool(descending)
        self.si_param = str(si_param)
        f64 = torch.float64
        knot_pos = np.linspace(0.0, n_channels - 1.0, n_si_knots)
        ch_idx = np.arange(n_channels, dtype=np.float64)

        # λ 그리드 — 단조 보장: lam_min + cumsum(softplus(·)). base는 오름차순 초기값.
        self.register_buffer("lam_min_init", softplus_inverse(torch.tensor(base[0], dtype=f64)))
        self.raw_lam_min = nn.Parameter(torch.zeros((), dtype=f64))
        self.register_buffer("dlam_init", softplus_inverse(torch.from_numpy(np.diff(base)).to(f64)))
        self.raw_dlam = nn.Parameter(torch.zeros(n_channels - 1, dtype=f64))

        # SiN n(λ) — 학습. n_sin_knots가 None이면 Cauchy(3계수), 아니면 채널축 knot 곡선
        # (n_sin_knots == n_channels 이면 채널별 자유 — phase 2 미세조정용).
        # Cauchy 초기값: 주파수 식별의 (λ, n) 표본 > Luke 2015 Sellmeier 근사.
        if self.n_sin_knots is None:
            if sin_init_samples is not None:
                sin_init = torch.from_numpy(fit_cauchy(*sin_init_samples))
            else:
                sin_init = torch.from_numpy(fit_cauchy(base, si3n4_n(base)))
            self.register_buffer("sin_init", sin_init)
            self.register_buffer("sin_scale", torch.clamp(0.5 * sin_init.abs(), min=1e-5))
            self.raw_sin = nn.Parameter(torch.zeros(3, dtype=f64))
        else:
            sin_curve = curve_inits.get("n_sin")
            if sin_curve is None:
                sin_curve = si3n4_n(lam_ch)
            sin_knot_pos = np.linspace(0.0, n_channels - 1.0, self.n_sin_knots)
            self.register_buffer("sin_interp", linear_interp_matrix(n_channels, self.n_sin_knots))
            self.register_buffer(
                "sin_init",
                torch.from_numpy(np.interp(sin_knot_pos, ch_idx, np.asarray(sin_curve))),
            )
            self.raw_sin = nn.Parameter(torch.zeros(self.n_sin_knots, dtype=f64))

        # SiO₂ Cauchy — freeze (게이지 고정: n과 λ는 동시 식별 불가 — CLAUDE.md).
        self.register_buffer("sio2_cauchy", torch.from_numpy(fit_cauchy(base, sio2_n(base))))

        # Si 기판 n·k — knot 모드: 채널축 knot 보간 (초기값: curve_inits > 문헌 테이블),
        # adachi 모드: MDF 계수 (초기값: 프리핏 JSON — scripts/fit_adachi_init.py 산출).
        if self.si_param == "adachi":
            self.register_buffer(
                "adachi_init", softplus_inverse(torch.from_numpy(adachi_si_init()))
            )
            self.register_buffer(
                "adachi_scale",
                torch.tensor([_ADACHI_SCALE_BY_NAME[m] for m in ADACHI_SI_PARAM_NAMES], dtype=f64),
            )
            self.raw_adachi = nn.Parameter(torch.zeros(len(ADACHI_SI_PARAM_NAMES), dtype=f64))
        else:
            self.register_buffer("interp", linear_interp_matrix(n_channels, n_si_knots))
            knot_lam = np.interp(knot_pos, ch_idx, lam_ch)
            n_si_lit, k_si_lit = si_nk(knot_lam)
            n_si_curve, k_si_curve = curve_inits.get("n_si"), curve_inits.get("k_si")
            n_si = (
                np.interp(knot_pos, ch_idx, np.asarray(n_si_curve))
                if n_si_curve is not None
                else n_si_lit
            )
            k_si = (
                np.interp(knot_pos, ch_idx, np.asarray(k_si_curve))
                if k_si_curve is not None
                else k_si_lit
            )
            self.register_buffer("si_n_init", torch.from_numpy(n_si))
            self.raw_si_n = nn.Parameter(torch.zeros(n_si_knots, dtype=f64))
            self.register_buffer(
                "si_k_init", softplus_inverse(torch.from_numpy(np.clip(k_si, 1e-4, None)))
            )
            self.raw_si_k = nn.Parameter(torch.zeros(n_si_knots, dtype=f64))

    @property
    def model_cfg(self) -> dict[str, Any]:
        """체크포인트에서 같은 구조를 복원하기 위한 생성 인자 스냅샷."""
        return {
            "n_channels": self.n_channels,
            "n_si_knots": self.n_si_knots,
            "n_sin_knots": self.n_sin_knots,
            "lam_init": list(self.lam_init),
            "descending": self.descending,
            "si_param": self.si_param,
        }

    def lam(self) -> Tensor:
        """학습된 λ 그리드. 반환 (W,) float64 — 채널 순서 (descending이면 감소)."""
        lam_min = softplus(self.lam_min_init + _LAM_MIN_SCALE * self.raw_lam_min)
        steps = softplus(self.dlam_init + _DLAM_SCALE * self.raw_dlam)
        grid = torch.cat([lam_min.reshape(1), lam_min + torch.cumsum(steps, dim=0)])
        return torch.flip(grid, dims=[0]) if self.descending else grid

    def spectra(self) -> tuple[Tensor, Tensor, Tensor]:
        """TMM 입력 물리량을 전부 계산한다.

        Returns:
            (lam (W,) float64, n_layers (4, W) complex128, ns (W,) complex128).
        """
        lam = self.lam()
        if self.n_sin_knots is None:
            n_sin = cauchy_n(lam, self.sin_init + self.sin_scale * self.raw_sin)
        else:
            n_sin = self.sin_interp @ (self.sin_init + _SIN_KNOT_SCALE * self.raw_sin)
        n_sio2 = cauchy_n(lam, self.sio2_cauchy)
        stack_r = torch.stack([n_sin, n_sio2, n_sin, n_sio2])
        n_layers = torch.complex(stack_r, torch.zeros_like(stack_r))  # 층은 k=0 가정
        if self.si_param == "adachi":
            # n·k가 학습된 λ의 함수 — MDF의 ε₂ ≥ 0 구조가 k ≥ 0을 보장한다.
            n_si, k_si = adachi_si_nk(
                lam, softplus(self.adachi_init + self.adachi_scale * self.raw_adachi)
            )
        else:
            n_si = self.interp @ (self.si_n_init + _SI_N_SCALE * self.raw_si_n)
            k_si = softplus(self.interp @ (self.si_k_init + _SI_K_SCALE * self.raw_si_k))
        ns = torch.complex(n_si, -k_si)  # n − i·k (tmm.py 부호 관례, k ≥ 0)
        return lam, n_layers, ns

    def forward(self, d: Tensor) -> Tensor:
        """두께 d: (B, 4) [nm] → 재구성 반사율 R: (B, W) float64."""
        lam, n_layers, ns = self.spectra()
        return tmm_reflectance(d, n_layers, 1.0, ns, lam)


def load_calibrated_stack(path: Path | str) -> tuple[CalibratedStack, dict[str, Any]]:
    """model.pt에서 캘리브레이션 모델을 복원한다. 반환 (model, 체크포인트 dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    mc = ckpt["model_cfg"]
    n_sin_knots = mc.get("n_sin_knots")
    model = CalibratedStack(
        n_channels=int(mc["n_channels"]),
        n_si_knots=int(mc["n_si_knots"]),
        n_sin_knots=None if n_sin_knots is None else int(n_sin_knots),
        lam_init=tuple(mc["lam_init"]),
        descending=bool(mc["descending"]),
        si_param=str(mc.get("si_param", "knot")),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def _fingerprint(cfg: dict[str, Any], model_cfg: dict[str, Any], n_fit: int) -> str:
    """resume 호환성 판별용 설정 지문 — 다른 설정의 resume.pt를 이어받지 않도록."""
    return json.dumps(
        {
            "model": model_cfg,
            "model_yaml": cfg["model"],
            "fit": cfg["fit"],
            "seed": cfg["seed"],
            "n_fit": n_fit,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


@torch.no_grad()
def _eval_rmse(model: CalibratedStack, d: Tensor, r_obs: Tensor, batch_size: int) -> float:
    """R 재구성 RMSE (전 행·전 채널) — 판정 게이트 (a)와 같은 정의."""
    sq_sum = 0.0
    for start in range(0, len(d), batch_size):
        pred = model(d[start : start + batch_size])
        sq_sum += float(((pred - r_obs[start : start + batch_size]) ** 2).sum())
    return float(np.sqrt(sq_sum / r_obs.numel()))


def identify_initial_grid(
    x: np.ndarray, d: np.ndarray, run_dir: Path, *, f_range: tuple[float, float] = (0.003, 0.022)
) -> dict[str, Any]:
    """두께축 주파수 식별 — 채널별 λ와 SiN n(λ) 초기 추정 (닫힌형, 결정론).

    타깃이 10 nm 전수 격자라 층 j로 조건화한 평균 g_c(d) = E[R_c | d_j = d](30점)를
    만들 수 있고, 이 곡선은 두께축에서 기본 주파수 f_j(c) = 2·n_j(λ_c)/λ_c [cycles/nm]
    로 진동한다 (특성행렬 M_j가 δ→δ+π에서 부호만 바뀌므로 R은 δ_j에 π 주기).
    SiO₂(층 2·4)는 게이지 고정으로 n(λ)가 알려져 있어 f₂, f₄에서 λ_c가 채널별로
    바로 풀린다 — λ축 fringe 정렬의 비볼록성을 완전히 우회한다. 층 2·4의 독립 추정
    일치도와 SiN 추정(f₁·λ/2)의 문헌 근접성이 자체 검증이 된다.

    Adam 기반 초기 λ 범위 후보 탐색은 이 방법으로 대체했다 — 후보 전부가
    RMSE ~0.12에서 정체하는 잘못된 fringe 차수 분지에 안착했다 (2026-08-11,
    docs/week_1.md).

    Args:
        x: (N, W) float32 — R_obs. **표본이 클수록 좋다** — 조건부 평균이 전수 격자의
            정확한 주변화에 가까워야 다른 층의 변동이 지워진다 (5만 행 무작위 표본은
            bin당 ~1.7천 행이라 주파수 추정이 눈에 띄게 흔들린다 — 합성 실험 확인).
            main은 holdout 제외 train 전체(~73만 행, bin당 ~2.4만 행)를 넘긴다.
        d: (N, 4) — 두께 [nm] (10 nm 격자).
        run_dir: 진단 수치를 train.log에 기록.
        f_range: 탐색할 주파수 구간 [cycles/nm].

    Returns:
        {"lam_grid": (W,) 채널 순서 λ [nm], "n_sin_samples": ((W,) λ, (W,) n),
         "diagnostics": {...}}
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

    # 주파수 추정: 각 f 후보의 투영행렬을 미리 만들고 (채널·층 공용),
    # 잔차 최소 f를 고른다. 기저 = [1, d, cos, sin, cos2, sin2] (완만한 배경 + 2배음).
    d_c = grid_vals - grid_vals.mean()
    f_grid = np.linspace(f_range[0], f_range[1], 1200)
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

    # λ_c 풀기: 2·n_sio2(λ)/λ = f 는 λ에 단조 감소 → 이분법 (층 2·4 각각).
    def solve_lam(f_arr: np.ndarray) -> np.ndarray:
        lo = np.full_like(f_arr, 100.0)
        hi = np.full_like(f_arr, 2000.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            above = 2.0 * sio2_n(mid) / mid > f_arr
            lo[above] = mid[above]
            hi[~above] = mid[~above]
        return 0.5 * (lo + hi)

    lam2, lam4 = solve_lam(freqs[1]), solve_lam(freqs[3])
    lam_grid = 0.5 * (lam2 + lam4)
    n_sin = 0.5 * (freqs[0] + freqs[2]) * lam_grid / 2.0

    # 신뢰도 마스크: 층 2·4의 독립 λ 추정이 어긋나는 채널은 주파수 추정 실패로 보고
    # (실데이터에서 Si E1 임계점 부근 fringe 대비 저하로 소수 채널이 크게 튐 —
    # 그대로 두면 λ 그리드에 꺾임이 남아 그 채널들의 재구성 오차가 σ의 10배를 넘는다)
    # λ·n_SiN을 이웃 신뢰 채널에서 선형 보간한다. λ는 채널축에서 매끈하다는 물리
    # 가정(분광기 그리드)이 근거다.
    bad = np.abs(lam2 - lam4) > 5.0
    n_bad = int(bad.sum())
    if 0 < n_bad <= n_ch - 2:
        ch = np.arange(n_ch, dtype=np.float64)
        lam_grid[bad] = np.interp(ch[bad], ch[~bad], lam_grid[~bad])
        n_sin[bad] = np.interp(ch[bad], ch[~bad], n_sin[~bad])

    # 단조 강제: 추정 오차로 국소 요철이 있으면 등화(iso) 대신 누적 최솟값/최댓값으로
    # 살짝 보정한다 (본 피팅의 단조 파라미터화가 요구하는 초기값 조건).
    direction = -1.0 if lam_grid[0] > lam_grid[-1] else 1.0
    mono = np.minimum.accumulate(lam_grid) if direction < 0 else np.maximum.accumulate(lam_grid)
    fix = mono != lam_grid
    if np.any(np.diff(mono) * direction <= 0):
        mono = mono + direction * np.arange(n_ch) * 1e-6  # 동률 제거 (강단조)
    lam_grid = mono

    diagnostics = {
        "lam24_dev_median": float(np.median(np.abs(lam2 - lam4))),
        "lam24_dev_max": float(np.abs(lam2 - lam4).max()),
        "lam_range": [float(lam_grid.min()), float(lam_grid.max())],
        "descending": bool(lam_grid[0] > lam_grid[-1]),
        "n_sin_range": [float(n_sin.min()), float(n_sin.max())],
        "n_sin_vs_luke_reldev_median": float(
            np.median(np.abs(n_sin - si3n4_n(lam_grid)) / si3n4_n(lam_grid))
        ),
        "monotone_fixups": int(fix.sum()),
        "unreliable_channels": n_bad,
    }
    log_line(
        run_dir,
        f"[freq-id] λ {diagnostics['lam_range'][0]:.1f}–{diagnostics['lam_range'][1]:.1f} nm "
        f"{'내림' if diagnostics['descending'] else '오름'}차순 / 층2·4 λ 편차 중앙값 "
        f"{diagnostics['lam24_dev_median']:.2f} nm (최대 {diagnostics['lam24_dev_max']:.2f}) / "
        f"n_SiN {diagnostics['n_sin_range'][0]:.3f}–{diagnostics['n_sin_range'][1]:.3f} "
        f"(Luke 대비 중앙 상대편차 {diagnostics['n_sin_vs_luke_reldev_median']:.3%}) / "
        f"불신 채널 보간 {n_bad} / 단조 보정 {diagnostics['monotone_fixups']}채널",
    )
    return {"lam_grid": lam_grid, "n_sin_samples": (lam_grid, n_sin), "diagnostics": diagnostics}


def fit_calibration(
    x_fit: np.ndarray,
    d_fit: np.ndarray,
    x_val: np.ndarray,
    d_val: np.ndarray,
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    lam_init: tuple[float, float] = (400.0, 800.0),
    descending: bool = False,
    lam_grid: np.ndarray | None = None,
    sin_init_samples: tuple[np.ndarray, np.ndarray] | None = None,
    curve_inits: dict[str, np.ndarray] | None = None,
    init_state: dict[str, Tensor] | None = None,
    device: torch.device | None = None,
    mirror_dir: Path | None = None,
    resume: bool = True,
    _abort_after_eval: int | None = None,
) -> dict[str, Any]:
    """본 피팅 — CalibratedStack을 (d_true, R_obs)에 맞춘다.

    옵티마이저 (fit.optimizer): "adam"(기본) = 미니배치 Adam,
    "lbfgs" = 전배치 L-BFGS(strong Wolfe 라인서치, 청크 누적 closure).
    L-BFGS는 라인서치·곡률 이력이 결정론적 손실을 전제하므로 미니배치 없이
    fit 표본 전체로 평가한다 — Adam 수렴 해의 폴리시(warm start) 용도.

    세션 유실 대비: best(val RMSE) 갱신 즉시 model.pt 저장, eval 블록마다
    resume.pt(+RNG·배치 generator) 저장·미러. 재개 시 무중단 실행과 동일 결과.

    Args:
        x_fit: (N, W) float32 — 피팅용 R_obs. d_fit: (N, 4) 두께 [nm].
        x_val / d_val: best 선택·게이트 (a) RMSE용 분리 표본.
        cfg: 전체 config (fit/model/seed 사용).
        run_dir: 산출물 디렉토리 (train.log / model.pt / resume.pt).
        lam_init: 초기 λ 범위 (lam_grid가 없을 때 균등 그리드로 사용).
        descending: λ 채널 방향 (lam_grid가 없을 때).
        lam_grid: 채널별 초기 λ (주파수 식별 결과) — 주면 lam_init/descending 대체.
        sin_init_samples: SiN Cauchy 초기값용 (λ, n) 표본 — 주파수 식별 결과.
        init_state: 같은 구조의 이전 run state_dict — 정밀 워름스타트 (파라미터·버퍼
            전부 복사, curve_inits보다 정확). 구조가 다르면 load_state_dict가 실패한다.
        mirror_dir: 지정 시 산출물을 매 eval 블록 복사 (Colab Drive 백업).
        resume: False면 resume.pt를 무시하고 처음부터.
        _abort_after_eval: 테스트 전용 — N번째 eval 블록 직후 RuntimeError로 중단.

    Returns:
        {"ckpt_path", "best_step", "best_val_rmse", "steps", "wall_sec",
         "gate_a": {"rmse", "threshold", "pass"}}
    """
    fit_cfg = cfg["fit"]
    steps_total = int(fit_cfg["steps"])
    batch_size = int(fit_cfg["batch_size"])
    eval_every = int(fit_cfg["eval_every"])
    eval_batch = int(fit_cfg.get("eval_batch", 8192))
    seed = int(cfg["seed"])
    if steps_total < 1 or eval_every < 1:
        raise ValueError(f"steps({steps_total})·eval_every({eval_every})는 1 이상이어야 한다")
    device = device or torch.device("cpu")

    n_sin_knots = cfg["model"].get("n_sin_knots")
    model = CalibratedStack(
        n_channels=x_fit.shape[1],
        n_si_knots=int(cfg["model"]["n_si_knots"]),
        n_sin_knots=None if n_sin_knots is None else int(n_sin_knots),
        lam_init=lam_init,
        descending=descending,
        lam_grid=lam_grid,
        sin_init_samples=sin_init_samples,
        curve_inits=curve_inits,
        si_param=str(cfg["model"].get("si_param", "knot")),
    )
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(device)
    opt_name = str(fit_cfg.get("optimizer", "adam"))
    if opt_name == "lbfgs":
        if str(fit_cfg.get("lr_schedule", "cosine")) != "none":
            raise ValueError(
                'optimizer=lbfgs는 lr_schedule: "none"과 함께 써야 한다 (라인서치가 보폭을 정한다)'
            )
        optimizer: torch.optim.Optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=float(fit_cfg["lr"]),
            max_iter=1,  # 외부 루프가 스텝을 관리한다 (eval·체크포인트 주기와 맞물리게)
            # max_eval 기본값은 max_iter*5//4 — max_iter=1이면 1이 되어 초기 평가가
            # 예산을 다 쓰고 라인서치 예산(max_ls)이 0이 된다 → t=0 반환, 파라미터가
            # 전혀 안 움직이는 조용한 정체 (실측 재현·수정 2026-08-12). 명시 필수.
            max_eval=32,
            history_size=int(fit_cfg.get("history_size", 20)),
            line_search_fn="strong_wolfe",
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(fit_cfg["lr"]))
    else:
        raise ValueError(f'fit.optimizer는 "adam"|"lbfgs" 중 하나여야 한다 (받은 값: {opt_name})')
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=str(fit_cfg.get("lr_schedule", "cosine")),
        warmup_steps=int(fit_cfg.get("warmup_steps", 0)),
        total_steps=steps_total,
    )
    # 배치 표집 전용 generator — 전역 RNG와 분리해 resume 복원을 단순·정확하게 한다.
    gen = torch.Generator().manual_seed(seed)

    x_fit_t = torch.from_numpy(x_fit).to(device=device, dtype=torch.float64)
    d_fit_t = torch.from_numpy(d_fit).to(device=device, dtype=torch.float64)
    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float64)
    d_val_t = torch.from_numpy(d_val).to(device=device, dtype=torch.float64)
    n = len(x_fit_t)

    fingerprint = _fingerprint(cfg, model.model_cfg, n)
    resume_path = run_dir / RESUME_NAME
    best_rmse = float("inf")
    best_step = -1
    done_steps = 0
    wall_prev = 0.0

    def save_best(step: int, rmse: float) -> None:
        """best 갱신 즉시 저장 — 세션이 언제 죽어도 최신 best가 남는다."""
        _atomic_save(
            {
                "model_cfg": model.model_cfg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": step,
                "val_rmse": rmse,
                "fingerprint": fingerprint,
            },
            run_dir / "model.pt",
        )
        _mirror_copy(run_dir, mirror_dir, ("model.pt",))

    if resume:
        if not resume_path.exists() and mirror_dir is not None:
            _mirror_copy(mirror_dir, run_dir, (RESUME_NAME,))
            if resume_path.exists():
                log_line(run_dir, "[calib] 미러에서 resume.pt 복원")
        if resume_path.exists():
            # resume.pt는 이 모듈이 만든 자기 산출물 — RNG 상태 등 비텐서 객체 포함
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise ValueError(
                    "resume.pt의 설정이 현재 config와 다르다 — "
                    "run_name을 바꾸거나 resume.pt를 지우고 다시 실행할 것"
                )
            model.load_state_dict(state["model"])
            model.to(device)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            gen.set_state(state["generator"])
            torch.set_rng_state(state["torch_rng"].cpu())
            np.random.set_state(state["numpy_rng"])  # noqa: NPY002 (seed.py와 동일 이유)
            random.setstate(state["py_rng"])
            best_rmse = state["best_val_rmse"]
            best_step = state["best_step"]
            done_steps = state["step"]
            wall_prev = state["wall_sec"]
            log_line(
                run_dir,
                f"[calib] resume: step {done_steps}까지 완료 상태에서 재개"
                f" (best {best_rmse:.5f} @ step {best_step})",
            )

    n_pix = float(n) * float(x_fit_t.shape[1])

    def full_batch_closure() -> Tensor:
        """전배치 MSE — L-BFGS 라인서치용 결정론 closure (청크 누적으로 메모리 상한 유지)."""
        optimizer.zero_grad()
        total = torch.zeros((), dtype=torch.float64, device=device)
        for s in range(0, n, batch_size):
            chunk = (
                mse_loss(
                    model(d_fit_t[s : s + batch_size]),
                    x_fit_t[s : s + batch_size],
                    reduction="sum",
                )
                / n_pix
            )
            chunk.backward()
            total += chunk.detach()
        return total

    t_start = time.perf_counter()
    loss_sum = 0.0
    loss_cnt = 0
    eval_block = 0
    t_block = time.perf_counter()
    for step in range(done_steps + 1, steps_total + 1):
        if opt_name == "lbfgs":
            loss = optimizer.step(full_batch_closure)
        else:
            idx = torch.randint(0, n, (batch_size,), generator=gen)
            loss = mse_loss(model(d_fit_t[idx]), x_fit_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item()
        loss_cnt += 1

        if step % eval_every == 0 or step == steps_total:
            val_rmse = _eval_rmse(model, d_val_t, x_val_t, eval_batch)
            marker = ""
            if val_rmse < best_rmse:
                best_rmse, best_step = val_rmse, step
                save_best(step, val_rmse)
                marker = " *"
            log_line(
                run_dir,
                f"[calib] step {step:5d}/{steps_total}  train_rmse"
                f" {np.sqrt(loss_sum / max(loss_cnt, 1)):.5f}  val_rmse {val_rmse:.5f}"
                f"  lr {optimizer.param_groups[0]['lr']:.2e}"
                f"  {time.perf_counter() - t_block:.1f}s{marker}",
            )
            loss_sum, loss_cnt = 0.0, 0
            t_block = time.perf_counter()
            _atomic_save(
                {
                    "fingerprint": fingerprint,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),  # noqa: NPY002
                    "py_rng": random.getstate(),
                    "best_val_rmse": best_rmse,
                    "best_step": best_step,
                    "wall_sec": wall_prev + time.perf_counter() - t_start,
                },
                resume_path,
            )
            _mirror_copy(run_dir, mirror_dir, ("train.log", RESUME_NAME))
            eval_block += 1
            if _abort_after_eval is not None and eval_block >= _abort_after_eval:
                raise RuntimeError("세션 중단(테스트 전용)")

    resume_path.unlink(missing_ok=True)
    wall_sec = wall_prev + time.perf_counter() - t_start
    ckpt_path = run_dir / "model.pt"
    if ckpt_path.is_relative_to(REPO_ROOT):
        ckpt_path = ckpt_path.relative_to(REPO_ROOT)
    gate_a = {"rmse": best_rmse, "threshold": GATE_A_RMSE, "pass": best_rmse < GATE_A_RMSE}
    log_line(
        run_dir,
        f"\n[calib] best val_rmse {best_rmse:.5f} @ step {best_step}"
        f" — 게이트 (a) RMSE < {GATE_A_RMSE}: {'통과' if gate_a['pass'] else '실패'}"
        f" (잔차 백색성 (c)는 scripts/diagnose_calibration.py로 별도 판정)",
    )
    return {
        "ckpt_path": str(ckpt_path),
        "best_step": best_step,
        "best_val_rmse": best_rmse,
        "steps": steps_total,
        "wall_sec": wall_sec,
        "gate_a": gate_a,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A 캘리브레이션")
    parser.add_argument("--config", required=True, help="configs/stage_a/*.yaml 경로")
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--fit-rows", type=int, default=None, help="피팅 행 수 (config 덮어씀)")
    parser.add_argument("--diag-rows", type=int, default=None, help="진단 행 수 (config 덮어씀)")
    parser.add_argument("--steps", type=int, default=None, help="피팅 스텝 수 (config 덮어씀)")
    parser.add_argument(
        "--lam-init", default=None, help='"400,800" — 초기 λ 범위 직접 지정 (후보 탐색 생략)'
    )
    parser.add_argument(
        "--descending", action="store_true", help="--lam-init와 함께: λ를 채널 내림차순으로"
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (기본: cuda 있으면 cuda)")
    parser.add_argument("--mirror-dir", default=None, help="산출물 미러 디렉토리 (Drive 백업)")
    parser.add_argument("--no-resume", action="store_true", help="resume.pt 무시, 처음부터")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.fit_rows is not None:
        cfg["data"]["fit_rows"] = args.fit_rows
    if args.diag_rows is not None:
        cfg["data"]["diag_rows"] = args.diag_rows
    if args.steps is not None:
        cfg["fit"]["steps"] = args.steps
    resume = not args.no_resume

    experiment = cfg.get("experiment")
    if not experiment:
        raise ValueError('config에 "experiment" 키가 필요하다 — runs/<experiment>/<run_name> 구조')
    run_dir = RUNS_DIR / str(experiment) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    prev: dict[str, Any] | None = None
    if metrics_path.exists():
        prev = json.loads(metrics_path.read_text())
    if resume and prev is not None and "result" in prev:
        log_line(run_dir, f"[calib] {experiment}/{cfg['run_name']}: 완료된 run — 스킵")
        return

    seed = int(cfg["seed"])
    device = resolve_device(args.device)
    data_cfg = cfg["data"]
    n_fit = int(data_cfg["fit_rows"])
    n_diag = int(data_cfg["diag_rows"])
    # 프로젝트 공통 holdout(학습 평가용)을 뺀 train 쪽에서만 표집한다 — Stage B 평가와
    # 물리 파라미터 피팅 사이의 정보 누설을 원천 차단 (파라미터 수백 개라 미미하지만).
    x, y, train_idx, _ = prepare_train_arrays(
        val_frac=float(data_cfg.get("val_frac", 0.1)), seed=seed
    )
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=n_fit + n_diag, replace=False)
    x_fit, d_fit = x[pick[:n_fit]], y[pick[:n_fit]]
    x_diag, d_diag = x[pick[n_fit:]], y[pick[n_fit:]]
    log_line(
        run_dir,
        f"run {experiment}/{cfg['run_name']}: fit {n_fit:,} + diag {n_diag:,} 행"
        f" (holdout 제외 train에서 표집) / device={device.type}",
    )

    # 초기화 — 이전 run 곡선(phase 2 미세조정) > CLI 지정 λ 범위 > 이전 실행의
    # 주파수 식별 결과(재개) > 주파수 식별.
    init_kwargs: dict[str, Any]
    init_record: dict[str, Any]
    init_from = cfg["model"].get("init_from_run")
    if init_from is not None:
        src_run = Path(init_from)
        if not src_run.is_absolute():
            src_run = REPO_ROOT / src_run
        prior, prior_ckpt = load_calibrated_stack(src_run / "model.pt")
        with torch.no_grad():
            lam_p, n_layers_p, ns_p = prior.spectra()
        n_sin_knots_cfg = cfg["model"].get("n_sin_knots")
        same_structure = (
            prior.model_cfg["si_param"] == str(cfg["model"].get("si_param", "knot"))
            and prior.model_cfg["n_sin_knots"]
            == (None if n_sin_knots_cfg is None else int(n_sin_knots_cfg))
            and prior.model_cfg["n_si_knots"] == int(cfg["model"]["n_si_knots"])
        )
        if same_structure:
            # 같은 구조 — state_dict 정밀 워름스타트 (파라미터·버퍼 전부, 예: L-BFGS 폴리시).
            init_kwargs = {"lam_grid": lam_p.numpy(), "init_state": prior.state_dict()}
        else:
            # 구조가 다르면 물리량(곡선)으로 이식 (예: phase 2 — Cauchy → 채널별 knot).
            init_kwargs = {
                "lam_grid": lam_p.numpy(),
                "curve_inits": {
                    "n_sin": n_layers_p[0].real.numpy(),
                    "n_si": ns_p.real.numpy(),
                    "k_si": (-ns_p.imag).numpy(),
                },
            }
        init_record = {
            "mode": "from-run-state" if same_structure else "from-run",
            "run": str(init_from),
            "src_step": int(prior_ckpt["step"]),
            "src_val_rmse": float(prior_ckpt["val_rmse"]),
        }
        log_line(
            run_dir,
            f"[init] {init_from} (step {prior_ckpt['step']},"
            f" val_rmse {prior_ckpt['val_rmse']:.5f}) "
            f"{'state_dict' if same_structure else '곡선'}에서 초기화",
        )
    elif args.lam_init is not None:
        lo, hi = (float(v) for v in args.lam_init.split(","))
        init_kwargs = {"lam_init": (lo, hi), "descending": bool(args.descending)}
        init_record = {"mode": "cli", "lam_init": [lo, hi], "descending": bool(args.descending)}
        log_line(run_dir, f"[freq-id] CLI 지정 λ {[lo, hi]} — 주파수 식별 생략")
    elif resume and prev is not None and prev.get("init", {}).get("mode") == "freq-id":
        init_record = prev["init"]
        init_kwargs = {
            "lam_grid": np.asarray(init_record["lam_grid"], dtype=np.float64),
            "sin_init_samples": (
                np.asarray(init_record["lam_grid"], dtype=np.float64),
                np.asarray(init_record["n_sin"], dtype=np.float64),
            ),
        }
        log_line(run_dir, "[freq-id] 이전 실행의 식별 결과 재사용")
    else:
        # 식별은 조건부 평균의 정확도가 생명이라 fit 서브셋이 아니라 holdout 제외
        # train 전체를 쓴다 (identify_initial_grid docstring 참조).
        ident = identify_initial_grid(x[train_idx], y[train_idx], run_dir)
        init_kwargs = {
            "lam_grid": ident["lam_grid"],
            "sin_init_samples": ident["n_sin_samples"],
        }
        init_record = {
            "mode": "freq-id",
            "lam_grid": [float(v) for v in ident["lam_grid"]],
            "n_sin": [float(v) for v in ident["n_sin_samples"][1]],
            "diagnostics": ident["diagnostics"],
        }
    del x, y

    metrics: dict[str, Any] = {
        "experiment": experiment,
        "run_name": cfg["run_name"],
        "seed": seed,
        "rows": {"fit": n_fit, "diag": n_diag},
        "config": cfg,
        "init": init_record,
    }
    # 시작 시점 설정·식별 스냅샷 (중단돼도 남고, 재개 시 init을 재사용한다).
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    result = fit_calibration(
        x_fit,
        d_fit,
        x_diag,
        d_diag,
        cfg,
        run_dir,
        device=device,
        mirror_dir=None if args.mirror_dir is None else Path(args.mirror_dir),
        resume=resume,
        **init_kwargs,
    )
    metrics["result"] = result
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    _mirror_copy(
        run_dir,
        None if args.mirror_dir is None else Path(args.mirror_dir),
        ("metrics.json", "train.log", "model.pt"),
    )
    log_line(run_dir, f"\n산출물: {run_dir}/ (metrics.json, train.log, model.pt)")


if __name__ == "__main__":
    main()
