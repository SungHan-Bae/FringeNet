"""Stage A 캘리브레이션 — 비식별 파장축과 미지 물성을 (d_true, R_obs)에서 역추정한다.

**설계 원칙: 물리 법칙을 자유도 개수로 강제한다.** 물성은 λ의 매끈한 함수이고 파장축은
분광기 격자 분산의 결과다. 이 제약을 파라미터화가 강제하지 않으면(채널별 자유 곡선)
손잡이가 모델 오차를 흡수해 RMSE는 내려가지만 나온 곡선은 물성이 아니다. 그래서 전체
자유 파라미터가 **1~7개**다:

| 대상 | 모델 | 자유 |
|---|---|---|
| λ(c) | 1/λ = ν₀(1 + r₁u + r₂u²), u = c/(W−1) | 0 또는 3 |
| SiO₂ | Malitson 1965 Sellmeier **동결** | 0 (게이지) — `sio2_scale`은 게이지 검정 전용 |
| SiN | Luke 2015 Sellmeier 형태, k=0 | 1~2 (B₁, C₁) |
| Si 기판 | Schinke 2015 실측표 + 에너지축 3차 스플라인 | 0~2 (ΔE, k 스케일) |

**게이지 고정은 원리적 요구다**: δ = 2πnd/λ 가 (n, λ) 공통 스케일에 불변이라 둘을 동시에
자유로 두면 해가 정해지지 않는다. SiO₂를 문헌값에 못박는 것이 그 선언이고, λ의 절대
스케일은 이 가정에 의존한다. SiN은 동결하지 않는다 — 게이지-불변 관측량 n_SiN/n_SiO₂ 가
문헌 대비 −2.15% 균일 편차라 동결은 이미 측정된 2%를 모델에 주입하는 셈이다.

판정 게이트와 결론은 reports/stage_a.md · scripts/diagnose_calibration.py.

사용법:
    python -m src.calibrate --config configs/stage_a/joint-lam3-sin2-si2-schinke.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares
from torch import Tensor, nn

from src.data.dataset import REPO_ROOT, prepare_train_arrays
from src.physics.dispersion import (
    HC_EV_NM,
    SI3N4_LUKE_SELLMEIER,
    SIO2_MALITSON_SELLMEIER,
    CoarseTableNK,
    TabulatedNK,
)
from src.physics.freq_id import describe_identification, identify_wavelength_grid
from src.physics.tmm import tmm_reflectance
from src.train import log_line
from src.train_gpu import _atomic_save

RUNS_DIR = REPO_ROOT / "runs"

# 노이즈 바닥과 하드 상한 — scripts/measure_noise.py 산출. 상한은 1.8억 관측 중
# R_obs < −0.0152 가 0건이라는 사실에서 온다 (가우시안이면 5σ = −0.043까지 나와야 한다).
NOISE_SIGMA = 0.008658
NOISE_BOUND = 0.0152
GATE_A_RMSE = 1.2 * NOISE_SIGMA  # 0.010390

# 고정 분할 — 진단 표본이 run 사이에 **비트 단위로 동일**해야 ablation 비교가 성립한다.
# diag = pick[50000:70000] 이므로 fit_rows를 줄여도 진단 표본은 바뀌지 않는다.
_SPLIT_SEED = 42
_SPLIT_FIT_ROWS = 50_000
_SPLIT_DIAG_ROWS = 20_000

# θ = 1 에 해당하는 물리량 변화 — 최소제곱이 균일 보폭으로 움직이게 하는 무차원화.
# lam_nu0만 상대 스케일(초기값의 0.1% ≈ λ 0.5 nm), 나머지는 절대.
_THETA_STEP_REL = {"lam_nu0": 1e-3}
_THETA_STEP_ABS = {
    "lam_r1": 1e-3,
    "lam_r2": 1e-3,
    "sin_b1": 0.03,  # Luke B₁ = 3.0249 의 약 1%
    "sin_c1": 1.8e-4,  # Luke C₁ = 0.018317 μm² 의 약 1%
    "si_de": 5e-3,  # Si 에너지축 시프트 [eV]
    "si_klog": 0.02,  # Si k 로그 스케일 (약 2%)
    "sio2_scale": 1e-3,  # SiO₂ n 배율 (0.1%) — 게이지 검정 전용, 기본은 동결
}
# Si 실측표의 기본값 = 채택 표. 기본값을 기각된 표로 두면 si_source를 빠뜨린 config가
# 조용히 그쪽으로 돈다 (Aspnes/Green도 `literature/`에 있고 si_source로 고른다).
DEFAULT_SI_SOURCE = "Si_nk_Schinke.yml"
PARAM_NAMES = (
    "lam_nu0",
    "lam_r1",
    "lam_r2",
    "sin_b1",
    "sin_c1",
    "si_de",
    "si_klog",
    "sio2_scale",
)


def fit_lam_coefficients(lam_grid: np.ndarray, *, trim_sigma: float = 4.0) -> tuple[float, ...]:
    """채널별 λ 추정에 1/λ = ν₀(1 + r₁u + r₂u²)를 강건 적합한다 (u = c/(W−1)).

    분광기 격자 분산은 채널 인덱스의 매끈·단조 함수다. 1단계의 채널별 추정에는 ~0.44 nm
    흔들림(주파수 후보 격자 양자화에서 오는 결정론적 편향)이 있고, 그대로 고정하면 그것만
    으로 R 오차 rms 0.0052 — 최종 모델에 남은 계통오차 전체(0.0041)보다 크다.

    Args:
        lam_grid: (W,) 채널 순서 λ [nm] (1단계 주파수 식별 결과).
        trim_sigma: 잔차 MAD 기준 이탈 채널 배제 임계 (주파수 식별 실패 방어).

    Returns:
        (ν₀ [1/nm], r₁, r₂).
    """
    lam = np.asarray(lam_grid, dtype=np.float64)
    n_ch = len(lam)
    u = np.arange(n_ch, dtype=np.float64) / (n_ch - 1.0)
    nu = 1.0 / lam
    keep = np.ones(n_ch, dtype=bool)
    poly = np.polyfit(u, nu, 2)
    for _ in range(5):
        poly = np.polyfit(u[keep], nu[keep], 2)
        resid = nu - np.polyval(poly, u)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep])))
        new = np.abs(resid - np.median(resid[keep])) < trim_sigma * 1.4826 * max(mad, 1e-30)
        if bool((new == keep).all()):
            break
        keep = new
    nu0 = float(np.polyval(poly, 0.0))
    return nu0, float(poly[1] / nu0), float(poly[0] / nu0)


class PhysicalStack(nn.Module):
    """물리 제약 캘리브레이션 모델 — 자유 파라미터를 이름으로 골라 켠다.

    Args:
        n_channels: 채널 수 W.
        lam_coeffs: (ν₀, r₁, r₂) 초기값 — `fit_lam_coefficients` 산출.
        free: 자유로 둘 파라미터 이름들 (PARAM_NAMES 부분집합). 비면 전부 동결.
        si_source: Si n·k 출처 — `literature/`의 tabulated nk 파일명, 또는 ablation
            대조군 `"coarse"` (거친 19점 표 + λ축 선형 보간).
    """

    def __init__(
        self,
        *,
        n_channels: int,
        lam_coeffs: tuple[float, float, float],
        free: tuple[str, ...] = (),
        si_source: str = DEFAULT_SI_SOURCE,
    ) -> None:
        super().__init__()
        unknown = set(free) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"모르는 자유 파라미터: {sorted(unknown)} (가능: {PARAM_NAMES})")
        f64 = torch.float64
        self.n_channels = int(n_channels)
        self.lam_coeffs = tuple(float(v) for v in lam_coeffs)
        self.free = tuple(free)
        self.si_source = si_source
        self._free_index = {name: i for i, name in enumerate(self.free)}

        inits = {
            "lam_nu0": self.lam_coeffs[0],
            "lam_r1": self.lam_coeffs[1],
            "lam_r2": self.lam_coeffs[2],
            "sin_b1": SI3N4_LUKE_SELLMEIER[0][0],
            "sin_c1": SI3N4_LUKE_SELLMEIER[1][0],
            "si_de": 0.0,
            "si_klog": 0.0,
            # 기본 1.0 동결. 자유로 두는 것은 **게이지 검정 전용** — Si를 에너지축에
            # 동결하면 임계점이 절대 앵커가 되어 적합값이 1로 돌아오는지가 λ 절대
            # 스케일의 독립 검증이 된다.
            "sio2_scale": 1.0,
        }
        for name, value in inits.items():
            self.register_buffer(f"init_{name}", torch.tensor(value, dtype=f64))
        steps = [
            _THETA_STEP_REL[n] * abs(inits[n]) if n in _THETA_STEP_REL else _THETA_STEP_ABS[n]
            for n in self.free
        ]
        self.register_buffer("theta_step", torch.tensor(steps, dtype=f64))
        self.theta = nn.Parameter(torch.zeros(len(self.free), dtype=f64))

        self.register_buffer(
            "u", torch.linspace(0.0, 1.0, self.n_channels, dtype=f64)
        )  # 채널 정규좌표
        self.register_buffer("sio2_b", torch.tensor(SIO2_MALITSON_SELLMEIER[0], dtype=f64))
        self.register_buffer("sio2_c", torch.tensor(SIO2_MALITSON_SELLMEIER[1], dtype=f64))
        # SiN 적외 항(√C₂ = 1.24 mm)은 대역에서 λ²에 비례하는 작은 보정이라 동결한다.
        self.register_buffer("sin_b2", torch.tensor(SI3N4_LUKE_SELLMEIER[0][1], dtype=f64))
        self.register_buffer("sin_c2", torch.tensor(SI3N4_LUKE_SELLMEIER[1][1], dtype=f64))
        self.si_nk = CoarseTableNK() if si_source == "coarse" else TabulatedNK(si_source)

    @property
    def model_cfg(self) -> dict[str, Any]:
        """체크포인트에서 같은 구조를 복원하기 위한 생성 인자 스냅샷."""
        return {
            "n_channels": self.n_channels,
            "lam_coeffs": list(self.lam_coeffs),
            "free": list(self.free),
            "si_source": self.si_source,
        }

    def _value(self, name: str) -> Tensor:
        """파라미터의 현재 물리값 (0-dim). 자유가 아니면 초기값 그대로."""
        init = getattr(self, f"init_{name}")
        idx = self._free_index.get(name)
        return init if idx is None else init + self.theta_step[idx] * self.theta[idx]

    def physical_values(self) -> dict[str, float]:
        """이름 → 현재 물리값 (보고용)."""
        return {name: float(self._value(name).detach()) for name in PARAM_NAMES}

    def lam(self) -> Tensor:
        """채널 순서 λ 그리드. 반환 (W,) float64 — 1/λ이 채널의 2차 다항식."""
        nu = self._value("lam_nu0") * (
            1.0 + self._value("lam_r1") * self.u + self._value("lam_r2") * self.u**2
        )
        return 1.0 / nu

    def spectra(self) -> tuple[Tensor, Tensor, Tensor]:
        """TMM 입력 물리량. 반환 (lam (W,), n_layers (4, W) complex, ns (W,) complex)."""
        lam = self.lam()
        n_sio2 = _sellmeier(lam, self.sio2_b, self.sio2_c) * self._value("sio2_scale")
        n_sin = _sellmeier(
            lam,
            torch.stack([self._value("sin_b1"), self.sin_b2]),
            torch.stack([self._value("sin_c1"), self.sin_c2]),
        )
        # Si는 에너지축에서 ΔE만큼 시프트해 평가한다 (온도·응력·조성의 임계점 이동).
        lam_si = HC_EV_NM / (HC_EV_NM / lam + self._value("si_de"))
        n_si, k_si = self.si_nk(lam_si)
        k_si = k_si * torch.exp(self._value("si_klog"))
        stack = torch.stack([n_sin, n_sio2, n_sin, n_sio2])  # 공기/SiN/SiO₂/SiN/SiO₂/Si
        return (
            lam,
            torch.complex(stack, torch.zeros_like(stack)),  # 층은 k = 0 가정
            torch.complex(n_si, -k_si),  # n − i·k (tmm.py 부호 관례)
        )

    def forward(self, d: Tensor) -> Tensor:
        """두께 d: (B, 4) [nm] → 재구성 반사율 R: (B, W) float64."""
        lam, n_layers, ns = self.spectra()
        return tmm_reflectance(d, n_layers, 1.0, ns, lam)


def _sellmeier(lam_nm: Tensor, b: Tensor, c_um2: Tensor) -> Tensor:
    """모듈 내부용 Sellmeier — dispersion.sellmeier_n_t와 동일 (import 축약)."""
    lam2 = (lam_nm * 1e-3).unsqueeze(-1) ** 2
    return torch.sqrt(1.0 + (b * lam2 / (lam2 - c_um2)).sum(dim=-1))


def load_physical_stack(path: Path | str) -> tuple[PhysicalStack, dict[str, Any]]:
    """model.pt에서 모델을 복원한다. 반환 (model, 체크포인트 dict).

    **구버전 호환**: 체크포인트에 없는 `init_*` 버퍼는 현재 기본값으로 채운다. 자유였던
    파라미터는 저장돼 있으니 채워지는 것은 그 run이 건드리지 않은 손잡이뿐이고, 복원되는
    물리 모델은 저장 당시와 동일하다. 이게 없으면 파라미터를 하나 추가하는 순간 기존 run
    전부가 로드 불가가 된다.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    mc = ckpt["model_cfg"]
    model = PhysicalStack(
        n_channels=int(mc["n_channels"]),
        lam_coeffs=tuple(mc["lam_coeffs"]),
        free=tuple(mc["free"]),
        si_source=str(mc["si_source"]),
    )
    state = dict(ckpt["state_dict"])
    current = model.state_dict()
    missing = set(current) - set(state)
    unexpected = missing - {f"init_{name}" for name in PARAM_NAMES}
    if unexpected:
        raise ValueError(f"체크포인트에 없는 키가 init_* 버퍼가 아니다: {sorted(unexpected)}")
    for key in missing:
        state[key] = current[key]
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_split(fit_rows: int) -> dict[str, np.ndarray]:
    """run 사이에 동일한 진단 표본을 쓰는 고정 분할. 반환 x_fit/d_fit/x_diag/d_diag."""
    if fit_rows > _SPLIT_FIT_ROWS:
        raise ValueError(f"fit_rows는 {_SPLIT_FIT_ROWS} 이하여야 한다 (진단 표본과 겹치지 않게)")
    x, y, train_idx, _ = prepare_train_arrays(val_frac=0.1, seed=_SPLIT_SEED)
    rng = np.random.default_rng(_SPLIT_SEED)
    pick = rng.choice(train_idx, size=_SPLIT_FIT_ROWS + _SPLIT_DIAG_ROWS, replace=False)
    fit_idx, diag_idx = pick[:fit_rows], pick[_SPLIT_FIT_ROWS:]
    return {
        "x_fit": x[fit_idx],
        "d_fit": y[fit_idx],
        "x_diag": x[diag_idx],
        "d_diag": y[diag_idx],
    }


def identify_lam_coefficients(run_dir: Path) -> tuple[tuple[float, ...], dict[str, Any]]:
    """두께축 주파수 식별을 수행해 λ 3계수를 얻는다.

    조건부 평균 E[R_c | d_j] 의 정확도가 생명이라 fit 서브셋이 아니라 **holdout 제외
    train 전체**(~73만 행, bin당 ~2.4만 행)를 쓴다. 닫힌형이라 같은 데이터면 항상 같은
    결과가 나온다 (run마다 재계산해도 동일).

    **분할 계약의 명시적 예외** — 이 단계만은 판정 표본 20,000행(train의 2.74%)을 포함한다.
    λ를 해방하는 run은 최소제곱이 λ를 다시 정하므로 판정 행을 빼도 게이트 수치가 6자리
    동일하고, λ 동결 run만 RMSE가 1.8% 폭으로 움직인다 (reports/stage_a.md 한계 절).

    Args:
        run_dir: 식별 진단을 train.log에 기록할 디렉토리.

    Returns:
        ((ν₀, r₁, r₂), 진단 dict) — 진단에는 채널별 λ 그리드와 자체 검증 수치가 담긴다.
    """
    x, y, train_idx, _ = prepare_train_arrays(val_frac=0.1, seed=_SPLIT_SEED)
    ident = identify_wavelength_grid(x[train_idx], y[train_idx])
    log_line(run_dir, describe_identification(ident["diagnostics"]))
    lam_grid = ident["lam_grid"]
    coeffs = fit_lam_coefficients(lam_grid)
    smooth = 1.0 / np.polyval(
        [coeffs[0] * coeffs[2], coeffs[0] * coeffs[1], coeffs[0]],
        np.arange(len(lam_grid)) / (len(lam_grid) - 1.0),
    )
    record = {
        **ident["diagnostics"],
        "lam_grid": [float(v) for v in lam_grid],
        "lam_coeffs": list(coeffs),
        # 채널별 추정과 매끈 곡선의 차이. 이것을 그대로 고정하면 그것만으로 R 오차
        # rms 0.0052가 생긴다 (남은 계통오차 전체 0.0041보다 크다).
        "smooth_fit_residual_rms_nm": float((smooth - lam_grid).std()),
        "smooth_fit_residual_max_nm": float(np.abs(smooth - lam_grid).max()),
    }
    log_line(
        run_dir,
        f"[freq-id] λ 3계수 적합: ν₀={coeffs[0]:.8f} r₁={coeffs[1]:+.6f} r₂={coeffs[2]:+.6f}"
        f" / 채널별 추정 대비 rms {record['smooth_fit_residual_rms_nm']:.3f} nm"
        f" (max {record['smooth_fit_residual_max_nm']:.3f})",
    )
    return coeffs, record


@torch.no_grad()
def residual_stats(
    model: PhysicalStack,
    x: np.ndarray,
    d: np.ndarray,
    *,
    channels: np.ndarray | None = None,
    batch: int = 4096,
) -> dict[str, float]:
    """잔차 요약 — RMSE와 유계 노이즈 위반율. channels를 주면 그 채널만 본다."""
    d_t = torch.from_numpy(d).to(torch.float64)
    sq = 0.0
    n_obs = 0
    n_viol = 0
    n_viol_loose = 0
    max_abs = 0.0
    for start in range(0, len(x), batch):
        pred = model(d_t[start : start + batch]).numpy()
        eps = x[start : start + batch].astype(np.float64) - pred
        if channels is not None:
            eps = eps[:, channels]
        sq += float((eps**2).sum())
        n_obs += eps.size
        abs_eps = np.abs(eps)
        n_viol += int((abs_eps > NOISE_BOUND).sum())
        n_viol_loose += int((abs_eps > NOISE_BOUND + 0.003).sum())
        max_abs = max(max_abs, float(abs_eps.max()))
    rmse = float(np.sqrt(sq / n_obs))
    return {
        "rmse": rmse,
        "rmse_over_sigma": rmse / NOISE_SIGMA,
        "violation_rate": n_viol / n_obs,
        "violation_rate_loose": n_viol_loose / n_obs,
        "max_abs_residual": max_abs,
        "n_obs": n_obs,
    }


def fit_physical(
    data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    lam_coeffs: tuple[float, float, float],
) -> dict[str, Any]:
    """신뢰영역 최소제곱 (scipy `least_squares`, **TRF**) — 자유도가 작아 한 번에 수렴한다.

    자유도 P ≤ 7이므로 2점 수치 야코비안이 충분하고, 그 야코비안이 그대로 **파라미터
    공분산**을 준다: cov = σ²(JᵗJ)⁻¹ — 물성값을 신뢰구간과 함께 보고하기 위한 것이
    경사하강 대신 이쪽을 쓰는 주된 이유다. (이름 주의: TRF이고 LM이 아니다. 실제 LM은
    게이트 (d)의 두께 역해 쪽이다.)

    Args:
        data: `load_split` 산출. cfg: 전체 config. run_dir: 산출물 디렉토리.
        lam_coeffs: λ 초기 계수 (ν₀, r₁, r₂).

    Returns:
        결과 dict (게이트 수치, 파라미터 표, 상관행렬, wall_sec 등).
    """
    model_cfg = cfg["model"]
    free = tuple(model_cfg.get("free", ()))
    model = PhysicalStack(
        n_channels=data["x_fit"].shape[1],
        lam_coeffs=lam_coeffs,
        free=free,
        si_source=str(model_cfg.get("si_source", DEFAULT_SI_SOURCE)),
    )
    hold = model_cfg.get("holdout_channels")
    hold_range = model_cfg.get("holdout_channel_range")
    n_ch = data["x_fit"].shape[1]
    if hold and hold_range:
        raise ValueError("holdout_channels 와 holdout_channel_range 를 함께 줄 수 없다")
    if hold_range:
        # **연속 블록** 홀드아웃 — 이웃 채널이 강상관이라 균등 간격판은 사실상 보간이고
        # 한계 효과가 +0.3%뿐이다. 대역 끝을 통째로 빼야 진짜 외삽이 된다.
        lo, hi = (int(v) for v in hold_range)
        if not 0 <= lo <= hi < n_ch:
            raise ValueError(f"holdout_channel_range {hold_range} 가 [0, {n_ch - 1}] 범위 밖")
        held = np.arange(lo, hi + 1, dtype=int)
        fit_channels = np.setdiff1d(np.arange(n_ch), held)
    elif hold:
        # 균등 간격 홀드아웃 — 매끈한 분산 모델만 이 채널을 예측할 수 있다
        # (채널별 자유 모델은 배정된 값이 없어 원리적으로 불가능).
        held = np.linspace(0, n_ch - 1, int(hold), dtype=int)
        fit_channels = np.setdiff1d(np.arange(n_ch), held)
    else:
        held, fit_channels = np.empty(0, dtype=int), np.arange(n_ch)

    x_fit_t = torch.from_numpy(data["x_fit"]).to(torch.float64)[:, fit_channels]
    d_fit_t = torch.from_numpy(data["d_fit"]).to(torch.float64)
    ch_idx = torch.from_numpy(fit_channels)
    n_eval = 0
    t_start = time.perf_counter()

    def residual(theta: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        with torch.no_grad():
            model.theta.copy_(torch.from_numpy(theta).to(torch.float64))
            pred = model(d_fit_t).index_select(1, ch_idx)
            res = (pred - x_fit_t).reshape(-1).numpy()
        n_eval += 1
        if n_eval % 10 == 1:
            log_line(
                run_dir,
                f"[phys] eval {n_eval:4d}  fit_rmse {np.sqrt((res**2).mean()):.6f}"
                f"  θ {np.array2string(theta, precision=3, max_line_width=200)}",
            )
        return res

    if free:
        result = least_squares(residual, np.zeros(len(free)), method="trf", xtol=1e-12, ftol=1e-12)
        theta_hat = result.x
        jac = result.jac
    else:
        theta_hat = np.zeros(0)
        jac = np.zeros((1, 0))
        residual(theta_hat)
    with torch.no_grad():
        model.theta.copy_(torch.from_numpy(theta_hat).to(torch.float64))

    # 파라미터 공분산 — 잔차 iid(σ) 가정이라 낙관적이다 (상관을 가진 모델 오차가 남아
    # 있으므로). 실질 불확실성은 독립 방법 간 일치도로 함께 본다 — reports/stage_a.md.
    params: list[dict[str, Any]] = []
    corr: list[list[float]] = []
    if free:
        try:
            cov_theta = NOISE_SIGMA**2 * np.linalg.inv(jac.T @ jac)
            sd_theta = np.sqrt(np.clip(np.diag(cov_theta), 0.0, None))
            outer = np.outer(sd_theta, sd_theta)
            corr = np.where(outer > 0, cov_theta / np.where(outer > 0, outer, 1.0), 0.0).tolist()
        except np.linalg.LinAlgError:
            sd_theta = np.full(len(free), np.nan)
        step = model.theta_step.numpy()
        for i, name in enumerate(free):
            init = float(getattr(model, f"init_{name}"))
            value = init + step[i] * theta_hat[i]
            sd = float(step[i] * sd_theta[i])
            params.append(
                {
                    "name": name,
                    "init": init,
                    "fitted": float(value),
                    "sd": sd,
                    "ci95": [float(value - 1.96 * sd), float(value + 1.96 * sd)],
                    "rel_change": (float(value / init - 1.0) if init != 0.0 else None),
                }
            )

    diag = residual_stats(model, data["x_diag"], data["d_diag"])
    out: dict[str, Any] = {
        "free": list(free),
        "n_free": len(free),
        "theta": theta_hat.tolist(),
        "params": params,
        "correlation": corr,
        "physical_values": model.physical_values(),
        "n_fit_rows": int(len(data["d_fit"])),
        "n_fit_channels": int(len(fit_channels)),
        "n_func_evals": n_eval,
        "wall_sec": time.perf_counter() - t_start,
        "diag": diag,
        "gate_a": {
            "rmse": diag["rmse"],
            "threshold": GATE_A_RMSE,
            "pass": diag["rmse"] < GATE_A_RMSE,
        },
        "gate_bound": {
            "violation_rate": diag["violation_rate"],
            "max_abs_residual": diag["max_abs_residual"],
            "bound": NOISE_BOUND,
            "pass": diag["violation_rate"] == 0.0,
        },
    }
    with torch.no_grad():
        lam = model.lam().numpy()
    out["lam_range"] = [float(lam.min()), float(lam.max())]
    if len(held):
        out["holdout"] = {
            "channels": held.tolist(),
            "held_out": residual_stats(model, data["x_diag"], data["d_diag"], channels=held),
            "fitted_channels": residual_stats(
                model, data["x_diag"], data["d_diag"], channels=fit_channels
            ),
        }
    _atomic_save(
        {
            "model_cfg": model.model_cfg,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "result": out,
        },
        run_dir / "model.pt",
    )
    log_line(
        run_dir,
        f"\n[phys] 자유도 {len(free)} / eval {n_eval} / {out['wall_sec']:.1f}s\n"
        f"[phys] 진단 RMSE {diag['rmse']:.6f} (σ 대비 {diag['rmse_over_sigma']:.3f})"
        f" — 게이트 (a) < {GATE_A_RMSE:.6f}: {'통과' if out['gate_a']['pass'] else '실패'}\n"
        f"[phys] 유계 노이즈 위반율 {diag['violation_rate']:.4%} "
        f"(max|잔차| {diag['max_abs_residual']:.5f} vs 상한 {NOISE_BOUND})"
        f" — 게이트 (b): {'통과' if out['gate_bound']['pass'] else '실패'}",
    )
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A 물리 제약 캘리브레이션")
    parser.add_argument("--config", required=True, help="configs/stage_a/*.yaml 경로")
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--force", action="store_true", help="완료된 run도 다시 실행")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """config를 읽어 피팅하고 runs/<실험>/<변형>/ 에 세 산출물을 남긴다."""
    args = _parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.run_name:
        cfg["run_name"] = args.run_name
    experiment, run_name = cfg["experiment"], cfg["run_name"]
    run_dir = RUNS_DIR / experiment / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and not args.force:
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "result" in prev:
            rmse = prev["result"]["diag"]["rmse"]
            print(f"[skip] {experiment}/{run_name} 이미 완료 (RMSE {rmse:.6f})")
            return

    log_line(run_dir, f"[phys] {experiment}/{run_name} 시작 — 자유 {cfg['model'].get('free', [])}")
    # λ 초기 계수: 두께축 주파수 식별 (결정론적 닫힌형 — run마다 재계산해도 같은 값).
    # 외부 산출물을 참조하지 않으므로 run이 자기완결적이다.
    lam_coeffs, lam_record = identify_lam_coefficients(run_dir)

    data = load_split(int(cfg["data"]["fit_rows"]))
    metrics: dict[str, Any] = {
        "experiment": experiment,
        "run_name": run_name,
        "config": cfg,
        "lam_coeffs": list(lam_coeffs),
        "lam_identification": lam_record,
        "rows": {"fit": int(len(data["d_fit"])), "diag": int(len(data["d_diag"]))},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics["result"] = fit_physical(data, cfg, run_dir, lam_coeffs=lam_coeffs)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{experiment}/{run_name}: RMSE {metrics['result']['diag']['rmse']:.6f}"
        f" / 위반율 {metrics['result']['diag']['violation_rate']:.4%}"
    )


if __name__ == "__main__":
    main()
