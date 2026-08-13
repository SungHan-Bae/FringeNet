"""문헌 광학상수 — Stage A 캘리브레이션의 물리 모델과 게이지 고정용.

**원본 파일을 저장소에 그대로 둔다**: `literature/*.yml` 은 refractiveindex.info
데이터베이스 파일 그대로다 (CC0 public domain). 문헌 표를 손으로 옮겨 적으면 뾰족한
임계점 구조가 조용히 깎이므로(아래 `_SI_*` 대조군이 그 예다) 파싱해서 쓴다.

수록 문헌:
  - SiO₂ (fused silica): **Malitson 1965 Sellmeier** — 게이지 고정(freeze) 대상.
    delta = 2πnd/λ 가 (n, λ)의 공통 스케일에 불변이라 SiO₂를 문헌값에 못박아야
    λ 그리드가 식별된다 (CLAUDE.md Level 2 게이지 고정).
  - Si₃N₄: **Luke et al. 2015 Sellmeier** — B₁(·C₁)이 자유 파라미터, 나머지는 동결.
    유효범위 310–5504 nm (그 아래 채널은 외삽 — 한계로 기록).
  - Si (결정질): **Aspnes & Studna 1983** 실측표 (에너지축 0.1 eV 균등이라 E1(3.4 eV)
    봉우리에 격자점이 놓인다) + Green 2008 / Schinke 2015 (ablation 대조군).

파장축이 비식별화되어 있으므로 (CLAUDE.md 데이터 계약) 여기의 λ[nm]는 캘리브레이션이
식별한 그리드에서만 평가된다.

제공하는 것:
  - `sellmeier_n_t`: λ·계수 양쪽으로 미분가능한 **정확한** Sellmeier (절단 근사 없음).
  - `TabulatedNK`: 실측 n·k를 **광자 에너지축 3차 스플라인**으로 평가 (λ로 미분가능).
    에너지축이 임계점 구조의 자연 좌표다.
  - `CoarseTableNK`: 거친 19점 표 + λ축 선형 보간 — **표·보간 품질의 기여를 재는
    ablation 대조군**. 실사용 금지 (E1 봉우리를 0.288 = 4.3% 깎는다).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "HC_EV_NM",
    "LITERATURE_DIR",
    "SI3N4_LUKE_SELLMEIER",
    "SIO2_MALITSON_SELLMEIER",
    "CoarseTableNK",
    "TabulatedNK",
    "load_formula1_coefficients",
    "load_tabulated_nk",
    "sellmeier_n_t",
    "si3n4_n",
    "si_nk",
    "si_nk_coarse_table",
    "sio2_n",
]

# Sellmeier: n²(λ) = 1 + Σ_i B_i λ² / (λ² − C_i),  λ[μm].
# 아래 하드코딩 값은 `literature/*.yml` 원본과 일치해야 한다
# (tests/test_dispersion_literature.py 가 대조한다).
_SIO2_SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)  # Malitson 1965
_SIO2_SELLMEIER_C_UM2 = (0.0684043**2, 0.1162414**2, 9.896161**2)
_SI3N4_SELLMEIER_B = (3.0249, 40314.0)  # Luke et al. 2015
_SI3N4_SELLMEIER_C_UM2 = (0.1353406**2, 1239.842**2)

# --- ablation 대조군 전용 ---
# 결정질 Si의 n·k를 문헌 그래프에서 눈대중으로 옮긴 19점 표. **실사용 금지** —
# 380 nm 미만이 특히 거칠고(270 nm의 n = 2.6은 c-Si 값이 아니다), λ축 선형 보간과
# 결합되면 E1 봉우리가 6.767 → 6.479로 깎인다. 원본 실측표(`TabulatedNK`) 대비
# 두께 역해 MAE가 0.663 → 1.104 nm로 나빠지는 것을 재는 대조군으로만 쓴다.
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
    """거친 19점 표의 (n, k) — 선형 보간, 범위 밖 끝값 유지. **ablation 대조군 전용**."""
    lam = np.asarray(lam_nm, dtype=np.float64)
    return np.interp(lam, _SI_LAM_NM, _SI_N), np.interp(lam, _SI_LAM_NM, _SI_K)


# ---------------------------------------------------------------------------
# 문헌 원본 파일 파싱
# ---------------------------------------------------------------------------

LITERATURE_DIR = Path(__file__).resolve().parent / "literature"

# 광자 에너지 변환 상수 E[eV] = HC_EV_NM / λ[nm] (CODATA hc).
HC_EV_NM = 1239.841984

# tabulated nk 블록의 한 줄: λ[μm] n k
_NUM3 = re.compile(r"^\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$")


def load_tabulated_nk(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """refractiveindex.info YAML의 `tabulated nk` 블록을 읽는다 (원본 파일 그대로).

    Args:
        filename: `literature/` 아래 파일명 (예: "Si_nk_Aspnes.yml").

    Returns:
        (lam_nm (M,), n (M,), k (M,)) — 전부 float64, λ 오름차순.
    """
    text = (LITERATURE_DIR / filename).read_text(encoding="utf-8")
    rows = [m.groups() for line in text.splitlines() if (m := _NUM3.match(line))]
    if not rows:
        raise ValueError(f"{filename}에서 tabulated nk 3열 데이터를 찾지 못했다")
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0] * 1000.0, arr[:, 1], arr[:, 2]


def load_formula1_coefficients(filename: str) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """refractiveindex.info `formula 1`(Sellmeier) 계수를 읽는다.

    형식: n²(λ) = 1 + c₀ + Σ_i B_i λ² / (λ² − C_i²),  λ[μm].

    Returns:
        (c0, B (i,), C_um (i,)) — C는 μm 단위 (제곱하지 않은 값).
    """
    text = (LITERATURE_DIR / filename).read_text(encoding="utf-8")
    match = re.search(r"coefficients:\s*([-\d.eE+\s]+)", text)
    if match is None:
        raise ValueError(f"{filename}에서 formula 1 coefficients를 찾지 못했다")
    values = [float(v) for v in match.group(1).split()]
    if len(values) < 3 or len(values) % 2 == 0:
        raise ValueError(f"{filename}의 계수 개수가 formula 1 형식(홀수)이 아니다: {len(values)}")
    return values[0], tuple(values[1::2]), tuple(values[2::2])


def _sellmeier_from_file(filename: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """문헌 파일에서 (B, C[μm²])를 읽는다. c₀ ≠ 0인 문헌은 이 프로젝트에 없다."""
    c0, b, c_um = load_formula1_coefficients(filename)
    if c0 != 0.0:
        raise ValueError(f"{filename}: c₀ ≠ 0 (={c0}) — 현재 구현이 다루지 않는다")
    return b, tuple(v**2 for v in c_um)


# 문헌 원본 파일에서 읽은 Sellmeier 계수 — 코드의 단일 출처.
SIO2_MALITSON_SELLMEIER = _sellmeier_from_file("SiO2_nk_Malitson.yml")
SI3N4_LUKE_SELLMEIER = _sellmeier_from_file("Si3N4_nk_Luke.yml")


def sellmeier_n_t(lam_nm: Tensor, b: Tensor, c_um2: Tensor) -> Tensor:
    """미분가능 Sellmeier n(λ) — λ·계수 양쪽으로 gradient가 흐른다.

    n²(λ) = 1 + Σ_i b_i λ² / (λ² − c_i),  λ[μm]. Cauchy 절단 근사를 쓰지 않는다.

    Args:
        lam_nm: (W,) 파장 [nm].
        b: (I,) 진동자 세기. c_um2: (I,) 공명 위치의 제곱 [μm²].

    Returns:
        n: (W,) — lam_nm과 같은 dtype.
    """
    lam2 = (lam_nm * 1e-3).unsqueeze(-1) ** 2  # (W, 1) [μm²]
    return torch.sqrt(1.0 + (b * lam2 / (lam2 - c_um2)).sum(dim=-1))


def _natural_cubic_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """자연 3차 스플라인 계수 — scipy 의존 없이 삼중대각 해로 구한다.

    Args:
        x: (M,) 강단조 증가 절점. y: (M,) 값.

    Returns:
        (4, M-1) — 구간 i에서 y(x) = Σ_p coef[p, i] · (x − x_i)^(3−p).
    """
    m = len(x)
    if m < 4 or np.any(np.diff(x) <= 0):
        raise ValueError(f"절점은 4개 이상·강단조 증가여야 한다 (받은 개수 {m})")
    h = np.diff(x)
    # 2차 미분 계수 sigma를 자연 경계조건(양끝 0)으로 푼다.
    a = np.zeros((m, m), dtype=np.float64)
    rhs = np.zeros(m, dtype=np.float64)
    a[0, 0] = a[-1, -1] = 1.0
    for i in range(1, m - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    sigma = np.linalg.solve(a, rhs)
    c3 = y[:-1]
    c2 = (y[1:] - y[:-1]) / h - h * (2.0 * sigma[:-1] + sigma[1:]) / 3.0
    c1 = sigma[:-1]
    c0 = (sigma[1:] - sigma[:-1]) / (3.0 * h)
    return np.stack([c0, c1, c2, c3])


class TabulatedNK(nn.Module):
    """문헌 실측 n·k 표를 **광자 에너지축 3차 스플라인**으로 평가한다 (λ로 미분가능).

    왜 에너지축인가: c-Si의 광학상수는 E1(3.4 eV)·E2(4.25 eV) 임계점 구조가
    지배하고, Aspnes & Studna 1983이 0.1 eV 균등 격자라 봉우리에 격자점이 놓인다.
    λ축 선형 보간은 이 봉우리를 깎아낸다 (`CoarseTableNK` 참조).

    k는 3자리 수 넘게 변하고 음수가 될 수 없으므로 **log k**를 스플라인한다.

    Args:
        filename: `literature/` 아래 tabulated nk 파일명.
        k_floor: k ≤ 0 인 절점에만 대입하는 대체값. **양수 절점은 절대 건드리지
            않는다** — Green 2008은 1450 nm에서 k = 1.4e-13까지 내려가므로 일괄
            clip을 걸면 그 절점을 재현하지 못한다.
    """

    def __init__(self, filename: str, *, k_floor: float = 1e-30) -> None:
        super().__init__()
        lam_nm, n, k = load_tabulated_nk(filename)
        energy = HC_EV_NM / lam_nm  # λ 오름차순 → 에너지 내림차순
        order = np.argsort(energy)
        e_sorted = energy[order]
        self.filename = filename
        self.lam_range = (float(lam_nm.min()), float(lam_nm.max()))
        f64 = torch.float64
        self.register_buffer("knots", torch.from_numpy(e_sorted).to(f64))
        self.register_buffer(
            "coef_n", torch.from_numpy(_natural_cubic_coefficients(e_sorted, n[order])).to(f64)
        )
        k_pos = np.where(k[order] > 0.0, k[order], k_floor)
        self.register_buffer(
            "coef_logk",
            torch.from_numpy(_natural_cubic_coefficients(e_sorted, np.log(k_pos))).to(f64),
        )

    def _eval(self, coef: Tensor, energy: Tensor) -> Tensor:
        """스플라인 평가 — coef (4, M-1), energy (W,) → (W,). energy는 표 범위로 clamp."""
        e = energy.clamp(self.knots[0], self.knots[-1])
        idx = (torch.searchsorted(self.knots, e.detach().contiguous()) - 1).clamp(
            0, self.knots.numel() - 2
        )
        dx = e - self.knots[idx]
        out = coef[0, idx]
        for p in range(1, 4):
            out = out * dx + coef[p, idx]
        return out

    def forward(self, lam_nm: Tensor) -> tuple[Tensor, Tensor]:
        """λ [nm] (W,) → (n (W,), k (W,)). k > 0 보장 (log 공간 스플라인)."""
        energy = HC_EV_NM / lam_nm
        return self._eval(self.coef_n, energy), torch.exp(self._eval(self.coef_logk, energy))


def si_nk_coarse_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """거친 19점 Si 표 (λ_nm, n, k) — 표·보간 품질의 기여를 재는 ablation 전용."""
    return _SI_LAM_NM.copy(), _SI_N.copy(), _SI_K.copy()


class CoarseTableNK(nn.Module):
    """거친 표 + **λ축 선형 보간** n·k — 표·보간 품질의 기여를 분리해 재는 대조군.

    `TabulatedNK`(원본 실측표 + 에너지축 3차 스플라인)와 같은 인터페이스를 갖되,
    거친 19점 표를 λ축에서 선형 보간한다. 범위 밖은 끝값 유지 (np.interp 관례).
    n·k **둘 다 λ축 선형**이어야 `si_nk`(np.interp)와 비트 수준으로 같고, 그래야
    "표·보간만 다르다"는 대조가 성립한다 (log 공간 보간을 쓰면 안 된다).
    """

    def __init__(self) -> None:
        super().__init__()
        lam_nm, n, k = si_nk_coarse_table()
        f64 = torch.float64
        self.filename = "coarse-19pt"
        self.lam_range = (float(lam_nm.min()), float(lam_nm.max()))
        self.register_buffer("knots", torch.from_numpy(lam_nm).to(f64))
        self.register_buffer("n_vals", torch.from_numpy(n).to(f64))
        self.register_buffer("k_vals", torch.from_numpy(k).to(f64))

    def _lerp(self, values: Tensor, lam: Tensor) -> Tensor:
        """λ축 선형 보간 (범위 밖 끝값 유지)."""
        x = lam.clamp(self.knots[0], self.knots[-1])
        idx = (torch.searchsorted(self.knots, x.detach().contiguous()) - 1).clamp(
            0, self.knots.numel() - 2
        )
        lo, hi = self.knots[idx], self.knots[idx + 1]
        t = (x - lo) / (hi - lo)
        return values[idx] * (1.0 - t) + values[idx + 1] * t

    def forward(self, lam_nm: Tensor) -> tuple[Tensor, Tensor]:
        """λ [nm] (W,) → (n (W,), k (W,))."""
        return self._lerp(self.n_vals, lam_nm), self._lerp(self.k_vals, lam_nm)
