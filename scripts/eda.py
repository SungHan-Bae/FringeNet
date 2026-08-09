"""Task 3 EDA — 두께가 스펙트럼에 어떻게 인코딩되는지 그림으로 확인한다.

생성물 (`reports/figures/`):
  fig1_layer_sweep.png          한 층만 쓸었을 때 fringe가 조밀해지는가
  fig2_layer_sensitivity.png    네 층 중 어느 층이 스펙트럼에 잘 드러나는가
  fig3_reflectance_distribution.png  반사율 분포·범위와 노이즈 성격

측정값은 `reports/eda_metrics.md` 에 표로 기록한다 (재실행 시 덮어씀).
해석은 손으로 쓴 `reports/eda_notes.md` 에 있다.

사용법:
    python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (  # noqa: E402
    CHANNEL_COLS,
    LAYER_COLS,
    N_CHANNELS,
    REPO_ROOT,
    load_frame,
)

FIG_DIR = REPO_ROOT / "reports" / "figures"
METRICS_PATH = REPO_ROOT / "reports" / "eda_metrics.md"

GRID_STEP_NM = 10
N_GRID = 30  # 10..300 nm
SEED = 42

# Task 2에서 확정한 데이터 노이즈 수준 (반사율 단위). 모든 민감도의 판단 기준선.
NOISE_SIGMA = 0.0087

# 범주형 팔레트 — 층(entity)에 고정 배정. scripts/validate_palette.js all-pairs 통과
# (worst CVD ΔE 9.2, normal-vision ΔE 16.3). aqua는 대비 3:1 미만이라 직접 라벨 필수.
LAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
LAYER_LABELS = ["layer_1 (SiN)", "layer_2 (SiO2)", "layer_3 (SiN)", "layer_4 (SiO2)"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"


def _style_axes(ax: plt.Axes) -> None:
    """눈에 띄지 않는 축·격자 (마크가 주인공이다)."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=INK_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)


def _title(ax: plt.Axes, text: str, subtitle: str = "") -> None:
    """제목 위, 부제목 아래로 배치한다 (pad를 부제목 높이만큼 확보해야 겹치지 않는다)."""
    ax.set_title(text, color=INK_PRIMARY, fontsize=11, loc="left", pad=22 if subtitle else 6)
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=8.5,
            va="bottom",
        )


def _stagger(values: list[float], min_gap: float) -> list[float]:
    """직접 라벨이 겹치지 않도록 y 위치를 최소 간격만큼 벌린다 (원래 순서 유지)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    for rank in range(1, len(order)):
        prev, cur = order[rank - 1], order[rank]
        if adjusted[cur] - adjusted[prev] < min_gap:
            adjusted[cur] = adjusted[prev] + min_gap
    return adjusted


class GridIndex:
    """전수 조합 격자에서 (i1, i2, i3, i4) -> 행 번호 조회.

    train이 30^4 전수 조합임을 Task 2에서 확인했으므로, 두께 조합을 코드로 바꿔
    역인덱스를 한 번만 만들어 두면 "한 층만 바꾼 행"을 정확히 집어낼 수 있다.
    """

    def __init__(self, thickness_nm: np.ndarray) -> None:
        grid = thickness_nm // GRID_STEP_NM - 1  # 10..300 nm -> 0..29
        if grid.min() < 0 or grid.max() >= N_GRID:
            raise ValueError("두께가 10..300 nm 격자를 벗어난다")
        codes = ((grid[:, 0] * N_GRID + grid[:, 1]) * N_GRID + grid[:, 2]) * N_GRID + grid[:, 3]
        self.row_of_code = np.empty(N_GRID**4, dtype=np.int64)
        self.row_of_code.fill(-1)
        self.row_of_code[codes] = np.arange(len(codes), dtype=np.int64)
        if (self.row_of_code < 0).any():
            raise ValueError("격자에 빈 조합이 있다 (전수 조합이 아니다)")

    def rows(self, idx: np.ndarray) -> np.ndarray:
        """idx: (M, 4) 격자 인덱스 -> (M,) 행 번호."""
        codes = ((idx[:, 0] * N_GRID + idx[:, 1]) * N_GRID + idx[:, 2]) * N_GRID + idx[:, 3]
        return self.row_of_code[codes]


# ---------------------------------------------------------------------------
# 그림 1 — 한 층만 쓸기
# ---------------------------------------------------------------------------
def figure_layer_sweep(x: np.ndarray, index: GridIndex) -> dict[str, float]:
    """다른 층을 10 nm에 고정하고 한 층만 10->300 nm로 쓴다.

    30개 스펙트럼을 그대로 겹쳐 그리면 실타래가 되어 "두께↑ → fringe 조밀"이 안 보인다.
    위 행은 히트맵(y=두께, x=채널, 색=반사율)으로 30개를 한 번에 보여주고 — 무늬가
    두께에 따라 조밀해지는 것이 줄무늬 기울기로 드러난다 —, 아래 행은 세 두께만 뽑아
    스펙트럼 모양 자체를 읽을 수 있게 한다.

    나머지 세 층은 **최소값 10 nm**에 고정한다. 150 nm에 두면 고정 층들이 만드는 fringe가
    스펙트럼을 지배해서 쓸린 층의 기여가 묻힌다. 최소값으로 눌러야 "이 층 하나가 무늬를
    얼마나 조밀하게 만드는가"가 분리되어 보인다.
    """
    base = 0  # 격자 인덱스 0 = 10 nm (나머지 층을 최소로 눌러 쓸린 층을 분리)
    cmap = plt.get_cmap("Blues")  # 단일 색상 계열, 밝음->어두움 (순차 = 크기)
    picks = [0, 14, 29]  # 10 / 150 / 300 nm
    pick_shades = [cmap(v) for v in (0.42, 0.68, 0.95)]

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), facecolor=SURFACE, layout="constrained")
    channels = np.arange(N_CHANNELS)
    metrics: dict[str, float] = {}
    images = []

    for layer in range(4):
        idx = np.full((N_GRID, 4), base, dtype=np.int64)
        idx[:, layer] = np.arange(N_GRID)
        spectra = x[index.rows(idx)]

        ax_top = axes[0, layer]
        images.append(
            ax_top.imshow(
                spectra,
                aspect="auto",
                origin="lower",
                extent=(0, N_CHANNELS - 1, 5, 305),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
        )
        _style_axes(ax_top)
        ax_top.grid(False)
        _title(
            ax_top,
            LAYER_LABELS[layer],
            "stripes tilt and tighten as this layer goes 10 → 300 nm",
        )
        ax_top.set_ylabel("swept thickness (nm)", fontsize=9, color=INK_SECONDARY)

        ax_bot = axes[1, layer]
        label_y = _stagger([float(spectra[p][-1]) for p in picks], min_gap=0.07)
        for shade, pick, y_text in zip(pick_shades, picks, label_y, strict=True):
            ax_bot.plot(channels, spectra[pick], color=shade, linewidth=1.3)
            ax_bot.annotate(
                f"{(pick + 1) * GRID_STEP_NM} nm",
                xy=(N_CHANNELS - 1, spectra[pick][-1]),
                xytext=(N_CHANNELS + 6, y_text),
                textcoords="data",
                color=shade,
                fontsize=8.5,
                va="center",
            )
        _style_axes(ax_bot)
        ax_bot.set_xlim(0, N_CHANNELS - 1 + 34)
        ax_bot.set_ylim(-0.05, 1.0)
        ax_bot.set_xlabel(
            "wavelength channel index (de-identified)", fontsize=9, color=INK_SECONDARY
        )
        ax_bot.set_ylabel("reflectance", fontsize=9, color=INK_SECONDARY)

    bar = fig.colorbar(images[0], ax=axes[0, :], fraction=0.015, pad=0.01)
    bar.set_label("reflectance", fontsize=9, color=INK_SECONDARY)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    fig.suptitle(
        "Sweeping one layer at a time (the other three pinned at the 10 nm minimum "
        "to isolate the swept layer)\n"
        "top: all 30 thicknesses as a heatmap   ·   bottom: three thicknesses read as spectra",
        fontsize=12.5,
        color=INK_PRIMARY,
        ha="left",
        x=0.012,
    )
    fig.savefig(FIG_DIR / "fig1_layer_sweep.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# 그림 2 — 층별 민감도
# ---------------------------------------------------------------------------
def figure_layer_sensitivity(
    x: np.ndarray, index: GridIndex, n_pairs: int = 20_000
) -> dict[str, float]:
    """한 층만 10 nm 바뀐 두 행의 스펙트럼 차이를 층별로 비교한다.

    두께 정보가 스펙트럼에 얼마나 실려 있는지를 노이즈 sigma와 같은 축에서 본다.
    ΔR이 sigma에 못 미치는 층은 원리적으로 관측이 어렵다.
    """
    rng = np.random.default_rng(SEED)
    per_channel: list[np.ndarray] = []
    by_thickness: list[np.ndarray] = []
    metrics: dict[str, float] = {}

    for layer in range(4):
        idx_lo = rng.integers(0, N_GRID, size=(n_pairs, 4))
        idx_lo[:, layer] = rng.integers(0, N_GRID - 1, size=n_pairs)  # +1 스텝 여유
        idx_hi = idx_lo.copy()
        idx_hi[:, layer] += 1

        delta = x[index.rows(idx_hi)] - x[index.rows(idx_lo)]
        per_channel.append(np.abs(delta).mean(axis=0))

        rms_per_pair = np.sqrt((delta**2).mean(axis=1))
        rms = float(np.sqrt((delta**2).mean()))
        metrics[f"layer_{layer + 1}_rms_delta"] = rms
        metrics[f"layer_{layer + 1}_snr"] = rms / NOISE_SIGMA
        # 대역 양 끝의 민감도 비 — 어느 채널이 두께 정보를 더 많이 싣는가
        band_edges = per_channel[-1]
        metrics[f"layer_{layer + 1}_sens_low_channel"] = float(band_edges[:10].mean())
        metrics[f"layer_{layer + 1}_sens_high_channel"] = float(band_edges[-10:].mean())

        # 그 층 자신의 두께 구간별 민감도
        curve = np.array(
            [rms_per_pair[idx_lo[:, layer] == g].mean() for g in range(N_GRID - 1)],
            dtype=np.float64,
        )
        by_thickness.append(curve)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), facecolor=SURFACE, layout="constrained"
    )
    channels = np.arange(N_CHANNELS)

    for layer in range(4):
        ax_a.plot(
            channels,
            per_channel[layer],
            color=LAYER_COLORS[layer],
            linewidth=1.6,
            label=LAYER_LABELS[layer],
        )
    _noise_line(ax_a)
    _style_axes(ax_a)
    _title(
        ax_a,
        "Spectral response to a +10 nm step",
        "mean |ΔR| per channel, 20,000 random base stacks per layer",
    )
    ax_a.set_xlabel("wavelength channel index (de-identified)", fontsize=9, color=INK_SECONDARY)
    ax_a.set_ylabel("mean |ΔR|", fontsize=9, color=INK_SECONDARY)
    ax_a.set_xlim(0, N_CHANNELS - 1)
    ax_a.set_ylim(0, None)  # 노이즈 바닥과의 거리를 정직하게 보이려면 0에서 시작해야 한다
    ax_a.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper left")

    thickness_axis = np.arange(N_GRID - 1) * GRID_STEP_NM + 10
    for layer in range(4):
        ax_b.plot(
            thickness_axis,
            by_thickness[layer],
            color=LAYER_COLORS[layer],
            linewidth=1.6,
            label=LAYER_LABELS[layer],
        )
        # 직접 라벨 — aqua가 대비 기준 미달이라 색만으로 식별시키지 않는다
        ax_b.annotate(
            LAYER_LABELS[layer].split(" ")[0],
            xy=(thickness_axis[-1], by_thickness[layer][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=LAYER_COLORS[layer],
            fontsize=8.5,
            va="center",
        )
    _noise_line(ax_b)
    _style_axes(ax_b)
    _title(
        ax_b,
        "Where each layer is observable",
        "RMS ΔR for a +10 nm step, binned by that layer's own thickness",
    )
    ax_b.set_xlabel("thickness of the stepped layer (nm)", fontsize=9, color=INK_SECONDARY)
    ax_b.set_ylabel("RMS ΔR", fontsize=9, color=INK_SECONDARY)
    ax_b.set_xlim(10, thickness_axis[-1] + 42)
    ax_b.set_ylim(0, None)
    # 직접 라벨이 이미 각 선을 지목하므로 범례를 겹쳐 놓지 않는다 (노이즈 선과 충돌).

    fig.savefig(
        FIG_DIR / "fig2_layer_sensitivity.png", dpi=150, bbox_inches="tight", facecolor=SURFACE
    )
    plt.close(fig)
    return metrics


def _noise_line(ax: plt.Axes) -> None:
    """노이즈 바닥을 같은 축 위에 그어 민감도를 절대 기준으로 읽게 한다."""
    ax.axhline(NOISE_SIGMA, color=INK_SECONDARY, linewidth=1.0, linestyle="--", alpha=0.8)
    # x는 축 비율로 잡는다 — 데이터 좌표로 두면 xlim이 0에서 시작하지 않을 때 잘린다.
    ax.annotate(
        f"noise σ ≈ {NOISE_SIGMA:.4f}",
        xy=(0.02, NOISE_SIGMA),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
    )


# ---------------------------------------------------------------------------
# 그림 3 — 반사율 분포와 노이즈 성격
# ---------------------------------------------------------------------------
def figure_reflectance_distribution(x: np.ndarray) -> dict[str, float]:
    """반사율 분포·범위, 채널별 포락선, 고주파 잔차의 분포 형태."""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(16.5, 4.8), facecolor=SURFACE, layout="constrained"
    )
    metrics: dict[str, float] = {}

    # (a) 전체 반사율 히스토그램 — 음의 꼬리는 전체의 0.35%라 선형 축에서는 보이지 않는다.
    #     로그 축으로 그려야 "물리적으로 불가능한 구간에 실제로 값이 있다"가 눈에 보인다.
    counts, edges = np.histogram(x, bins=250, range=(float(x.min()), float(x.max())))
    centers = 0.5 * (edges[:-1] + edges[1:])
    floor = 0.5
    ax_a.fill_between(
        centers, np.maximum(counts, floor), floor, step="mid", color=LAYER_COLORS[0], linewidth=0
    )
    ax_a.axvspan(float(x.min()), 0.0, color=LAYER_COLORS[1], alpha=0.16, linewidth=0)
    ax_a.axvline(0.0, color=INK_PRIMARY, linewidth=1.0, linestyle="--")
    neg_frac = float((x < 0).mean())
    ax_a.set_yscale("log")
    # 로그 축은 위쪽이 압축되어 채운 영역이 화면을 거의 다 먹는다. 한 decade 남짓
    # 여유를 두어 설명 텍스트가 마크 위에 겹치지 않을 자리를 만든다.
    ax_a.set_ylim(floor, counts.max() * 60)
    # 텍스트는 채워진 영역 위 빈 공간에 두고, 좁은 음수 구간은 화살표로 지목한다.
    ax_a.annotate(
        f"{100 * neg_frac:.2f}% of values fall below 0 (min {x.min():.4f})\n"
        "— physically impossible, so the data carries additive noise",
        xy=(float(x.min()) / 2, counts.max() * 1.6),
        xytext=(0.97, 0.97),
        textcoords="axes fraction",
        ha="right",
        va="top",
        color=INK_SECONDARY,
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "lw": 0.9},
    )
    _style_axes(ax_a)
    _title(ax_a, "Reflectance distribution", f"all {x.size:,} values in train · log count axis")
    ax_a.set_xlabel("reflectance", fontsize=9, color=INK_SECONDARY)
    ax_a.set_ylabel("count (log)", fontsize=9, color=INK_SECONDARY)
    metrics["r_min"] = float(x.min())
    metrics["r_max"] = float(x.max())
    metrics["r_mean"] = float(x.mean())
    metrics["neg_fraction"] = neg_frac

    # (b) 채널별 평균 +/- 표준편차 포락선
    ch_mean, ch_std = x.mean(axis=0), x.std(axis=0)
    channels = np.arange(N_CHANNELS)
    ax_b.fill_between(
        channels,
        ch_mean - ch_std,
        ch_mean + ch_std,
        color=LAYER_COLORS[0],
        alpha=0.22,
        linewidth=0,
        label="±1 std across stacks",
    )
    ax_b.plot(channels, ch_mean, color=LAYER_COLORS[0], linewidth=1.6, label="channel mean")
    _style_axes(ax_b)
    _title(ax_b, "Per-channel envelope", "how much each channel varies across all 810,000 stacks")
    ax_b.set_xlabel("wavelength channel index (de-identified)", fontsize=9, color=INK_SECONDARY)
    ax_b.set_ylabel("reflectance", fontsize=9, color=INK_SECONDARY)
    ax_b.set_xlim(0, N_CHANNELS - 1)
    ax_b.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower right")
    metrics["channel_std_min"] = float(ch_std.min())
    metrics["channel_std_max"] = float(ch_std.max())

    # (c) 고주파 잔차 — 노이즈가 균등인지 가우시안인지
    # r_i = y_i - (y_{i-1} + y_{i+1})/2. 참 스펙트럼이 국소 선형이면 잔차는 노이즈만 남고,
    # 분산은 1.5*sigma^2. 균등 노이즈면 초과 첨도 -0.6, 가우시안이면 0 이어야 한다.
    #
    # 주의 — 이 잔차는 2차 차분의 -1/2 배다(r = -d2/2). 따라서 여기서 나오는 sigma 추정은
    # verify_data.py 의 2차 차분 추정과 **같은 추정량**이지 독립적인 확인이 아니다.
    # 독립적인 교차검증은 음수 하한 기반 a/sqrt(3) 쪽이다. 여기서 새로 얻는 정보는
    # 크기(sigma)가 아니라 **분포 모양(첨도)** 이다.
    #
    # 표본은 무작위로 뽑는다. 행이 (layer_1..layer_4) 사전식 정렬이라 x[:N] 은
    # layer_1 = 10 nm 구석만 보게 된다.
    rng = np.random.default_rng(SEED)
    sample = x[rng.choice(len(x), size=100_000, replace=False)]
    residual = sample[:, 1:-1] - 0.5 * (sample[:, :-2] + sample[:, 2:])
    # 곡률이 가장 적은 행(진폭 하위 10%)에서 따로 재면 노이즈 고유의 값에 가장 가깝다.
    # 곡률 오염은 sigma 를 위로, 첨도를 0(가우시안) 쪽으로 밀기 때문에, 두 값을 나란히
    # 보고하면 참값이 어느 쪽에 있는지 방향까지 읽을 수 있다.
    spread = sample.max(axis=1) - sample.min(axis=1)
    flat_residual = residual[spread <= np.percentile(spread, 10)].astype(np.float64).ravel()
    residual = residual.astype(np.float64).ravel()

    def _excess_kurtosis(v: np.ndarray) -> float:
        return float(((v - v.mean()) ** 4).mean() / v.var() ** 2 - 3.0)

    metrics["sigma_from_residual"] = float(residual.std() / np.sqrt(1.5))
    metrics["sigma_from_residual_flat"] = float(flat_residual.std() / np.sqrt(1.5))
    excess_kurtosis = _excess_kurtosis(residual)
    metrics["residual_excess_kurtosis"] = excess_kurtosis
    metrics["residual_excess_kurtosis_flat"] = _excess_kurtosis(flat_residual)

    counts_r, edges_r = np.histogram(residual, bins=200, density=True)
    centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])
    ax_c.fill_between(centers_r, counts_r, color=LAYER_COLORS[3], alpha=0.85, linewidth=0)
    gauss = np.exp(-0.5 * (centers_r / residual.std()) ** 2) / (residual.std() * np.sqrt(2 * np.pi))
    ax_c.plot(
        centers_r, gauss, color=INK_PRIMARY, linewidth=1.4, linestyle="--", label="Gaussian fit"
    )
    _style_axes(ax_c)
    _title(
        ax_c,
        "High-frequency residual",
        f"excess kurtosis {excess_kurtosis:+.3f} all rows, "
        f"{metrics['residual_excess_kurtosis_flat']:+.3f} flattest 10%   "
        "(uniform noise → −0.6, Gaussian → 0)",
    )
    ax_c.set_xlabel("residual  y[i] − (y[i−1]+y[i+1])/2", fontsize=9, color=INK_SECONDARY)
    ax_c.set_ylabel("density", fontsize=9, color=INK_SECONDARY)
    ax_c.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper right")

    fig.savefig(
        FIG_DIR / "fig3_reflectance_distribution.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
def write_metrics(metrics: dict[str, float]) -> None:
    """측정값을 표로 기록한다 — 해석은 eda_notes.md 에 손으로 쓴다."""
    lines = [
        "# EDA 측정값 (자동 생성)",
        "",
        "`python scripts/eda.py` 가 덮어쓴다. 해석은 `eda_notes.md` 참고.",
        "",
        "## 층별 민감도 — +10 nm 스텝이 스펙트럼에 만드는 변화",
        "",
        f"노이즈 바닥 σ = {NOISE_SIGMA} (Task 2 검증값) 기준.",
        "",
        "| 층 | RMS ΔR | SNR = RMS ΔR / σ | mean \\|ΔR\\| 채널 0~9 | 채널 216~225 | 비 |",
        "|---|---|---|---|---|---|",
    ]
    for layer in range(1, 5):
        low = metrics[f"layer_{layer}_sens_low_channel"]
        high = metrics[f"layer_{layer}_sens_high_channel"]
        lines.append(
            f"| {LAYER_LABELS[layer - 1]} "
            f"| {metrics[f'layer_{layer}_rms_delta']:.5f} "
            f"| {metrics[f'layer_{layer}_snr']:.2f} "
            f"| {low:.4f} | {high:.4f} | {high / low:.2f}× |"
        )

    lines += [
        "",
        "## 반사율 분포",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 범위 | [{metrics['r_min']:.6f}, {metrics['r_max']:.6f}] |",
        f"| 평균 | {metrics['r_mean']:.6f} |",
        f"| R < 0 비율 | {100 * metrics['neg_fraction']:.4f}% |",
        f"| 채널별 표준편차 범위 | [{metrics['channel_std_min']:.4f}, "
        f"{metrics['channel_std_max']:.4f}] |",
        "",
        "## 노이즈 성격 (고주파 잔차)",
        "",
        "| 항목 | 값 | 비고 |",
        "|---|---|---|",
        f"| σ — 전체 표본 (상한) | {metrics['sigma_from_residual']:.6f} "
        "| 2차 차분과 **같은 추정량** (r = −d2/2), 독립 확인이 아니다 |",
        f"| σ — 평평한 행 10% | {metrics['sigma_from_residual_flat']:.6f} "
        "| 곡률 오염이 가장 적어 참값에 가깝다 |",
        f"| 초과 첨도 — 전체 | {metrics['residual_excess_kurtosis']:+.3f} "
        "| 균등 −0.6 / 가우시안 0 |",
        f"| 초과 첨도 — 평평한 행 | {metrics['residual_excess_kurtosis_flat']:+.3f} "
        "| 곡률이 첨도를 0쪽으로 밀므로 이쪽이 노이즈 고유값에 가깝다 |",
        "",
    ]
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("데이터 로드 중...")
    frame = load_frame("train")
    x = frame[CHANNEL_COLS].to_numpy(dtype=np.float32)
    thickness = frame[LAYER_COLS].to_numpy(dtype=np.int64)
    del frame
    index = GridIndex(thickness)
    print(f"  x={x.shape}, 격자 인덱스 준비 완료")

    metrics: dict[str, float] = {}
    print("fig1 — 한 층 쓸기...")
    metrics |= figure_layer_sweep(x, index)
    print("fig2 — 층별 민감도...")
    metrics |= figure_layer_sensitivity(x, index)
    print("fig3 — 반사율 분포...")
    metrics |= figure_reflectance_distribution(x)

    write_metrics(metrics)

    print(f"\n그림 3종 -> {FIG_DIR}")
    print(f"측정값   -> {METRICS_PATH}")
    print("\n층별 민감도 (+10 nm 스텝):")
    for layer in range(1, 5):
        print(
            f"  {LAYER_LABELS[layer - 1]:16s} RMS ΔR = {metrics[f'layer_{layer}_rms_delta']:.5f}"
            f"   SNR = {metrics[f'layer_{layer}_snr']:5.2f}"
        )
    print(
        f"\n노이즈 σ: 전체 {metrics['sigma_from_residual']:.6f} (상한) / "
        f"평평한 행 {metrics['sigma_from_residual_flat']:.6f}"
    )
    print(
        f"초과 첨도: 전체 {metrics['residual_excess_kurtosis']:+.3f} / "
        f"평평한 행 {metrics['residual_excess_kurtosis_flat']:+.3f}  (균등 −0.6 / 가우시안 0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
