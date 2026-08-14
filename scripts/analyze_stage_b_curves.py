"""Stage B β ablation — 같은 학습 적합 수준에서 일반화를 대조한다.

라운드 1의 원 비교(최종 에폭끼리)는 **적합 수준이 서로 다른 지점을 맞대고 있다**. 물리 항이
최적화를 늦추면 β>0 run은 덜 적합된 상태로 끝나고, 덜 적합된 모델은 일반화 격차가 자명하게
작다 — 그래서 "격차가 β와 함께 줄어든다"는 관측만으로는 정규화 효과를 주장할 수 없다.

이 스크립트는 축을 에폭에서 **train_l1(지도 항 학습 MAE)** 로 바꿔 대조한다. 같은 train_l1에서
val MAE를 비교하면 최적화 지연과 일반화 효과가 분리된다.

`train_l1`이 물리 항을 뺀 순수 지도 항이라는 것이 이 분석의 전제다 (src/losses.py의
`PhysicsParts.sup` → src/train_gpu.py가 `total`이 아니라 `sup`을 누적한다). β를 섞어 넣은
값이면 run 사이 비교가 성립하지 않는다.

**Δ는 노이즈와 같은 자리수다.** 에폭별 val MAE의 지터가 적합 구간에 따라 0.05~0.42 nm이고
val-vs-train 기울기가 약 0.67이라, 끝점 한 쌍의 Δ(0.01~0.05 nm)는 단독으로 유의하지 않다.
그래서 에폭별 Δ 분포와 부트스트랩 신뢰구간을 함께 내고, **적합 구간 절단을 여러 개** 보고한다
(하나를 골라 쓰면 별이 붙는 절단만 남는다).

읽는 데 필요한 주의:
- `train_l1`은 에폭 중 러닝 평균이다(모델이 변하는 중, BatchNorm train 모드). 에폭 말의
  깨끗한 eval 값이 아니므로 위로 편향돼 있다. 후반부 편향은 에폭 간 차이 수준(~0.01 nm)이라
  읽는 오프셋보다 작지만, 초반은 크므로 통계는 수렴 구간으로 제한한다.
- 같은 train_l1을 맞추면 **LR이 어긋난다** (β=0의 대조점이 더 이른 에폭 = 더 높은 LR).
  미수렴 지점이라 β=0에 불리한 방향이므로, β>0가 그래도 지면 방향은 견고하다.

산출물:
  reports/stage_b_curves.md
  reports/figures/fig_stage_b_curves.png

사용법:
    python scripts/analyze_stage_b_curves.py
    python scripts/analyze_stage_b_curves.py --run runs/stage_b/beta0 --run runs/stage_b/beta30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "reports" / "stage_b_curves.md"
FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig_stage_b_curves.png"

DEFAULT_RUNS = ("beta0", "beta30", "beta100", "beta300")

# 색 토큰은 scripts/eda.py와 같다 (그림들이 한 벌로 읽히게). 그림 라벨은 영문 — 저장소의
# 기존 그림 규약이고, 한글 폰트는 환경 의존이라 이식성이 없다.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
# β는 순서 있는 **크기**라 단일 색상 순차 램프를 쓴다 (범주형 팔레트가 아니다).
# Blues step 0.60/0.73/0.87/1.00 — 밝기 단조, 대비 4/4 >= 3:1, 인접 CVD ΔE 10.5.
# 순차 램프는 인접 단계가 닮는 것이 정상이므로 범례와 직접 라벨을 함께 붙여
# 색만으로 계열을 식별하지 않게 한다.
BETA_RAMP = ["#4a98c9", "#2676b8", "#09529d", "#08306b"]

# 통계를 자를 적합 구간. 이보다 위는 에폭 1~5의 급강하 구간이라 러닝 평균 편향이 크다.
CONVERGED_TRAIN_L1 = 8.0
# 절단 민감도를 함께 보고한다 — 하나만 쓰면 유의성이 절단 선택의 산물이 된다.
CUTS = (8.0, 4.0, 3.0)
# 부트스트랩 재표본 수·시드 (재현 고정)
N_BOOT = 4000
BOOT_SEED = 0
# 러닝 평균 편향/지터 추정에 쓰는 마지막 에폭 수
JITTER_EPOCHS = 5

_EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/\d+\s+train_l1 (?P<train>[\d.]+)\s+val_mae (?P<val>[\d.]+)"
)
_PHYS_RE = re.compile(r"val_phys (?P<val_phys>[\d.]+)")


class Curve(NamedTuple):
    """한 run의 학습 곡선. 배열은 모두 (E,) — 에폭 순서.

    Attributes:
        name: run 이름.
        beta: 물리 항 가중 (metrics.json 기준).
        epoch: 에폭 번호.
        train: train_l1 — 물리 항을 뺀 지도 항 MAE [nm].
        val: holdout MAE [nm].
        val_phys: holdout 재구성 L1 (β 곱하기 전). 물리 항이 없으면 NaN.
        best_epoch: 체크포인트로 남은 에폭.
    """

    name: str
    beta: float
    epoch: np.ndarray
    train: np.ndarray
    val: np.ndarray
    val_phys: np.ndarray
    best_epoch: int


def read_curve(run_dir: Path) -> Curve:
    """run 디렉토리의 train.log·metrics.json에서 곡선을 읽는다."""
    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.json"
    for path in (log_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(f"{path}가 없다 — git 추적 대상이므로 pull 상태를 확인할 것")
    epoch, train, val, phys = [], [], [], []
    for line in log_path.read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if m is None:
            continue
        epoch.append(int(m["epoch"]))
        train.append(float(m["train"]))
        val.append(float(m["val"]))
        mp = _PHYS_RE.search(line)
        phys.append(float(mp["val_phys"]) if mp else np.nan)
    if not epoch:
        raise ValueError(f"{log_path}에서 에폭 줄을 못 찾았다 — 로그 형식이 바뀌었나")
    metrics = json.loads(metrics_path.read_text())["model"]
    curve = Curve(
        name=run_dir.name,
        beta=float(metrics["physics"]["beta"]),
        epoch=np.asarray(epoch),
        train=np.asarray(train),
        val=np.asarray(val),
        val_phys=np.asarray(phys),
        best_epoch=int(metrics["best_epoch"]),
    )
    # train_l1이 단조 감소여야 train 축 보간이 성립한다. 어긋나면 조용히 틀린 값을 내므로 막는다.
    if not np.all(np.diff(curve.train) < 0):
        raise ValueError(f"{run_dir.name}: train_l1이 단조 감소가 아니다 — train 축 보간 불가")
    return curve


def interp_on_train(curve: Curve, values: np.ndarray, target: np.ndarray | float) -> np.ndarray:
    """train_l1 = target 지점의 `values`를 선형보간한다. 곡선 범위 밖이면 NaN.

    train 축은 감소하므로 뒤집어 넘긴다 (np.interp는 증가 축을 요구한다).
    """
    t = np.atleast_1d(np.asarray(target, dtype=float))
    out = np.interp(t, curve.train[::-1], values[::-1], left=np.nan, right=np.nan)
    inside = (t >= curve.train[-1]) & (t <= curve.train[0])
    return np.where(inside, out, np.nan)


class Matched(NamedTuple):
    """한 run의 best 에폭을 대조군의 같은 train_l1과 맞댄 결과 (단일 에폭 읽기)."""

    name: str
    beta: float
    train_l1: float
    val: float
    val_ref: float
    gap: float
    gap_ref: float
    equiv_epoch: float
    best_epoch: int
    val_phys: float
    val_phys_ref: float


def match_at_best(curve: Curve, ref: Curve) -> Matched:
    """curve의 best 에폭 적합 수준에서 ref를 보간해 맞댄다."""
    i = int(np.flatnonzero(curve.epoch == curve.best_epoch)[0])
    t = float(curve.train[i])
    val_ref = float(interp_on_train(ref, ref.val, t)[0])
    return Matched(
        name=curve.name,
        beta=curve.beta,
        train_l1=t,
        val=float(curve.val[i]),
        val_ref=val_ref,
        gap=float(curve.val[i] - t),
        gap_ref=float(val_ref - t),
        equiv_epoch=float(interp_on_train(ref, ref.epoch.astype(float), t)[0]),
        best_epoch=curve.best_epoch,
        val_phys=float(curve.val_phys[i]),
        val_phys_ref=float(interp_on_train(ref, ref.val_phys, t)[0]),
    )


def delta_per_epoch(curve: Curve, ref: Curve, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """적합 구간(train_l1 ≤ cut)의 에폭마다 Δval = val(curve) − val(ref, 같은 train_l1).

    Returns:
        (train_l1, delta) — 둘 다 (K,), 대조군 범위를 벗어난 에폭은 뺀다.
    """
    sel = curve.train <= cut
    t = curve.train[sel]
    d = curve.val[sel] - interp_on_train(ref, ref.val, t)
    ok = np.isfinite(d)
    return t[ok], d[ok]


class DeltaStats(NamedTuple):
    """에폭별 Δval 분포 요약. ci_lo/ci_hi는 중앙값의 부트스트랩 95% 구간."""

    cut: float
    n: int
    median: float
    ci_lo: float
    ci_hi: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


def delta_stats(curve: Curve, ref: Curve, cut: float) -> DeltaStats | None:
    """Δval 중앙값과 부트스트랩 신뢰구간. 표본이 부족하면 None."""
    _, d = delta_per_epoch(curve, ref, cut)
    if d.size < 4:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.median(rng.choice(d, (N_BOOT, d.size)), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return DeltaStats(cut=cut, n=int(d.size), median=float(np.median(d)), ci_lo=lo, ci_hi=hi)


def epoch_jitter(curve: Curve, cut: float) -> float:
    """적합 구간에서 인접 에폭 val MAE 차의 절대값 중앙값 — Δ를 읽을 때의 잣대."""
    sel = curve.train[1:] <= cut
    d = np.abs(np.diff(curve.val))[sel]
    return float(np.median(d)) if d.size else float("nan")


def tail_jitter(curve: Curve) -> float:
    """후반 에폭 val MAE의 표준편차 — 수렴 지점의 재현성 규모."""
    return float(curve.val[-JITTER_EPOCHS:].std())


def running_mean_bias(curve: Curve) -> float:
    """train_l1 러닝 평균 편향의 자체 추정 — 마지막 두 에폭의 차이."""
    return float(curve.train[-2] - curve.train[-1])


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.grid(True, color=INK_MUTED, alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)


def _title(ax: plt.Axes, text: str, subtitle: str = "") -> None:
    ax.set_title(text, color=INK_PRIMARY, fontsize=10.5, loc="left", pad=20 if subtitle else 6)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 8),
            xycoords="axes fraction",
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8.5,
            va="bottom",
        )


def _panel_trajectory(ax: plt.Axes, curves: list[Curve], colors: dict[str, str]) -> None:
    """(a) 전 구간 궤적 — 궤적이 겹치면 val은 적합 수준의 함수이고 β와 무관하다는 뜻이다."""
    lim = (2.0, 34.0)
    ax.plot(lim, lim, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        "val = train  (zero gap)",
        xy=(21.0, 21.0),
        xytext=(3, 3),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=7.5,
        rotation=45,
        rotation_mode="anchor",
    )
    for c in curves:
        ax.plot(
            c.train, c.val, color=colors[c.name], linewidth=2.0, zorder=3, label=f"β = {c.beta:g}"
        )
        i = int(np.flatnonzero(c.epoch == c.best_epoch)[0])
        ax.plot(
            c.train[i],
            c.val[i],
            "o",
            color=colors[c.name],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=4,
        )
    ax.annotate(
        "training progress",
        xy=(3.1, 2.75),
        xytext=(11.0, 2.45),
        color=INK_SECONDARY,
        fontsize=8,
        va="center",
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 0.9},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator([2, 3, 5, 10, 20, 30]))
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        handlelength=1.6,
        borderaxespad=0.8,
    )
    legend.set_title("physics weight", prop={"size": 8})
    legend.get_title().set_color(INK_SECONDARY)
    _title(ax, "(a) Full trajectory", "all four runs collapse onto one curve · dot = best epoch")
    ax.set_xlabel(
        "train_l1 — supervised train MAE, physics term excluded [nm]",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("holdout MAE [nm]", fontsize=9, color=INK_SECONDARY)


def _panel_zoom(
    ax: plt.Axes, curves: list[Curve], colors: dict[str, str], matched: list[Matched]
) -> None:
    """(b) 수렴 구간 확대 — 끝점끼리가 아니라 같은 train_l1에서 세로로 읽는다."""
    xlim, ylim = (2.14, 2.86), (2.28, 2.94)
    for c in curves:
        ax.plot(c.train, c.val, color=colors[c.name], linewidth=2.0, zorder=3)
    off = []
    for m in matched:
        if not (xlim[0] <= m.train_l1 <= xlim[1]):
            off.append(m)
            continue
        color = colors[m.name]
        if m.beta > 0:
            ax.plot(
                [m.train_l1] * 2,
                [m.val_ref, m.val],
                color=color,
                linewidth=1.4,
                linestyle=(0, (2, 2)),
                zorder=2,
            )
            ax.plot(
                m.train_l1,
                m.val_ref,
                "_",
                color=color,
                markersize=9,
                markeredgewidth=1.6,
                zorder=4,
            )
            ax.annotate(
                f"{m.val - m.val_ref:+.3f}",
                xy=(m.train_l1, (m.val + m.val_ref) / 2),
                xytext=(8, -3),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=8.5,
                # 곡선이 지나는 자리라 라벨 뒤를 배경색으로 덮는다
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2},
            )
        ax.plot(
            m.train_l1,
            m.val,
            "o",
            color=color,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=5,
        )
        ax.annotate(
            f"β={m.beta:g}",
            xy=(m.train_l1, m.val),
            xytext=(0, 12),
            textcoords="offset points",
            color=color,
            fontsize=9,
            ha="center",
            fontweight="bold",
        )
    if off:
        ax.annotate(
            " · ".join(
                f"β={m.beta:g} ends off-panel at ({m.train_l1:.2f}, {m.val:.2f})" for m in off
            ),
            xy=(0.98, 0.03),
            xycoords="axes fraction",
            color=INK_MUTED,
            fontsize=7.5,
            ha="right",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _style_axes(ax)
    _title(
        ax,
        "(b) Converged region (zoom)",
        "Δ read vertically at matched train_l1 — β>0 sits above (worse)",
    )
    ax.set_xlabel("train_l1 [nm]", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("holdout MAE [nm]", fontsize=9, color=INK_SECONDARY)


def _panel_delta(ax: plt.Axes, curves: list[Curve], ref: Curve, colors: dict[str, str]) -> None:
    """(c) 에폭별 Δval 분포 — 0과 구별되는지가 질문이므로 점 분포 + 중앙값·신뢰구간으로 본다."""
    rows = [c for c in curves if c.beta > 0]
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.1, zorder=2)
    for row, c in enumerate(rows):
        y = len(rows) - 1 - row
        t, d = delta_per_epoch(c, ref, CONVERGED_TRAIN_L1)
        ax.plot(
            d,
            np.full_like(d, y),
            "o",
            color=colors[c.name],
            markersize=7,
            alpha=0.42,
            markeredgecolor="none",
            zorder=3,
        )
        s = delta_stats(c, ref, CONVERGED_TRAIN_L1)
        if s is None:
            continue
        ax.plot(
            [s.ci_lo, s.ci_hi],
            [y, y],
            color=colors[c.name],
            linewidth=2.4,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.plot(
            s.median,
            y,
            "D",
            color=colors[c.name],
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=1.8,
            zorder=5,
        )
        ax.annotate(
            f"median {s.median:+.3f}   n={s.n}",
            xy=(s.median, y),
            xytext=(0, 15),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8,
            ha="center",
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"β = {c.beta:g}" for c in reversed(rows)], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.annotate(
        "β>0 worse →",
        xy=(0.99, 0.02),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=8,
        ha="right",
    )
    ax.annotate(
        "← β>0 better",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=8,
        ha="left",
    )
    _style_axes(ax)
    ax.grid(False, axis="y")
    _title(
        ax,
        "(c) Δ holdout MAE at matched fit",
        f"one dot per epoch (train_l1 ≤ {CONVERGED_TRAIN_L1:g}) · diamond = median, "
        "bar = 95% CI (bootstrap)",
    )
    ax.set_xlabel("Δ holdout MAE  (β>0 − β=0) [nm]", fontsize=9, color=INK_SECONDARY)


def make_figure(curves: list[Curve], ref: Curve, matched: list[Matched]) -> None:
    """3면 그림: 전 구간 궤적 · 수렴 구간 확대 · 같은 적합에서의 Δval 분포."""
    colors = {c.name: BETA_RAMP[min(i, len(BETA_RAMP) - 1)] for i, c in enumerate(curves)}
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(16.5, 5.2), facecolor=SURFACE, layout="constrained"
    )
    _panel_trajectory(ax_a, curves, colors)
    _panel_zoom(ax_b, curves, colors, matched)
    _panel_delta(ax_c, curves, ref, colors)
    fig.suptitle(
        "Stage B — most of the physics term's cost is slower optimization; at matched fit it is "
        "never better, and the residual sits on the harmful side",
        color=INK_PRIMARY,
        fontsize=12.5,
        x=0.005,
        ha="left",
    )
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_report(curves: list[Curve], ref: Curve, matched: list[Matched]) -> str:
    """reports/stage_b_curves.md 본문."""
    others = [c for c in curves if c.beta > 0]
    lines = [
        "# Stage B — 적합 수준을 맞춘 β 대조",
        "",
        f"산출: `python scripts/{Path(__file__).name}` (재실행 시 덮어씀). "
        "입력은 `runs/stage_b/*/train.log` + `metrics.json`.",
        "",
        "최종 에폭끼리의 비교는 **적합 수준이 다른 지점을 맞댄다**. 물리 항이 최적화를 늦추면",
        "β>0 run은 덜 적합된 상태로 끝나고, 덜 적합된 모델은 일반화 격차가 자명하게 작다.",
        "축을 에폭에서 `train_l1`로 바꾸면 최적화 지연과 일반화 효과가 분리된다.",
        "",
        "## 1. 같은 train_l1에서의 대조 (각 run의 best 에폭 적합 수준)",
        "",
        "`val(β=0)`은 대조군 곡선을 그 지점에서 선형보간한 값이다. `등가에폭`은 대조군이 같은",
        "적합에 도달한 에폭 — 물리 항이 삼킨 최적화 예산이다.",
        "**Δval은 단일 에폭 읽기라 그 자체로는 유의하지 않다** (§2가 판정한다).",
        "",
        "| run | β | train_l1 | val | val(β=0) | Δval | Δ% | 격차 | 격차(β=0) | β=0 등가에폭 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in matched:
        pct = "—" if m.beta == 0 else f"{(m.val / m.val_ref - 1) * 100:+.2f}%"
        dv = "—" if m.beta == 0 else f"{m.val - m.val_ref:+.4f}"
        lines.append(
            f"| `{m.name}` | {m.beta:g} | {m.train_l1:.4f} | {m.val:.4f} | {m.val_ref:.4f} | "
            f"{dv} | {pct} | {m.gap:.4f} | {m.gap_ref:.4f} | "
            f"{m.equiv_epoch:.1f} / {m.best_epoch} |"
        )
    lines += [
        "",
        "**최적화 예산**: β=100은 30에폭을 다 쓰고도 대조군의 "
        f"{matched[2].equiv_epoch:.0f}에폭 수준에 머문다 — 물리 항이 학습의 "
        f"{(1 - matched[2].equiv_epoch / matched[2].best_epoch) * 100:.0f}%를 삼켰다. "
        f"β=300은 {(1 - matched[3].equiv_epoch / matched[3].best_epoch) * 100:.0f}%다.",
        "",
        "**격차 비교가 무효인 이유가 이 표에 있다**: 원 비교에서 격차는 β와 함께 줄어들지만"
        " (0.151 → 0.059), 같은 적합 수준에서 보면 대조군의 격차가 오히려 더 작다"
        f" (β=100 지점에서 {matched[2].gap:.4f} vs {matched[2].gap_ref:.4f})."
        " 덜 적합된 모델은 격차가 자명하게 작으므로, 적합 수준을 맞추지 않은 격차 비교는"
        " 정규화 효과의 증거가 되지 못한다.",
        "",
        "## 2. Δval은 노이즈와 같은 자리수다 — 절단을 여러 개 본다",
        "",
        "에폭별 val MAE 지터가 크다. 적합 구간별 인접 에폭 |Δval| 중앙값:",
        "",
        "| run | train_l1 ≤ 8 | ≤ 4 | ≤ 3 |",
        "|---|---|---|---|",
    ]
    for c in curves:
        vals = [epoch_jitter(c, cut) for cut in CUTS]
        cells = " | ".join("—" if not np.isfinite(v) else f"{v:.4f}" for v in vals)
        lines.append(f"| `{c.name}` | {cells} |")
    lines += [
        "",
        "val-vs-train 기울기가 약 0.67이므로 train 축 정렬 오차 0.05 nm도 Δval 0.03 nm를 만든다.",
        "따라서 끝점 한 쌍이 아니라 **적합 구간의 모든 에폭**을 표본으로 쓰고, 중앙값의",
        f"부트스트랩 95% 신뢰구간(재표본 {N_BOOT:,}, 시드 {BOOT_SEED})을 함께 본다.",
        "**절단을 하나 고르면 유의성이 절단 선택의 산물이 되므로 셋을 다 싣는다.**",
        "",
        "| run | 절단 | n | Δval 중앙값 | 95% CI | 0 배제 |",
        "|---|---|---|---|---|---|",
    ]
    signs = []
    for c in others:
        for cut in CUTS:
            s = delta_stats(c, ref, cut)
            if s is None:
                lines.append(f"| `{c.name}` | ≤ {cut:g} | — | 표본 부족 | — | — |")
                continue
            signs.append(s.median)
            lines.append(
                f"| `{c.name}` | ≤ {cut:g} | {s.n} | {s.median:+.4f} | "
                f"[{s.ci_lo:+.4f}, {s.ci_hi:+.4f}] | {'예' if s.excludes_zero else '아니오'} |"
            )
    n_pos = sum(1 for v in signs if v > 0)
    lines += [
        "",
        f"**읽는 법**: 같은 적합에서 일반화 이득이 있으려면 Δval이 **음수**여야 한다"
        f" (β>0 곡선이 대조군 아래). 실제로는 {n_pos}/{len(signs)} 전부 양수로, β>0가 더 좋게"
        " 나오는 run·절단이 하나도 없다. 다만 신뢰구간은 절단에 따라 0을 포함한다 →"
        " **방향은 손해 쪽으로 일관되고 크기(0.05 nm 미만)는 이 데이터로 분해되지 않는다.**",
        "",
        "손해를 예상할 근거는 착수 전에 두 개 선언돼 있었고 둘 다 이 방향을 가리킨다:",
        "① 사전등록 1 — `R_obs = R(d) + ε`이라 물리 항은 지도 항의 **노이즈 낀 대리**다"
        " (정보를 더하지 않고 gradient에 ε를 더한다).",
        "② README §3.2의 수용 리스크 — 디코더가 게이트 (b) 미통과(위반율 9.99%)라"
        " **0.34 nm 계통 편향을 가진 기준으로 예측을 당긴다.** 이 표가 그 리스크의 측정값이다.",
        "",
        "## 3. 결론 — 손해의 대부분은 최적화 지연이고, 남은 몫도 손해 쪽이다",
        "",
        "원 비교의 열화와 같은 적합에서의 Δ를 나란히 두면 크기 차이가 분명하다.",
        "",
        "| run | 원 비교 Δval | 같은 적합 Δval | 최적화 지연이 설명하는 비율 |",
        "|---|---|---|---|",
    ]
    ref_best = ref.val[np.flatnonzero(ref.epoch == ref.best_epoch)[0]]
    for m in matched:
        if m.beta == 0:
            continue
        raw = m.val - ref_best
        matched_d = m.val - m.val_ref
        lines.append(
            f"| `{m.name}` | {raw:+.4f} | {matched_d:+.4f} | {(1 - matched_d / raw) * 100:.1f}% |"
        )
    lines += [
        "",
        "**백분율은 큰 β에서만 의미가 있다.** β=30의 원 열화 자체가 0.09 nm라 분모가 작고,"
        " 같은 적합 Δ가 §2의 노이즈 안이라 66%는 해상도 밖이다. 크기로 읽는 것이 안전하다 —"
        " **같은 적합에서 물리 항이 움직이는 폭은 어느 β에서도 0.05 nm 미만이고 부호는 항상"
        " 손해 쪽이며**, 원 비교의 나머지 열화는 최적화 지연 몫이다.",
        "",
        "## 4. 물리 항은 같은 적합 수준에서도 자기 목적함수를 개선한다",
        "",
        "`val_phys` = holdout 재구성 L1 (β 곱하기 전). 참 두께에서의 하한은 E|ε| = 0.0075.",
        "적합 수준을 맞춰도 개선이 남는다 — 손실은 지시받은 일을 하고 있고, 그 목적이 두께",
        "정확도와 정렬돼 있지 않을 뿐이다.",
        "",
        "| run | β | train_l1 | val_phys | val_phys(β=0) | Δ |",
        "|---|---|---|---|---|---|",
    ]
    for m in matched:
        d = "—" if m.beta == 0 else f"**{m.val_phys - m.val_phys_ref:+.6f}**"
        lines.append(
            f"| `{m.name}` | {m.beta:g} | {m.train_l1:.4f} | {m.val_phys:.6f} | "
            f"{m.val_phys_ref:.6f} | {d} |"
        )
    lines += [
        "",
        "## 5. 이 분석이 의존하는 것",
        "",
        "- **`train_l1`은 물리 항을 뺀 지도 항이다** (`src/losses.py`의 `PhysicsParts.sup`을 "
        "`src/train_gpu.py`가 `total` 대신 누적한다). β를 섞은 값이면 run 간 비교가 "
        "성립하지 않는다.",
        f"- **러닝 평균 편향**: `train_l1`은 에폭 중 평균이라 위로 편향돼 있다. 대조군 마지막 "
        f"두 에폭의 차이로 후반 편향을 {running_mean_bias(ref):.4f} nm 규모로 추정한다 — "
        "읽는 Δ 오프셋보다 작다.",
        "- **LR 불일치**: 같은 train_l1을 맞추면 대조군 쪽이 더 이른 에폭 = 더 높은 LR이다. "
        "미수렴 지점이라 대조군에 불리한 방향이므로, β>0가 그래도 지는 방향은 견고하다.",
        f"- **수렴 지점 재현성**: 대조군 마지막 {JITTER_EPOCHS}에폭 val 표준편차 "
        f"±{tail_jitter(ref):.4f} nm.",
        "- **이 축이 재는 것은 조합 보간이다** (무작위 split, 격자 위). 격자 밖·미학습 두께 "
        "값으로의 외삽은 held-out 두께 값 split이 재는 별 질문이다.",
        "",
        f"그림: `figures/{FIG_PATH.name}`",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="run 디렉토리 (반복 지정). 첫 번째가 대조군이다. 기본값은 stage_b β 4런",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dirs = (
        [Path(r) for r in args.runs]
        if args.runs
        else [REPO_ROOT / "runs" / "stage_b" / n for n in DEFAULT_RUNS]
    )
    curves = [read_curve(d) for d in run_dirs]
    ref = curves[0]
    if ref.beta != 0.0:
        print(f"경고: 대조군으로 쓰는 첫 run의 β가 {ref.beta:g}다 (0이 아니다)", file=sys.stderr)
    matched = [match_at_best(c, ref) for c in curves]
    if any(not np.isfinite(m.val_ref) for m in matched):
        bad = [m.name for m in matched if not np.isfinite(m.val_ref)]
        raise ValueError(f"대조군 곡선의 train_l1 범위를 벗어난 run: {bad} — 보간 불가")

    make_figure(curves, ref, matched)
    OUT_PATH.write_text(build_report(curves, ref, matched))

    print("같은 train_l1에서의 대조 (best 에폭 적합 수준):")
    for m in matched:
        if m.beta == 0:
            print(
                f"  {m.name:8} β={m.beta:6g}  train_l1 {m.train_l1:.4f}  val {m.val:.4f} (대조군)"
            )
            continue
        print(
            f"  {m.name:8} β={m.beta:6g}  train_l1 {m.train_l1:.4f}  "
            f"val {m.val:.4f} vs β=0 {m.val_ref:.4f}  Δ {m.val - m.val_ref:+.4f}  "
            f"등가에폭 {m.equiv_epoch:.1f}/{m.best_epoch}"
        )
    print("\n에폭별 Δval 중앙값과 부트스트랩 95% CI:")
    for c in curves[1:]:
        for cut in CUTS:
            s = delta_stats(c, ref, cut)
            if s is None:
                print(f"  {c.name:8} 절단 ≤{cut:<4g} 표본 부족")
                continue
            print(
                f"  {c.name:8} 절단 ≤{cut:<4g} n={s.n:2d}  중앙값 {s.median:+.4f}  "
                f"CI [{s.ci_lo:+.4f}, {s.ci_hi:+.4f}]  "
                f"{'0 배제' if s.excludes_zero else '0 포함'}"
            )
    print(f"\n산출물: {OUT_PATH}\n         {FIG_PATH}")


if __name__ == "__main__":
    main()
