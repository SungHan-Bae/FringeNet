"""헤드라인 그림 2종 — 개선 사다리(holdout)와 격자 밖 전이 반전(holdout↔test).

수치는 전부 산출물에서 읽는다 (metrics.json · reports/task8_judge.json ·
reports/leaderboard.json) — 손으로 적는 수치가 없으므로 run·judge·제출 기록이
갱신되면 재실행만으로 그림이 따라온다.

산출물:
  reports/figures/fig_headline.png   (재실행 시 덮어씀)
  reports/figures/fig_offgrid.png    (재실행 시 덮어씀)

사용법:
    python scripts/make_headline_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "reports" / "figures"

# 색 토큰은 scripts/eda.py와 같다 (그림들이 한 벌로 읽히게). 그림 라벨은 영문 — 저장소의
# 기존 그림 규약이고, 한글 폰트는 환경 의존이라 이식성이 없다.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
# 사다리는 순서 있는 **개선 단계**라 단일 색상 순차 램프 (범주형 팔레트가 아니다).
# 인접 단계가 닮는 것이 정상이므로 식별은 색이 아니라 행 라벨·직접 값 라벨이 맡는다.
LADDER_RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4a98c9", "#2676b8", "#08306b"]
REFERENCE_GRAY = "#b5b3ac"
ACCENT = LADDER_RAMP[-1]

ADOPTED_RUN = "task8/d2-fft"
SKIP_MLP_RUN = "strong_baseline/winner-repro-asis"
PIPE_RAW = "raw"  # leaderboard.json의 pipeline 필드가 이 접두로 시작하면 raw 제출이다


def _val_mae(run: str) -> float:
    metrics = json.loads((REPO_ROOT / "runs" / run / "metrics.json").read_text())
    return float(metrics["model"]["val_mae"])


def _judge_row(run: str) -> dict[str, float]:
    """task8_judge.json에서 run 행(post-LM lm · 되돌림 후 fb_mae)을 찾는다."""
    sidecar = json.loads((REPO_ROOT / "reports" / "task8_judge.json").read_text())
    for row in sidecar["runs"]:
        if row["run"] == run:
            return row
    raise KeyError(f"task8_judge.json에 {run} 행이 없다 — judge를 다시 돌릴 것")


def _leaderboard(run: str, raw: bool) -> dict:
    """leaderboard.json에서 (run, raw 여부)로 제출 기록을 찾는다."""
    book = json.loads((REPO_ROOT / "reports" / "leaderboard.json").read_text())
    for sub in book["submissions"]:
        if sub["run"] == run and sub["pipeline"].startswith(PIPE_RAW) == raw:
            return sub
    kind = "raw" if raw else "물리 파이프라인"
    raise KeyError(f"leaderboard.json에 {run}의 {kind} 제출 기록이 없다")


def _style_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=INK_MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def make_headline() -> list[tuple[str, float]]:
    judge = _judge_row(ADOPTED_RUN)
    skip_mlp = _val_mae(SKIP_MLP_RUN)
    rungs = [
        ("MLP baseline (0.65M)", _val_mae("mlp_baseline/dropout0.0")),
        ("1D CNN flatten-dilated-bound (0.66M)", _val_mae("level1_cnn/flatten-dilated-bound")),
        ("+ training budget (100 epochs)", _val_mae("cnn_recipe/budget100")),
        ("+ residual, depth x2, rFFT branch (1.52M)", _val_mae(ADOPTED_RUN)),
        ("+ LM inversion (frozen TMM decoder)", float(judge["lm"])),
        ("+ label-free fallback rule", float(judge["fb_mae"])),
    ]

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), width_ratios=[2.5, 1.0], facecolor=SURFACE
    )

    # ── 왼쪽: 사다리 (위→아래 = 진행 순서) ────────────────────────────────────
    labels = [name for name, _ in rungs]
    values = [v for _, v in rungs]
    y = range(len(rungs) - 1, -1, -1)
    ax.barh(list(y), values, height=0.62, color=LADDER_RAMP, edgecolor="none")
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.06, yi, f"{v:.4f}", va="center", ha="left", fontsize=9.5, color=INK_PRIMARY)
    ax.axvline(skip_mlp, color=INK_SECONDARY, lw=1.4, ls=(0, (4, 3)))
    ax.text(
        skip_mlp + 0.06,
        len(rungs) - 0.55,
        f"213M skip-MLP alone: {skip_mlp:.4f}",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
    )
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("holdout MAE [nm]  (81,000 rows, same split)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.18)
    _style_axes(ax)
    ax.set_title(
        "1.52M CNN + physics-based refinement", fontsize=11.5, color=INK_PRIMARY, loc="left"
    )

    # ── 오른쪽: 최종 대결 확대 — 축이 0에서 시작하지 않으므로 길이(막대)가 아니라
    # 위치(점)로 부호화한다 (잘린 축의 막대는 차이를 과장한다).
    duel = [("1.5M + physics", float(judge["fb_mae"])), ("213M skip-MLP", skip_mlp)]
    colors = [ACCENT, REFERENCE_GRAY]
    duel_y = (1, 0)
    axz.scatter(
        [v for _, v in duel], duel_y, s=140, color=colors, zorder=3, edgecolor=SURFACE, lw=1.5
    )
    for yi, (_, v), c in zip(duel_y, duel, colors, strict=True):
        axz.vlines(v, -0.6, yi, color=c, lw=1.1, alpha=0.45, zorder=2)
        axz.text(v, yi - 0.22, f"{v:.4f}", va="top", ha="center", fontsize=10, color=INK_PRIMARY)
    axz.set_yticks(list(duel_y), [name for name, _ in duel], fontsize=9)
    axz.set_ylim(-0.6, 1.6)
    lo = min(v for _, v in duel)
    axz.set_xlim(lo * 0.985, skip_mlp * 1.008)
    axz.set_xlabel("holdout MAE [nm], zoomed", fontsize=9.5, color=INK_SECONDARY)
    _style_axes(axz)
    delta = skip_mlp - float(judge["fb_mae"])
    axz.set_title(
        f"140x fewer parameters, {delta:.4f} nm better",
        fontsize=11.5,
        color=INK_PRIMARY,
        loc="left",
    )

    fig.suptitle(
        "FringeNet — physics at inference: a 1.5M pipeline overtakes the 213M single model",
        fontsize=13,
        color=INK_PRIMARY,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "fig_headline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"산출물: {out}")
    for name, v in rungs:
        print(f"  {name:44s} {v:.4f}")
    print(f"  {'213M skip-MLP (reference)':44s} {skip_mlp:.4f}")
    return rungs


def make_offgrid() -> None:
    """격자 밖 전이 slopegraph — holdout(격자 위 보간) vs 리더보드 test(연속 두께)."""
    judge = _judge_row(ADOPTED_RUN)
    series = [
        # (라벨, holdout, test, 강조 여부)
        (
            "d2-fft + LM + fallback (adopted)",
            float(judge["fb_mae"]),
            _leaderboard(ADOPTED_RUN, raw=False)["test_mae"],
            True,
        ),
        (
            "213M skip-MLP, raw",
            _val_mae(SKIP_MLP_RUN),
            _leaderboard(SKIP_MLP_RUN, raw=True)["test_mae"],
            False,
        ),
        (
            "d2-fft CNN, raw",
            _val_mae(ADOPTED_RUN),
            _leaderboard(ADOPTED_RUN, raw=True)["test_mae"],
            False,
        ),
        (
            "d2-se CNN, raw",
            _val_mae("task8/d2-se"),
            _leaderboard("task8/d2-se", raw=True)["test_mae"],
            False,
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    for label, hold, test, adopted in series:
        color = ACCENT if adopted else REFERENCE_GRAY
        lw = 2.4 if adopted else 1.6
        ax.plot([0, 1], [hold, test], color=color, lw=lw, zorder=3 if adopted else 2)
        ax.scatter([0, 1], [hold, test], s=46, color=color, zorder=4, edgecolor=SURFACE, lw=1.2)
        ink = INK_PRIMARY if adopted else INK_SECONDARY
        ax.text(-0.03, hold, f"{label}  {hold:.4f}", va="center", ha="right", fontsize=9, color=ink)
        pct = (test - hold) / hold * 100.0
        pct_txt = f"{pct:+.0f}%" if abs(pct) >= 1 else f"{pct:+.1f}%"
        right = f"{test:g} ({pct_txt})"  # test는 리더보드 기록값 그대로 (0.33895는 5자리다)
        if adopted:
            rank = _leaderboard(ADOPTED_RUN, raw=False).get("rank")
            if rank is not None:
                right += f" — rank {rank}"
        ax.text(1.03, test, right, va="center", ha="left", fontsize=9, color=ink)

    ax.set_xlim(-0.85, 0.55 + 0.85)
    ax.set_xticks(
        [0, 1],
        [
            "holdout MAE [nm]\n(on-grid interpolation)",
            "leaderboard test MAE [nm]\n(off-grid, continuous)",
        ],
    )
    ax.tick_params(axis="x", labelsize=9.5)
    _style_axes(ax, grid_axis="y")
    ax.set_title(
        "Off-grid reversal — every raw network degrades on continuous thicknesses;\n"
        "only the physics pipeline transfers",
        fontsize=12,
        color=INK_PRIMARY,
        loc="left",
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_offgrid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"산출물: {out}")
    for label, hold, test, _ in series:
        print(f"  {label:36s} {hold:.4f} -> {test:.4f}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_headline()
    make_offgrid()


if __name__ == "__main__":
    main()
