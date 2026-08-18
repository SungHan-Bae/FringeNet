"""헤드라인 그림 — baseline → CNN → 물리 보정 사다리와 213M 단독 대비.

수치는 전부 산출물에서 읽는다 (metrics.json 4종 + reports/cnn_recipe_judge.json) —
손으로 적는 수치가 없으므로 run·judge가 갱신되면 재실행만으로 그림이 따라온다.

산출물:
  reports/figures/fig_headline.png   (재실행 시 덮어씀)

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
OUT_PATH = REPO_ROOT / "reports" / "figures" / "fig_headline.png"

# 색 토큰은 scripts/eda.py와 같다 (그림들이 한 벌로 읽히게). 그림 라벨은 영문 — 저장소의
# 기존 그림 규약이고, 한글 폰트는 환경 의존이라 이식성이 없다.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
# 사다리는 순서 있는 **개선 단계**라 단일 색상 순차 램프 (범주형 팔레트가 아니다).
# 인접 단계가 닮는 것이 정상이므로 식별은 색이 아니라 행 라벨·직접 값 라벨이 맡는다.
LADDER_RAMP = ["#9ecae1", "#6baed6", "#4a98c9", "#2676b8", "#08306b"]
REFERENCE_GRAY = "#b5b3ac"


def _val_mae(run: str) -> float:
    metrics = json.loads((REPO_ROOT / "runs" / run / "metrics.json").read_text())
    return float(metrics["model"]["val_mae"])


def _judge_budget100() -> dict[str, float]:
    """cnn_recipe_judge.json에서 budget100 행(post-LM·되돌림 후)을 찾는다."""
    sidecar = json.loads((REPO_ROOT / "reports" / "cnn_recipe_judge.json").read_text())
    for row in sidecar["runs"]:
        if row["run"] == "cnn_recipe/budget100":
            return row
    raise KeyError("cnn_recipe_judge.json에 cnn_recipe/budget100 행이 없다 — judge를 다시 돌릴 것")


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.grid(True, color=INK_MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    judge = _judge_budget100()
    skip_mlp = _val_mae("strong_baseline/winner-repro-asis")
    rungs = [
        ("MLP baseline (0.65M)", _val_mae("mlp_baseline/dropout0.0")),
        ("1D CNN flatten-dilated-bound", _val_mae("level1_cnn/flatten-dilated-bound")),
        ("+ training budget (100 epochs)", _val_mae("cnn_recipe/budget100")),
        ("+ LM inversion (frozen TMM decoder)", float(judge["lm"])),
        ("+ label-free fallback rule", float(judge["fb_mae"])),
    ]

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), width_ratios=[2.5, 1.0], facecolor=SURFACE
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
        "0.66M CNN + physics-based refinement", fontsize=11.5, color=INK_PRIMARY, loc="left"
    )

    # ── 오른쪽: 최종 대결 확대 — 축이 0에서 시작하지 않으므로 길이(막대)가 아니라
    # 위치(점)로 부호화한다 (잘린 축의 막대는 차이를 과장한다).
    duel = [("0.66M + physics", float(judge["fb_mae"])), ("213M skip-MLP", skip_mlp)]
    colors = [LADDER_RAMP[-1], REFERENCE_GRAY]
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
        f"322x fewer parameters, {delta:.4f} nm better",
        fontsize=11.5,
        color=INK_PRIMARY,
        loc="left",
    )

    fig.suptitle(
        "FringeNet — physics at inference: a 0.66M model overtakes the 213M single model",
        fontsize=13,
        color=INK_PRIMARY,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"산출물: {OUT_PATH}")
    for name, v in rungs:
        print(f"  {name:38s} {v:.4f}")
    print(f"  {'213M skip-MLP (reference)':38s} {skip_mlp:.4f}")


if __name__ == "__main__":
    main()
