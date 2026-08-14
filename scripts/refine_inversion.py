"""역산 refinement — 물리를 손실이 아니라 **추론 후 보정**으로 쓴다 (사전등록 2).

Stage B의 물리 손실 항은 세 축에서 기각됐다. 남은 질문은 같은 물리를 **추론 시점**에
쓰면 값어치가 있느냐다: 신경망 예측 d_hat을 출발점으로 동결 TMM 디코더를 관측 R에 맞춰
재적합한다. 라벨을 쓰지 않으므로 test·실계측에도 그대로 적용된다.

네 팔은 **출발점만** 다르다 — 그 차이가 측정의 전부다.

    cnn        신경망 예측 그대로 (기준선)
    cnn+LM     본 검정. d_hat에서 출발
    center+LM  격자 중앙 균일값에서 출발 — 물리 단독. CNN의 기여가 "올바른 분지"인지 가른다
    truth+LM   d_true에서 출발 (**라벨 사용**) — 디코더 내재 편향이 정하는 상한, 게이트 (d)

후처리이므로 평가 규약대로 기준선과 **별도 행**으로 분리 보고한다. 격자 스냅은 쓰지 않는다.

산출물:
  reports/inversion_refine.md          (재실행 시 덮어씀)
  reports/figures/fig_inversion_refine.png

사용법:
    python scripts/refine_inversion.py --run runs/level1_cnn/flatten-dilated-bound
    python scripts/refine_inversion.py --run ... --rows 5000     # 빠른 확인 (무작위 표본)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda import (  # noqa: E402 — 팔레트·축 스타일 단일 출처
    INK_MUTED,
    INK_SECONDARY,
    LAYER_COLORS,
    SURFACE,
    _style_axes,
    _title,
)
from evaluate_axes import NOISE_MEAN_ABS, SUBSAMPLE_SEED, load_run  # noqa: E402

from src.data.dataset import (  # noqa: E402
    LAYER_COLS,
    REPO_ROOT,
    prepare_from_config,
    subsample_indices,
)
from src.evaluate import mae_per_layer, predict  # noqa: E402
from src.losses import DEFAULT_DECODER, FrozenDecoder  # noqa: E402
from src.physics.invert import (  # noqa: E402
    DEFAULT_BOX_NM,
    PHYSICAL_RANGE_NM,
    lm_invert,
    residual_l1_rows,
)

FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig_inversion_refine.png"
OUT_PATH = REPO_ROOT / "reports" / "inversion_refine.md"

# 격자 중앙 — "물리 단독" 대조군의 출발점. 라벨도 예측도 쓰지 않는 상수다.
CENTER_NM = 0.5 * (PHYSICAL_RANGE_NM[0] + PHYSICAL_RANGE_NM[1])
# 이동 상한 τ [nm] — "많이 움직인 행은 보정을 버린다"는 **라벨 없는** 규칙. inf = 사전등록 원안.
MOVE_CAPS = (1.0, 2.0, 5.0, 10.0, 20.0, float("inf"))
# 분지 실패 판정 [nm]. 격자 간격 10 nm의 절반 — 이보다 크면 이웃 조합으로 넘어간 것이다.
BRANCH_FAIL_NM = 5.0


def arm_stats(name: str, d: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """한 팔의 nm 오차 요약. mae_per_layer로 층별을 내고 꼬리를 함께 본다."""
    abs_err = np.abs(d - y)
    row_err = abs_err.mean(axis=1)
    return {
        "name": name,
        **mae_per_layer(d.astype(np.float64), y.astype(np.float64)),
        "median": float(np.median(row_err)),
        "p99": float(np.percentile(row_err, 99)),
        "max": float(row_err.max()),
        "branch_fail": float((row_err > BRANCH_FAIL_NM).mean()),
        "out_of_physical": float(((d < PHYSICAL_RANGE_NM[0]) | (d > PHYSICAL_RANGE_NM[1])).mean()),
        "row_err": row_err,
    }


def move_cap_curve(d_hat: np.ndarray, d_ref: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    """이동 상한 τ 규칙 — τ보다 많이 움직인 행은 보정을 버리고 d_hat을 쓴다.

    라벨을 쓰지 않으므로 test에도 적용 가능한 규칙이다. 다만 **τ를 holdout MAE로 고르면
    평가셋 선택**이 되므로, 곡선을 통째로 싣고 사전등록 원안(τ = ∞)을 판정에 쓴다.
    """
    move = np.abs(d_ref - d_hat).max(axis=1)
    out = []
    for tau in MOVE_CAPS:
        keep = move <= tau
        blended = np.where(keep[:, None], d_ref, d_hat)
        out.append(
            {
                "tau": tau,
                "kept": float(keep.mean()),
                "mae": mae_per_layer(blended, y.astype(np.float64))["overall"],
            }
        )
    return out


def decile_table(base_err: np.ndarray, ref_err: np.ndarray) -> list[dict[str, float]]:
    """CNN 행별 오차 십분위로 나눠 refinement 효과를 본다 (라벨을 쓰는 **진단**이다).

    "CNN이 올바른 분지까지만 데려다 놓으면 물리가 마무리한다"는 주장은 십분위마다 다르게
    나타난다 — 이미 가까운 행은 다듬어지고, 멀리 있는 행은 다른 분지로 굳는다.
    """
    order = np.argsort(base_err)
    return [
        {
            "decile": i + 1,
            "base": float(base_err[idx].mean()),
            "refined": float(ref_err[idx].mean()),
            "improved": float((ref_err[idx] < base_err[idx]).mean()),
        }
        for i, idx in enumerate(np.array_split(order, 10))
    ]


def figure_path_for(out_path: Path) -> Path:
    """리포트와 짝이 되는 그림 경로 — `--out`으로 리포트를 옮기면 그림도 함께 옮긴다.

    고정 경로로 두면 스모크 실행이 커밋된 그림을 조용히 덮어쓴다 (실제로 한 번 덮었다).
    """
    return out_path.parent / "figures" / f"fig_{out_path.stem}.png"


def figure(arms: list[dict[str, Any]], deciles: list[dict[str, float]], fig_path: Path) -> None:
    """2패널 — (a) 행별 오차 ECDF, (b) CNN 오차 십분위별 before/after.

    그림 안 글자는 **영문**이다 — 기본 폰트에 한글 글리프가 없어 네모로 렌더된다
    (저장소의 다른 그림도 같은 이유로 영문이다).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    colors = {
        "cnn": INK_MUTED,
        "cnn+LM": LAYER_COLORS[0],
        "center+LM": LAYER_COLORS[1],
        "truth+LM": LAYER_COLORS[2],
    }

    ax = axes[0]
    _style_axes(ax)
    for arm in arms:
        err = np.sort(arm["row_err"])
        ax.plot(
            err,
            np.arange(1, len(err) + 1) / len(err),
            color=colors.get(arm["name"], INK_SECONDARY),
            linewidth=1.6,
            label=f"{arm['name']} — MAE {arm['overall']:.3f} nm",
        )
    ax.set_xscale("log")
    ax.set_xlabel("per-row mean absolute error [nm]")
    ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _title(ax, "(a) Error distribution", "four arms differing only in the LM starting point")

    ax = axes[1]
    _style_axes(ax)
    pos = np.arange(len(deciles))
    ax.bar(pos - 0.2, [d["base"] for d in deciles], width=0.4, color=INK_MUTED, label="cnn")
    ax.bar(
        pos + 0.2,
        [d["refined"] for d in deciles],
        width=0.4,
        color=LAYER_COLORS[0],
        label="cnn+LM",
    )
    ax.set_yscale("log")
    ax.set_xticks(pos, [str(d["decile"]) for d in deciles])
    ax.set_xlabel("decile of CNN per-row error (1 = most accurate)")
    ax.set_ylabel("mean absolute error [nm]")
    ax.legend(frameon=False, fontsize=8)
    improved = float((arms[1]["row_err"] < arms[0]["row_err"]).mean())
    _title(ax, "(b) Where it helps", f"all refinements accepted; {improved:.0%} of rows improve")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def render(
    arms: list[dict[str, Any]],
    caps: list[dict[str, float]],
    deciles: list[dict[str, float]],
    phys: dict[str, float],
    meta: dict[str, Any],
) -> list[str]:
    lines = [
        "# 역산 refinement — 추론 후 물리 보정 (사전등록 2)",
        "",
        "`scripts/refine_inversion.py` 산출 — 재실행 시 덮어쓴다. 해석은 리포트 본문에서 한다.",
        "",
        f"- 대상 run `{meta['run']}` (holdout MAE {meta['recorded_mae']:.4f} nm), "
        f"평가 {meta['rows']:,}행 / holdout {meta['holdout_rows']:,}행",
        f"- 디코더 `{meta['decoder']}` (Stage A 확정, 동결) · complex128",
        f"- LM {meta['iters']}회 · 중앙차분 {meta['step_nm']} nm · 상자 {meta['box']} nm · "
        f"감쇠 행별 · 청크 {meta['chunk']:,}",
        "- **후처리이므로 기준선과 별도 행이다.** 격자 스냅은 쓰지 않는다 (평가 규약).",
        "",
        "## 1. 정확도 — 네 팔은 출발점만 다르다",
        "",
        "`truth+LM`만 라벨을 쓴다 (상한). 나머지 셋은 라벨 없이 계산되므로 test에도 적용된다.",
        "",
        "| 팔 | 출발점 | MAE [nm] | " + " | ".join(LAYER_COLS) + " | 중앙값 | p99 | 최대 | "
        f"분지 실패(>{BRANCH_FAIL_NM:.0f} nm) | 범위 밖 |",
        "|---|---|---|" + "---|" * (len(LAYER_COLS) + 5),
    ]
    starts = {
        "cnn": "—",
        "cnn+LM": "d_hat",
        "center+LM": f"{CENTER_NM:.0f} nm 균일",
        "truth+LM": "d_true (라벨)",
    }
    for arm in arms:
        per_layer = " | ".join(f"{arm[c]:.3f}" for c in LAYER_COLS)
        lines.append(
            f"| `{arm['name']}` | {starts[arm['name']]} | **{arm['overall']:.4f}** | {per_layer} |"
            f" {arm['median']:.3f} | {arm['p99']:.3f} | {arm['max']:.2f} |"
            f" {arm['branch_fail']:.2%} | {arm['out_of_physical']:.2%} |"
        )

    base, refined = arms[0], arms[1]
    delta = refined["row_err"] - base["row_err"]
    lines += [
        "",
        "## 2. 행별 효과 — cnn → cnn+LM",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 좋아진 행 | {(delta < 0).mean():.2%} |",
        f"| 나빠진 행 | {(delta > 0).mean():.2%} |",
        f"| 좋아진 행의 평균 개선 | {-delta[delta < 0].mean():.3f} nm |",
        f"| 나빠진 행의 평균 악화 | {delta[delta > 0].mean():.3f} nm |",
        f"| 순 효과 (MAE 감소) | {-delta.mean():.4f} nm |",
        "",
        "### 이동 상한 τ — 많이 움직인 행은 보정을 버리는 규칙 (라벨 불필요)",
        "",
        "**τ는 holdout으로 고르면 안 된다** (평가셋 선택). 판정은 사전등록 원안 τ = ∞로 한다.",
        "",
        "| τ [nm] | 보정 수용률 | MAE [nm] |",
        "|---|---|---|",
    ]
    for cap in caps:
        label = "∞ (원안)" if cap["tau"] == float("inf") else f"{cap['tau']:.0f}"
        lines.append(f"| {label} | {cap['kept']:.2%} | {cap['mae']:.4f} |")

    lines += [
        "",
        "## 3. 어디가 좋아지는가 — CNN 오차 십분위별 (라벨을 쓰는 진단)",
        "",
        "| 십분위 | " + " | ".join(str(d["decile"]) for d in deciles) + " |",
        "|---|" + "---|" * len(deciles),
        "| cnn [nm] | " + " | ".join(f"{d['base']:.3f}" for d in deciles) + " |",
        "| cnn+LM [nm] | " + " | ".join(f"{d['refined']:.3f}" for d in deciles) + " |",
        "| 좋아진 행 | " + " | ".join(f"{d['improved']:.0%}" for d in deciles) + " |",
        "",
        "## 4. 물리 잔차 — 라벨 없는 지표는 무엇을 보는가",
        "",
        "행별 재구성 L1 `|R_dec(d) − R_obs|`의 중앙값. LM이 직접 내리는 양이므로 **이것이",
        "내려간 것은 방법이 통했다는 증거가 아니다** — 노이즈만 남았을 때의 바닥이",
        f"{NOISE_MEAN_ABS}이고, 참 두께에서의 값이 디코더 계통오차까지 포함한 실질 바닥이다.",
        "",
        "| 팔 | 잔차 중앙값 |",
        "|---|---|",
    ]
    for name, value in phys.items():
        lines.append(f"| `{name}` | {value:.6f} |")
    lines += ["", f"그림: `figures/{meta['figure']}`", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="역산 refinement — 추론 후 물리 보정")
    parser.add_argument("--run", default="runs/level1_cnn/flatten-dilated-bound")
    parser.add_argument("--decoder", default=DEFAULT_DECODER, help="동결 디코더 체크포인트")
    parser.add_argument("--rows", type=int, default=None, help="holdout에서 무작위 N행만")
    parser.add_argument("--iters", type=int, default=30, help="LM 반복 수")
    parser.add_argument("--chunk", type=int, default=4096, help="LM 배치 크기")
    parser.add_argument("--out", default=None, help=f"리포트 경로 (기본 {OUT_PATH})")
    args = parser.parse_args()

    run_dir = Path(args.run)
    model, metrics = load_run(run_dir)
    x_all, y_all, _, holdout_idx = prepare_from_config(metrics["config"])
    idx = subsample_indices(len(holdout_idx), args.rows, seed=SUBSAMPLE_SEED)
    x = x_all[holdout_idx][idx]
    y = y_all[holdout_idx][idx].astype(np.float64)
    del x_all, y_all

    d_hat = predict(model, x).astype(np.float64)
    decoder = FrozenDecoder(args.decoder, dtype=torch.complex128)
    print(
        f"{run_dir.name}: 기록 {metrics['model']['val_mae']:.4f} / "
        f"재추론 {mae_per_layer(d_hat, y)['overall']:.4f} nm — {len(x):,}행"
    )

    starts = {
        "cnn+LM": d_hat,
        "center+LM": np.full_like(d_hat, CENTER_NM),
        "truth+LM": y,
    }
    solved: dict[str, np.ndarray] = {}
    for name, d_init in starts.items():
        t0 = time.perf_counter()
        solved[name] = lm_invert(
            decoder, x, d_init, iters=args.iters, damping="row", chunk=args.chunk
        )
        print(
            f"  {name:10s} MAE {mae_per_layer(solved[name], y)['overall']:.4f} nm"
            f"  ({time.perf_counter() - t0:.0f}s)"
        )

    arms = [arm_stats("cnn", d_hat, y)] + [arm_stats(n, d, y) for n, d in solved.items()]
    caps = move_cap_curve(d_hat, solved["cnn+LM"], y)
    deciles = decile_table(arms[0]["row_err"], arms[1]["row_err"])
    phys = {
        name: float(np.median(residual_l1_rows(decoder, d, x)))
        for name, d in [("cnn", d_hat), *solved.items()]
    }

    out_path = Path(args.out) if args.out else OUT_PATH
    fig_path = figure_path_for(out_path)
    figure(arms, deciles, fig_path)
    meta = {
        "run": args.run,
        "recorded_mae": metrics["model"]["val_mae"],
        "rows": len(x),
        "holdout_rows": len(holdout_idx),
        "decoder": decoder.provenance["decoder"],
        "iters": args.iters,
        "step_nm": 1e-3,
        "box": list(DEFAULT_BOX_NM),
        "chunk": args.chunk,
        "figure": fig_path.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(render(arms, caps, deciles, phys, meta)) + "\n")
    print(f"\n산출물: {out_path}\n         {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
