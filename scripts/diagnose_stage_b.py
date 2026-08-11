"""Stage B 진단 — β ablation 결과의 재현 가능한 분석 (Task 7).

리포트(reports/stage_b.md)의 모든 수치·그림은 이 스크립트 산출이다
(08-11 리뷰 규율: 진단을 scratchpad가 아니라 scripts/에 보존).

분석 4종:
  1. run 요약 + 무결성: 각 model.pt를 holdout에 재추론해 기록된 val MAE 재현 확인.
  2. 신뢰도 지표: 행별 물리 잔차 |R_dec(d_hat) − R_obs| ↔ 행별 실제 오차 |d_hat − d|의
     상관 + 십분위 캘리브레이션 (라벨 없이 오차를 추정할 수 있는가 — test 시 신뢰도).
  3. 물리 타깃 품질(역산 실험): 디코더만으로 d를 직접 최적화(행별 Adam)했을 때
     참값에서 얼마나 벗어나는가 — β>0 열화가 "타깃 편향"인지 "최적화 간섭"인지 판별.
     beta0 예측을 초기값으로 한 역산 = 물리 후처리 refinement의 성능 상한 측정.
  4. 두께 구간별 오차: 층별 MAE를 참 두께(30격자)별로 — layer_4 40~60 nm 민감도
     저하 구간 확인(백로그 열린 항목) + β 열화가 어느 구간에 몰리는지.

산출물:
  reports/stage_b_metrics.md
  reports/figures/fig_stage_b_learning_curves.png
  reports/figures/fig_stage_b_confidence.png
  reports/figures/fig_stage_b_inversion.png
  reports/figures/fig_stage_b_thickness_bins.png

사용: python -m scripts.diagnose_stage_b  (CPU, 수 분)
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.dataset import REPO_ROOT, prepare_train_arrays
from src.evaluate import load_model_checkpoint, mae_per_layer, predict
from src.physics.decoder import TMMDecoder, load_tmm_decoder

RUNS = ["beta0", "beta30", "beta100", "beta300"]
BETAS = {"beta0": 0.0, "beta30": 30.0, "beta100": 100.0, "beta300": 300.0}
STAGE_B_DIR = REPO_ROOT / "runs" / "stage_b"
FIG_DIR = REPO_ROOT / "reports" / "figures"
OUT_MD = REPO_ROOT / "reports" / "stage_b_metrics.md"

SEED = 42
INVERSION_ROWS = 2048  # truth-init 진단용 서브셋
INVERSION_STEPS = 800
INVERSION_LR = 0.5
REFINE_STEPS = 400  # holdout 전체 refinement (수렴은 ~200 step — 로그 확인)
REFINE_CHUNK = 8192

# β 순서(크기가 있는 양) — 단일 색상 sequential ramp, 밝음(0) → 어두움(300)
RAMP = {"beta0": "#9dc1ef", "beta30": "#5691dd", "beta100": "#2f6bbf", "beta300": "#173f78"}
# 2종 비교(beta0 vs beta300) 고정 색 — CVD-safe 파랑/주황
C_BETA0, C_BETA300 = "#2f6bbf", "#d97706"


def _style(ax: plt.Axes) -> None:
    """recessive 그리드·스파인 — 데이터 잉크가 앞에 서게."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def per_row_residual(decoder: TMMDecoder, d_hat: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
    """행별 물리 잔차 mean_ch |R_dec(d_hat) − R_obs|. 반환 (N,) float64."""
    out = np.empty(len(d_hat), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(d_hat), 8192):
            r = decoder(torch.from_numpy(d_hat[s : s + 8192]))
            out[s : s + 8192] = (r - torch.from_numpy(x_obs[s : s + 8192])).abs().mean(1).numpy()
    return out


def invert_physics(
    decoder: TMMDecoder,
    d_init: np.ndarray,
    x_obs: np.ndarray,
    tag: str,
    steps: int = INVERSION_STEPS,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """디코더만으로 행별 d를 L1(R_dec(d), R_obs)에 최적화한다 (라벨 미사용).

    Args:
        d_init: (N, 4) 초기값 [nm]. x_obs: (N, 226) 관측 반사율.

    Returns:
        (d_opt (N, 4), 최종 mean 잔차).
    """
    d = torch.from_numpy(d_init.astype(np.float32).copy()).requires_grad_()
    x_t = torch.from_numpy(x_obs)
    opt = torch.optim.Adam([d], lr=INVERSION_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        loss = (decoder(d) - x_t).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 200 == 0 or step == steps - 1):
            print(f"  [invert:{tag}] step {step:4d}  residual {loss.item():.5f}")
    with torch.no_grad():
        d_opt = d.clamp(10.0, 300.0)  # 물리 범위 (격자 밖 이탈은 관찰상 없음 — 안전장치)
        final = float((decoder(d_opt) - x_t).abs().mean())
    return d_opt.numpy(), final


def refine_holdout(
    decoder: TMMDecoder, d_init: np.ndarray, x_obs: np.ndarray, steps: int = REFINE_STEPS
) -> tuple[np.ndarray, float]:
    """holdout 전체를 청크 단위로 역산 refinement — 행별 독립 문제라 분할은 결과 불변.

    Returns:
        (d_opt (N, 4), 전체 mean 잔차).
    """
    out = np.empty_like(d_init)
    res_sum = 0.0
    for s in range(0, len(d_init), REFINE_CHUNK):
        d_opt, res = invert_physics(
            decoder,
            d_init[s : s + REFINE_CHUNK],
            x_obs[s : s + REFINE_CHUNK],
            tag=f"refine:{s // REFINE_CHUNK}",
            steps=steps,
            verbose=False,
        )
        out[s : s + REFINE_CHUNK] = d_opt
        res_sum += res * len(d_opt)
        done = min(s + REFINE_CHUNK, len(d_init))
        print(f"  [refine] {done:,}/{len(d_init):,}행 (잔차 {res:.5f})")
    return out, res_sum / len(d_init)


def parse_train_log(run: str) -> tuple[np.ndarray, np.ndarray]:
    """train.log에서 (epoch, val_mae) 곡선을 뽑는다."""
    pat = re.compile(r"epoch\s+(\d+)/\d+\s+train_l1\s+[\d.]+\s+val_mae\s+([\d.]+)")
    ep, mae = [], []
    for line in (STAGE_B_DIR / run / "train.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            ep.append(int(m.group(1)))
            mae.append(float(m.group(2)))
    return np.asarray(ep), np.asarray(mae)


def main() -> None:
    torch.manual_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Stage B 진단 수치 (scripts/diagnose_stage_b.py 산출)",
        "",
        f"holdout 81,000행 (val_frac 0.1, seed {SEED}) / 디코더 "
        "runs/stage_a/sio2-freeze-refine (complex64). 해석은 reports/stage_b.md.",
        "",
    ]

    x, y, _, holdout_idx = prepare_train_arrays(val_frac=0.1, seed=SEED)
    x_v, y_v = x[holdout_idx], y[holdout_idx]
    del x, y
    decoder, dec_meta = load_tmm_decoder(REPO_ROOT / "runs" / "stage_a" / "sio2-freeze-refine")
    print(f"디코더: {dec_meta} / holdout {len(x_v):,}행")

    # ---- 1. run 요약 + 무결성 (기록 val MAE 재현) --------------------------------
    preds: dict[str, np.ndarray] = {}
    recorded: dict[str, dict] = {}
    lines += [
        "## 1. run 요약 + 무결성",
        "",
        "| run | β | holdout MAE [nm] | 재추론 MAE | L1 | L2 | L3 | L4 | val_phys | best ep |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in RUNS:
        m = json.loads((STAGE_B_DIR / run / "metrics.json").read_text())["model"]
        recorded[run] = m
        model = load_model_checkpoint(STAGE_B_DIR / run / "model.pt")
        pred = predict(model, x_v)
        preds[run] = pred
        again = mae_per_layer(pred, y_v)
        drift = abs(again["overall"] - m["val_mae"])
        assert drift < 1e-3, f"{run}: 재추론 {again['overall']:.4f} vs 기록 {m['val_mae']:.4f}"
        pl = m["val_mae_per_layer"]
        phys = m.get("val_phys_l1")
        lines.append(
            f"| {run} | {BETAS[run]:g} | {m['val_mae']:.4f} | {again['overall']:.4f} |"
            + "".join(f" {pl[f'layer_{i}']:.3f} |" for i in range(1, 5))
            + (f" {phys:.4f} |" if phys is not None else " — |")
            + f" {m['best_epoch']} |"
        )
        print(f"{run}: 기록 {m['val_mae']:.4f} 재현 OK (드리프트 {drift:.2e})")
    lines.append("")

    # 물리 잔차의 바닥: 참값 d에서의 잔차 (노이즈 + 디코더 오차만 남는 수준)
    res_true = per_row_residual(decoder, y_v, x_v)
    lines += [
        f"- 물리 잔차 바닥 mean|R_dec(d_true) − R_obs| = **{res_true.mean():.5f}**"
        " (균등 노이즈 ±0.015의 E|ε| = 0.0075 + 디코더 오차)",
        "",
    ]

    # ---- 학습곡선 그림 -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for run in RUNS:
        ep, mae = parse_train_log(run)
        ax.plot(ep, mae, color=RAMP[run], linewidth=1.8, label=f"β = {BETAS[run]:g}")
        # β=0과 β=30은 종점이 거의 겹친다 — 직접 라벨을 위아래로 벌린다
        dy = {"beta0": -7, "beta30": 7}.get(run, 0)
        ax.annotate(
            f"β={BETAS[run]:g}",
            (ep[-1], mae[-1]),
            xytext=(6, dy),
            textcoords="offset points",
            color=RAMP[run],
            fontsize=9,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("holdout MAE [nm] (log)")
    ax.set_title("Stage B learning curves by physics-loss weight β")
    ax.set_xlim(1, 33.5)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_stage_b_learning_curves.png", dpi=150)
    plt.close(fig)

    # ---- 2. 신뢰도 지표: 행별 잔차 ↔ 행별 오차 ----------------------------------
    lines += [
        "## 2. 신뢰도 지표 — 행별 물리 잔차 ↔ 행별 실제 오차",
        "",
        "행별 잔차 r_i = mean_ch |R_dec(d_hat_i) − R_obs_i|,"
        " 행별 오차 e_i = mean_4 |d_hat_i − d_i|.",
        "",
        "| run | Pearson r | Spearman ρ |",
        "|---|---|---|",
    ]
    residuals: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    for run in RUNS:
        e = np.abs(preds[run] - y_v).mean(1)
        r = per_row_residual(decoder, preds[run], x_v)
        residuals[run], errors[run] = r, e
        lines.append(f"| {run} | {pearsonr(r, e)[0]:.3f} | {spearmanr(r, e)[0]:.3f} |")
    lines.append("")

    # beta0 십분위 캘리브레이션 (배포 시 나올 모델 = 대조군에 대해 보고)
    r0, e0 = residuals["beta0"], errors["beta0"]
    deciles = np.quantile(r0, np.linspace(0, 1, 11))
    lines += [
        "beta0의 잔차 십분위 캘리브레이션 (잔차로 오차를 얼마나 선별하나):",
        "",
        "| 잔차 십분위 | 잔차 구간 | mean 오차 [nm] | median 오차 [nm] |",
        "|---|---|---|---|",
    ]
    decile_mean = []
    for i in range(10):
        sel = (r0 >= deciles[i]) & (r0 <= deciles[i + 1] if i == 9 else r0 < deciles[i + 1])
        decile_mean.append(e0[sel].mean())
        lines.append(
            f"| D{i + 1} | {deciles[i]:.4f}–{deciles[i + 1]:.4f} |"
            f" {e0[sel].mean():.3f} | {np.median(e0[sel]):.3f} |"
        )
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    hb = axes[0].hexbin(r0, e0, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=axes[0], label="rows (log)")
    axes[0].set_xlabel("per-row physics residual mean|R_dec(d_hat) - R_obs|")
    axes[0].set_ylabel("per-row error mean|d_hat - d| [nm]")
    axes[0].set_title(f"beta0 — Spearman ρ = {spearmanr(r0, e0)[0]:.3f}")
    axes[1].plot(range(1, 11), decile_mean, marker="o", markersize=6, color=C_BETA0, linewidth=1.8)
    axes[1].set_xlabel("physics-residual decile (D1 low → D10 high)")
    axes[1].set_ylabel("mean error [nm]")
    axes[1].set_title("beta0 — actual error by residual decile")
    axes[1].set_xticks(range(1, 11))
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_stage_b_confidence.png", dpi=150)
    plt.close(fig)

    # ---- 3. 물리 역산 — (a) 타깃 품질 진단, (b) holdout 전체 후처리 refinement ----
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(x_v), size=INVERSION_ROWS, replace=False)
    print(f"\n역산 (a) truth-init 진단: {INVERSION_ROWS}행, {INVERSION_STEPS} steps")
    d_truth_opt, res_truth = invert_physics(decoder, y_v[sub], x_v[sub], "truth-init")
    truth_mae = float(np.abs(d_truth_opt - y_v[sub]).mean())
    truth_pl = np.abs(d_truth_opt - y_v[sub]).mean(0)

    print(f"\n역산 (b) beta0 → refinement: holdout 전체 {len(x_v):,}행, {REFINE_STEPS} steps")
    refined, res_refined = refine_holdout(decoder, preds["beta0"], x_v)
    ref_metrics = mae_per_layer(refined, y_v)
    err_init = np.abs(preds["beta0"] - y_v).mean(1)
    err_ref = np.abs(refined - y_v).mean(1)
    pct_improved = float((err_ref < err_init).mean())
    pct_gt5 = float((err_ref > 5.0).mean())

    lines += [
        "## 3. 물리 역산 — 디코더 단독 최적화 (라벨 미사용)",
        "",
        "행별로 d를 L1(R_dec(d), R_obs)에 Adam 최적화. (a) 참값 초기화 = 물리 손실"
        " 최적해가 참값에서 얼마나 벗어나는가(β→∞가 당기는 타깃의 품질, "
        f"{INVERSION_ROWS}행 표본). (b) beta0 예측 초기화 = 추론 시 물리 후처리"
        " refinement (holdout **전체**).",
        "",
        "| 대상 | 행 수 | init MAE [nm] | 역산 후 MAE [nm] | 역산 후 잔차 | 층별 MAE |",
        "|---|---|---|---|---|---|",
        f"| (a) truth-init | {INVERSION_ROWS:,} | 0 | **{truth_mae:.4f}** |"
        f" {res_truth:.5f} | " + "  ".join(f"{v:.3f}" for v in truth_pl) + " |",
        f"| (b) beta0 + refinement | {len(x_v):,} | {recorded['beta0']['val_mae']:.4f} |"
        f" **{ref_metrics['overall']:.4f}** | {res_refined:.5f} | "
        + "  ".join(f"{ref_metrics[f'layer_{i}']:.3f}" for i in range(1, 5))
        + " |",
        "",
        f"- (b) 행별 오차 분포: median {float(np.median(err_ref)):.3f} nm,"
        f" P90 {float(np.quantile(err_ref, 0.9)):.3f} nm, 개선된 행 {pct_improved:.1%},"
        f" 5 nm 초과 잔류(오분지 의심) {pct_gt5:.2%}.",
        f"- 참값에서의 잔차 바닥(§1): {res_true.mean():.5f} — 역산 후 잔차가 이보다 아래면"
        " 노이즈를 d로 흡수하고 있다는 신호 (truth-init {:.5f}는 근소 하회 = 예상 규모).".format(
            res_truth
        ),
        "",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [
        "beta0 CNN",
        "beta0 + TMM refinement",
        f"physics optimum\n(truth-init, n={INVERSION_ROWS})",
    ]
    vals = [recorded["beta0"]["val_mae"], ref_metrics["overall"], truth_mae]
    bars = axes[0].barh(
        labels[::-1], vals[::-1], color=[C_BETA0, C_BETA300, "#9aa3ad"][::-1], height=0.55
    )
    for b, v in zip(bars, vals[::-1], strict=True):
        axes[0].annotate(
            f"{v:.3f} nm",
            (b.get_width(), b.get_y() + b.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    axes[0].axvline(0.3955, color="#555555", linewidth=1.2, linestyle="--")
    axes[0].annotate(
        "213M strong baseline 0.3955",
        (0.3955, 2.35),
        fontsize=8.5,
        color="#555555",
        rotation=90,
        va="top",
        xytext=(4, 0),
        textcoords="offset points",
    )
    axes[0].set_xlabel("holdout MAE [nm]")
    axes[0].set_title("TMM inversion accuracy (decoder-only, no labels)")
    axes[0].set_xlim(0, max(vals) * 1.18)
    bins = np.logspace(np.log10(0.02), np.log10(50), 60)
    axes[1].hist(err_init, bins=bins, color=C_BETA0, alpha=0.7, label="beta0 CNN")
    axes[1].hist(err_ref, bins=bins, color=C_BETA300, alpha=0.7, label="+ TMM refinement")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("per-row error mean|d_hat - d| [nm] (log)")
    axes[1].set_ylabel("rows")
    axes[1].set_title("Per-row error, before vs after refinement")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_stage_b_inversion.png", dpi=150)
    plt.close(fig)

    # ---- 4. 두께 구간별 오차 (layer_4 40~60 nm 확인 + β 열화 위치) ---------------
    grid = np.arange(10, 301, 10)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    bin_tbl: dict[str, np.ndarray] = {}
    for run, color in [("beta0", C_BETA0), ("beta300", C_BETA300)]:
        err = np.abs(preds[run] - y_v)  # (N, 4)
        curves = np.zeros((4, len(grid)))
        for j in range(4):
            for gi, val in enumerate(grid):
                curves[j, gi] = err[y_v[:, j] == val, j].mean()
        bin_tbl[run] = curves
        for j, ax in enumerate(axes.flat):
            ax.plot(grid, curves[j], color=color, linewidth=1.8, label=f"{run}")
    for j, ax in enumerate(axes.flat):
        ax.set_title(f"layer_{j + 1}")
        if j == 3:
            ax.axvspan(40, 60, alpha=0.12, color="gray")  # EDA 민감도 저하 구간
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("true thickness [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAE [nm]")
    fig.suptitle(
        "Per-layer MAE vs true thickness — beta0 vs beta300"
        " (gray band: layer_4 low-sensitivity 40-60 nm)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_stage_b_thickness_bins.png", dpi=150)
    plt.close(fig)

    l4_dip = bin_tbl["beta0"][3, (grid >= 40) & (grid <= 60)].mean()
    l4_rest = bin_tbl["beta0"][3, (grid < 40) | (grid > 60)].mean()
    worsen = (bin_tbl["beta300"] - bin_tbl["beta0"]) / bin_tbl["beta0"]
    lines += [
        "## 4. 두께 구간별 오차",
        "",
        f"- beta0 layer_4: 민감도 저하 구간(40–60 nm) MAE **{l4_dip:.3f}** vs 그 외"
        f" **{l4_rest:.3f}** ({l4_dip / l4_rest:.2f}×) — EDA가 예고한 사각 구간 확인.",
        f"- beta300 상대 열화(구간 평균): 층별 최대 {worsen.max():.0%}"
        f" (layer_{int(np.unravel_index(worsen.argmax(), worsen.shape)[0]) + 1},"
        f" {grid[np.unravel_index(worsen.argmax(), worsen.shape)[1]]} nm 구간),"
        f" 전 구간 중앙값 {np.median(worsen):.0%}.",
        "",
        "곡선 전체는 fig_stage_b_thickness_bins.png.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"\n산출: {OUT_MD}")
    print(f"그림 4종: {FIG_DIR}/fig_stage_b_*.png")


if __name__ == "__main__":
    main()
