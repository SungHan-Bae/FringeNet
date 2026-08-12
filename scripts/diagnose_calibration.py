"""Stage A 게이트 진단 — (a) 재구성 RMSE + (c) 잔차 백색성 (CLAUDE.md Level 2 판정 게이트).

캘리브레이션 산출물(runs/stage_a/<run>/model.pt)을 읽어, 피팅에 쓰지 않은 진단
표본에서 잔차 eps = R_obs − R_TMM(d_true)의 크기·구조를 측정한다. 판정 기준:
  (a) RMSE < 1.2σ ≈ 0.0105 — 관측 R에 σ ≈ 0.0087 노이즈가 있어 완벽한 모델도
      σ 아래로 못 내려간다.
  (c) 잔차가 두께·채널에 대해 구조 없이 백색이고 크기가 σ와 일치 — (a)를 통과해도
      구조가 남으면 모델 오차로 보고 기각한다. 백색성의 수치 진단:
      채널 프로파일 평탄성, 두께 bin별 RMS 평탄성, 채널 방향 lag-1 자기상관
      (iid 노이즈면 ~0, 매끈한 모델 오차면 양수), RMSE/σ_hf 비(σ_hf = 2차 차분
      고주파 추정 — 매끈한 구조는 RMSE만 키우고 σ_hf는 못 키운다).

생성물:
  reports/figures/fig_stage_a1_dispersion.png   학습된 λ 그리드·n(λ) vs 문헌 (육안 확인 기록)
  reports/figures/fig_stage_a2_residuals.png    잔차 백색성 진단 4패널
  reports/stage_a_gate.md                       게이트 수치 표 (재실행 시 덮어씀)

사용법:
    python scripts/diagnose_calibration.py --run runs/stage_a/sio2-freeze
    python scripts/diagnose_calibration.py --run runs/stage_a/sio2-freeze-adachi --tag adachi
    # --tag는 산출물 이름에 _<tag>를 붙인다 — 채택 디코더의 게이트 기록(무태그)을
    # 덮어쓰지 않고 변형 실험을 별도 문서로 남기기 위함.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda import (  # noqa: E402 — 팔레트·축 스타일 단일 출처 (scripts/eda.py, 검증된 팔레트)
    INK_MUTED,
    INK_SECONDARY,
    LAYER_COLORS,
    LAYER_LABELS,
    SURFACE,
    _style_axes,
    _title,
)

from src.calibrate import GATE_A_RMSE, load_calibrated_stack  # noqa: E402
from src.data.dataset import REPO_ROOT, prepare_train_arrays  # noqa: E402
from src.physics.dispersion import si3n4_n, si_nk, sio2_n  # noqa: E402

FIG_DIR = REPO_ROOT / "reports" / "figures"
GATE_PATH = REPO_ROOT / "reports" / "stage_a_gate.md"

NOISE_SIGMA = 0.0087  # 데이터 노이즈 바닥 (CLAUDE.md 데이터 계약)

# 재료별 색 — 층 팔레트에서 재사용 (SiN=layer1 파랑, SiO₂=layer2 주황, Si=초록).
C_SIN, C_SIO2, C_SI = LAYER_COLORS[0], LAYER_COLORS[1], LAYER_COLORS[2]


def load_diag_split(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """metrics.json의 설정으로 캘리브레이션과 동일한 진단 표본을 재현한다.

    Returns:
        (x_diag (M, 226) float32, d_diag (M, 4) float32, metrics dict).
    """
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["config"]
    seed = int(cfg["seed"])
    n_fit, n_diag = int(cfg["data"]["fit_rows"]), int(cfg["data"]["diag_rows"])
    x, y, train_idx, _ = prepare_train_arrays(
        val_frac=float(cfg["data"].get("val_frac", 0.1)), seed=seed
    )
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=n_fit + n_diag, replace=False)
    return x[pick[n_fit:]], y[pick[n_fit:]], metrics


@torch.no_grad()
def compute_residuals(model: torch.nn.Module, x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """eps = R_obs − R_TMM(d_true). 반환 (M, W) float64."""
    d_t = torch.from_numpy(d).to(torch.float64)
    out = np.empty(x.shape, dtype=np.float64)
    for start in range(0, len(x), 4096):
        pred = model(d_t[start : start + 4096]).numpy()
        out[start : start + 4096] = x[start : start + 4096].astype(np.float64) - pred
    return out


def whiteness_metrics(eps: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    """게이트 (c)용 잔차 구조 수치를 계산한다.

    Args:
        eps: (M, W) 잔차. d: (M, 4) 두께 [nm].

    Returns:
        rmse/bias/채널 프로파일/두께 bin RMS/lag-1 자기상관/고주파 σ 추정 등.
    """
    rmse = float(np.sqrt((eps**2).mean()))
    bias = float(eps.mean())
    ch_rmse = np.sqrt((eps**2).mean(axis=0))  # (W,)
    ch_mean = eps.mean(axis=0)

    # 두께 bin별 RMS — 층마다 30개 격자값으로 묶는다. 백색이면 평평해야 한다.
    bins: dict[str, dict[str, list[float]]] = {}
    for j in range(4):
        values = np.unique(d[:, j])
        rms = [float(np.sqrt((eps[d[:, j] == v] ** 2).mean())) for v in values]
        bins[f"layer_{j + 1}"] = {"thickness": [float(v) for v in values], "rms": rms}

    # 채널 방향 lag-1 자기상관 — 채널별 평균(프로파일)을 뺀 뒤 계산해 프로파일
    # 구조와 행 내부의 매끈한 오차를 분리한다. iid 노이즈면 ~0.
    centered = eps - ch_mean
    rho1 = float((centered[:, 1:] * centered[:, :-1]).mean() / (centered**2).mean())

    # 고주파 σ (2차 차분): Var(eps[i-1] − 2eps[i] + eps[i+1]) = 6σ² (iid 성분만 잡는다).
    d2 = eps[:, :-2] - 2.0 * eps[:, 1:-1] + eps[:, 2:]
    sigma_hf = float(np.sqrt((d2**2).mean() / 6.0))

    return {
        "rmse": rmse,
        "bias": bias,
        "channel_rmse": ch_rmse,
        "channel_mean": ch_mean,
        "channel_rmse_ratio": float(ch_rmse.max() / ch_rmse.min()),
        "thickness_bins": bins,
        "bin_rms_ratio": {k: float(max(v["rms"]) / min(v["rms"])) for k, v in bins.items()},
        "rho1": rho1,
        "sigma_hf": sigma_hf,
        "rmse_over_sigma_hf": rmse / sigma_hf,
        "rmse_over_sigma": rmse / NOISE_SIGMA,
    }


def figure_dispersion(model: torch.nn.Module, run_name: str, out_path: Path) -> dict[str, Any]:
    """학습된 λ 그리드와 n(λ) 곡선을 문헌값과 겹쳐 그린다 (육안 확인 기록 — CLAUDE.md)."""
    with torch.no_grad():
        lam_t, n_layers, ns = model.spectra()
    lam = lam_t.numpy()
    n_sin_l = n_layers[0].real.numpy()
    n_sio2_l = n_layers[1].real.numpy()
    n_si_l, k_si_l = ns.real.numpy(), (-ns.imag).numpy()
    ch = np.arange(len(lam))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.42, wspace=0.25)

    ax = axes[0, 0]
    _style_axes(ax)
    init = np.linspace(*model.lam_init, len(lam))
    if model.descending:
        init = init[::-1]
    ax.plot(ch, init, color=INK_MUTED, lw=1.4, ls="--")
    ax.plot(ch, lam, color=C_SIN, lw=2)
    ax.annotate(
        "calibrated λ",
        (ch[30], lam[30]),
        xytext=(0, 10),
        textcoords="offset points",
        color=C_SIN,
        fontsize=9,
    )
    ax.annotate(
        "initial grid",
        (ch[-60], init[-60]),
        xytext=(0, -16),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlabel("wavelength channel index (de-identified)")
    ax.set_ylabel("λ [nm]")
    _title(
        ax,
        "Calibrated λ grid",
        f"{'descending' if model.descending else 'ascending'}, {lam.min():.0f}–{lam.max():.0f} nm",
    )

    ax = axes[0, 1]
    _style_axes(ax)
    order = np.argsort(lam)
    ax.plot(lam[order], si3n4_n(lam)[order], color=C_SIN, lw=1.4, ls="--", alpha=0.6)
    ax.plot(lam[order], n_sin_l[order], color=C_SIN, lw=2)
    ax.plot(lam[order], sio2_n(lam)[order], color=C_SIO2, lw=1.4, ls="--", alpha=0.6)
    ax.plot(lam[order], n_sio2_l[order], color=C_SIO2, lw=2)
    mid = order[len(order) // 2]
    ax.annotate(
        "SiN calibrated (dashed: Luke 2015)",
        (lam[mid], n_sin_l[mid]),
        xytext=(0, 12),
        textcoords="offset points",
        color=C_SIN,
        fontsize=9,
    )
    ax.annotate(
        "SiO2 frozen (gauge)",
        (lam[mid], n_sio2_l[mid]),
        xytext=(0, 12),
        textcoords="offset points",
        color=C_SIO2,
        fontsize=9,
    )
    ax.set_xlabel("λ [nm]")
    ax.set_ylabel("n")
    _title(ax, "Layer refractive index n(λ)", "solid = calibrated, dashed = literature")

    ax = axes[1, 0]
    _style_axes(ax)
    ax.plot(lam[order], si_nk(lam)[0][order], color=C_SI, lw=1.4, ls="--", alpha=0.6)
    ax.plot(lam[order], n_si_l[order], color=C_SI, lw=2)
    ax.annotate(
        "Si n calibrated (dashed: A&S 1983)",
        (lam[mid], n_si_l[mid]),
        xytext=(0, 12),
        textcoords="offset points",
        color=C_SI,
        fontsize=9,
    )
    ax.set_xlabel("λ [nm]")
    ax.set_ylabel("n")
    _title(ax, "Si substrate n(λ)", "piecewise-linear knot interpolation")

    ax = axes[1, 1]
    _style_axes(ax)
    ax.plot(
        lam[order],
        np.clip(si_nk(lam)[1], 1e-5, None)[order],
        color=C_SI,
        lw=1.4,
        ls="--",
        alpha=0.6,
    )
    ax.plot(lam[order], np.clip(k_si_l, 1e-5, None)[order], color=C_SI, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("λ [nm]")
    ax.set_ylabel("k (log)")
    _title(ax, "Si substrate k(λ)", "solid = calibrated (softplus ≥ 0), dashed = literature")

    fig.suptitle(f"Stage A dispersion curves — {run_name}", y=0.99, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return {
        "lam_min": float(lam.min()),
        "lam_max": float(lam.max()),
        "descending": bool(model.descending),
        "n_sin_range": [float(n_sin_l.min()), float(n_sin_l.max())],
        "n_sio2_range": [float(n_sio2_l.min()), float(n_sio2_l.max())],
        "n_si_range": [float(n_si_l.min()), float(n_si_l.max())],
        "k_si_range": [float(k_si_l.min()), float(k_si_l.max())],
    }


def figure_residuals(
    eps: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    model: torch.nn.Module,
    wm: dict[str, Any],
    run_name: str,
    out_path: Path,
) -> None:
    """잔차 백색성 4패널 — 채널 프로파일 / 두께 bin / 분포 / 예시 행 재구성."""
    w = eps.shape[1]
    ch = np.arange(w)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.42, wspace=0.25)

    ax = axes[0, 0]
    _style_axes(ax)
    ax.axhline(NOISE_SIGMA, color=INK_MUTED, lw=1.2, ls="--")
    ax.annotate(
        f"noise σ = {NOISE_SIGMA}",
        (w * 0.02, NOISE_SIGMA),
        xytext=(0, 6),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.plot(ch, wm["channel_rmse"], color=C_SIN, lw=2)
    ax.plot(ch, wm["channel_mean"], color=C_SIO2, lw=1.6)
    ax.annotate(
        "per-channel RMSE",
        (ch[20], wm["channel_rmse"][20]),
        xytext=(0, 10),
        textcoords="offset points",
        color=C_SIN,
        fontsize=9,
    )
    ax.annotate(
        "per-channel mean (bias)",
        (ch[20], wm["channel_mean"][20]),
        xytext=(0, -14),
        textcoords="offset points",
        color=C_SIO2,
        fontsize=9,
    )
    ax.axhline(0.0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("wavelength channel index (de-identified)")
    ax.set_ylabel("residual [R]")
    _title(ax, "Channel profile", "white residual: RMSE flat at σ, mean at 0")

    ax = axes[0, 1]
    _style_axes(ax)
    ax.axhline(NOISE_SIGMA, color=INK_MUTED, lw=1.2, ls="--")
    for j in range(4):
        b = wm["thickness_bins"][f"layer_{j + 1}"]
        ax.plot(b["thickness"], b["rms"], color=LAYER_COLORS[j], lw=2, alpha=0.9)
        ax.annotate(
            LAYER_LABELS[j],
            (b["thickness"][-1], b["rms"][-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=LAYER_COLORS[j],
            fontsize=8,
            va="center",
        )
    ax.set_xlim(0, 385)
    ax.set_xlabel("thickness (nm)")
    ax.set_ylabel("residual RMS [R]")
    _title(ax, "Residual RMS by thickness bin", "white residual: flat across layers and thickness")

    ax = axes[1, 0]
    _style_axes(ax)
    lim = max(0.02, float(np.percentile(np.abs(eps), 99.9)))
    grid = np.linspace(-lim, lim, 400)
    sample = eps.ravel()[:: max(1, eps.size // 200_000)]
    ax.hist(sample, bins=120, range=(-lim, lim), density=True, color=C_SIN, alpha=0.55)
    a_unif = NOISE_SIGMA * np.sqrt(3.0)
    ax.plot(
        grid,
        np.where(np.abs(grid) <= a_unif, 1.0 / (2 * a_unif), 0.0),
        color=INK_SECONDARY,
        lw=1.6,
        ls="--",
    )
    ax.plot(
        grid,
        np.exp(-0.5 * (grid / NOISE_SIGMA) ** 2) / (NOISE_SIGMA * np.sqrt(2 * np.pi)),
        color=INK_MUTED,
        lw=1.4,
        ls=":",
    )
    ax.annotate(
        "uniform ±σ√3",
        (a_unif * 0.55, 1.0 / (2 * a_unif)),
        xytext=(0, 8),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.annotate(
        "gaussian σ",
        (-NOISE_SIGMA * 1.9, 0.6 / (2 * a_unif)),
        xytext=(0, 8),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=9,
    )
    ax.set_xlabel("residual [R]")
    ax.set_ylabel("density")
    _title(ax, "Residual distribution", "should match data noise (near-uniform, σ ≈ 0.0087)")

    ax = axes[1, 1]
    _style_axes(ax)
    row_rmse = np.sqrt((eps**2).mean(axis=1))
    idx_med = int(np.argsort(row_rmse)[len(row_rmse) // 2])
    idx_worst = int(np.argmax(row_rmse))
    with torch.no_grad():
        pred = model(torch.from_numpy(d[[idx_med, idx_worst]]).to(torch.float64)).numpy()
    for k, (idx, offset, label) in enumerate(
        [(idx_med, 0.0, "median row"), (idx_worst, 0.55, "worst row")]
    ):
        ax.plot(ch, x[idx].astype(np.float64) + offset, color=INK_MUTED, lw=1.0)
        ax.plot(ch, pred[k] + offset, color=LAYER_COLORS[k], lw=1.6, alpha=0.9)
        ax.annotate(
            f"{label} (RMSE {row_rmse[idx]:.4f})",
            (ch[4], pred[k][4] + offset),
            xytext=(0, 12),
            textcoords="offset points",
            color=LAYER_COLORS[k],
            fontsize=9,
        )
    ax.set_xlabel("wavelength channel index (de-identified)")
    ax.set_ylabel("R (+offset)")
    _title(ax, "Reconstruction examples", "gray = R_obs, color = R_TMM(d_true); worst +0.55")

    fig.suptitle(f"Stage A residual whiteness — {run_name}", y=0.99, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def write_gate_report(
    run_name: str,
    ckpt: dict[str, Any],
    wm: dict[str, Any],
    disp: dict[str, Any],
    n_diag: int,
    gate_path: Path,
    suffix: str,
) -> None:
    """게이트 수치 표를 gate_path 에 쓴다 (스크립트 산출, 덮어씀)."""
    gate_a = wm["rmse"] < GATE_A_RMSE

    def mark(ok: bool) -> str:  # noqa: FBT001 — 표기 전용
        return "✓" if ok else "✗"

    c_checks = [
        ("|bias| < 0.001", abs(wm["bias"]) < 1e-3, f"{wm['bias']:+.5f}"),
        (
            "RMSE/σ ∈ [0.9, 1.2]",
            0.9 <= wm["rmse_over_sigma"] <= 1.2,
            f"{wm['rmse_over_sigma']:.3f}",
        ),
        (
            "채널 RMSE max/min < 1.3",
            wm["channel_rmse_ratio"] < 1.3,
            f"{wm['channel_rmse_ratio']:.3f}",
        ),
        (
            "두께 bin RMS max/min < 1.3 (전 층)",
            max(wm["bin_rms_ratio"].values()) < 1.3,
            " / ".join(f"{v:.3f}" for v in wm["bin_rms_ratio"].values()),
        ),
        ("|lag-1 자기상관| < 0.1", abs(wm["rho1"]) < 0.1, f"{wm['rho1']:+.4f}"),
        (
            "RMSE/σ_hf < 1.15",
            wm["rmse_over_sigma_hf"] < 1.15,
            f"{wm['rmse_over_sigma_hf']:.3f} (σ_hf {wm['sigma_hf']:.5f})",
        ),
    ]
    lines = [
        "# Stage A 게이트 진단 (`scripts/diagnose_calibration.py` 산출 — 손으로 고치지 말 것)",
        "",
        f"- run: `stage_a/{run_name}`, best step {ckpt['step']}, "
        f"진단 표본 {n_diag:,}행 (피팅과 분리)",
        f"- 학습된 λ: {disp['lam_min']:.1f}–{disp['lam_max']:.1f} nm, "
        f"{'내림' if disp['descending'] else '오름'}차순 / "
        f"n(SiN) {disp['n_sin_range'][0]:.3f}–{disp['n_sin_range'][1]:.3f}, "
        f"n(SiO₂ freeze) {disp['n_sio2_range'][0]:.3f}–{disp['n_sio2_range'][1]:.3f}, "
        f"n(Si) {disp['n_si_range'][0]:.2f}–{disp['n_si_range'][1]:.2f}, "
        f"k(Si) {disp['k_si_range'][0]:.4f}–{disp['k_si_range'][1]:.4f}",
        "",
        "## 게이트 (a) — 재구성 RMSE",
        "",
        "| 항목 | 값 | 기준 | 판정 |",
        "|---|---|---|---|",
        f"| 진단 표본 RMSE | {wm['rmse']:.5f} | < {GATE_A_RMSE} (= 1.2σ) | {mark(gate_a)} |",
        "",
        "## 게이트 (c) — 잔차 백색성 (수치 진단; 최종 판정은 그림 육안 확인과 함께)",
        "",
        "| 진단 | 값 | 판정 |",
        "|---|---|---|",
        *(f"| {name} | {val} | {mark(ok)} |" for name, ok, val in c_checks),
        "",
        f"그림: `figures/fig_stage_a1_dispersion{suffix}.png` (분산 곡선 육안 확인), "
        f"`figures/fig_stage_a2_residuals{suffix}.png` (백색성 4패널)",
        "",
    ]
    gate_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A 게이트 진단")
    parser.add_argument("--run", default="runs/stage_a/sio2-freeze", help="run 디렉토리")
    parser.add_argument(
        "--tag",
        default="",
        help="산출물 이름 접미사 (_<tag>) — 채택 디코더의 무태그 게이트 기록 보호용",
    )
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    suffix = f"_{args.tag}" if args.tag else ""
    fig1_path = FIG_DIR / f"fig_stage_a1_dispersion{suffix}.png"
    fig2_path = FIG_DIR / f"fig_stage_a2_residuals{suffix}.png"
    gate_path = GATE_PATH.with_name(f"stage_a_gate{suffix}.md")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_calibrated_stack(run_dir / "model.pt")
    x_diag, d_diag, metrics = load_diag_split(run_dir)
    run_name = metrics["run_name"]
    print(
        f"run stage_a/{run_name}: best step {ckpt['step']} (val_rmse {ckpt['val_rmse']:.5f}),"
        f" 진단 {len(x_diag):,}행"
    )

    eps = compute_residuals(model, x_diag, d_diag)
    wm = whiteness_metrics(eps, d_diag.astype(np.float64))
    disp = figure_dispersion(model, run_name, fig1_path)
    figure_residuals(eps, x_diag, d_diag, model, wm, run_name, fig2_path)
    write_gate_report(run_name, ckpt, wm, disp, len(x_diag), gate_path, suffix)

    print(
        f"게이트 (a) RMSE {wm['rmse']:.5f} vs {GATE_A_RMSE}: "
        f"{'통과' if wm['rmse'] < GATE_A_RMSE else '실패'}"
    )
    print(
        f"(c) bias {wm['bias']:+.5f} / RMSE/σ {wm['rmse_over_sigma']:.3f} / "
        f"채널비 {wm['channel_rmse_ratio']:.3f} / rho1 {wm['rho1']:+.4f} / "
        f"RMSE/σ_hf {wm['rmse_over_sigma_hf']:.3f}"
    )
    print(f"산출물: {gate_path}, {fig1_path}, {fig2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
