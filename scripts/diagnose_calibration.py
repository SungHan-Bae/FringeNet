"""Stage A 게이트 진단 — 캘리브레이션된 물리 디코더를 여섯 잣대로 판정한다.

재구성 RMSE만으로는 부족하다 — 계통오차를 파라미터로 흡수하는 유연한 모델이 항상
유리해지기 때문이다. 그래서 다음을 함께 본다:

1. **유계 노이즈 위반율** (게이트 b). 데이터 노이즈는 |ε| ≤ 0.0152로 유계다 —
   근거 둘: (i) 채널축 고차 차분 σ = 0.008658 → 균등분포면 a = σ√3 = 0.014997,
   (ii) 1.83억 관측 중 R_obs < −0.0152 가 **0건** (가우시안이면 5σ = −0.043까지
   나와야 한다). 따라서 잔차가 이 상한을 넘는 관측은 통계 없이 모델 오류의 증거다.
   완벽한 모델은 위반율 0%이므로, 위반율은 모델 오차 크기의 연속 척도로도 쓴다.
2. **두께 nm 역해 MAE** (게이트 d). R 단위 RMSE는 이 프로젝트가 쓰는 단위가 아니다.
   디코더를 배치 Levenberg–Marquardt로 역해해 d̂를 뽑고 층별 MAE를 잰다 — Stage B의
   물리 손실이 강제할 수 있는 정확도의 상한이 이 값이다.
3. **잔차 구조의 국소화** — 채널·두께 축 위반율 프로파일. 잔차가 어느 λ에 몰리는지가
   무슨 물리가 부족한지를 가리킨다 (실제로 c-Si 임계점 E1·E2를 지목했다).
4. **채널 홀드아웃**(피팅에서 뺀 채널을 예측) — 매끈한 물리 분산만 통과할 수 있다.
   run의 metrics.json에 기록되며 여기서는 표로 옮긴다.

산출물:
  reports/figures/fig_stage_a.png   분산 곡선(문헌 대조) + 잔차 구조 4패널
  reports/stage_a_gate.md           게이트 수치 표 (재실행 시 덮어씀)

사용법:
    python scripts/diagnose_calibration.py                      # 기본 ablation 세트
    python scripts/diagnose_calibration.py --runs A B --invert-rows 4000
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

from eda import (  # noqa: E402 — 팔레트·축 스타일 단일 출처
    INK_MUTED,
    INK_SECONDARY,
    LAYER_COLORS,
    LAYER_LABELS,
    SURFACE,
    _style_axes,
    _title,
)

from src.calibrate import (  # noqa: E402
    GATE_A_RMSE,
    NOISE_BOUND,
    NOISE_SIGMA,
    load_physical_stack,
    load_split,
)
from src.data.dataset import REPO_ROOT  # noqa: E402
from src.physics.dispersion import TabulatedNK, si3n4_n, sio2_n  # noqa: E402

FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig_stage_a.png"
GATE_PATH = REPO_ROOT / "reports" / "stage_a_gate.md"

# 기본 ablation 사다리 — 거친 Si 표 → 원본 실측표 → λ 해방 → 물성 손잡이 추가.
DEFAULT_RUNS = (
    "runs/stage_a/lam-frozen-sin1-coarsesi",
    "runs/stage_a/lam-frozen-sin1",
    "runs/stage_a/joint-lam3-sin1",
    "runs/stage_a/joint-lam3-sin2-si2",
)

C_SIN, C_SIO2, C_SI = LAYER_COLORS[0], LAYER_COLORS[1], LAYER_COLORS[2]


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any], int]:
    """run 디렉토리에서 모델과 metrics를 읽는다.

    Returns:
        (model, metrics dict, 자유 파라미터 수).
    """
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    model, _ = load_physical_stack(run_dir / "model.pt")
    return model, metrics, int(metrics["result"]["n_free"])


@torch.no_grad()
def residual_array(model: torch.nn.Module, x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """eps = R_obs − R_model(d_true). 반환 (M, W) float64."""
    d_t = torch.from_numpy(d).to(torch.float64)
    out = np.empty(x.shape, dtype=np.float64)
    for start in range(0, len(x), 4096):
        pred = model(d_t[start : start + 4096]).numpy()
        out[start : start + 4096] = x[start : start + 4096].astype(np.float64) - pred
    return out


def structure_metrics(eps: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    """잔차 구조 — 채널·두께 프로파일과 백색성 수치."""
    viol = np.abs(eps) > NOISE_BOUND
    rmse = float(np.sqrt((eps**2).mean()))
    ch_rmse = np.sqrt((eps**2).mean(axis=0))
    bins: dict[str, list[float]] = {}
    for j in range(4):
        values = np.unique(d[:, j])
        bins[f"layer_{j + 1}"] = [float(viol[d[:, j] == v].mean()) for v in values]
    centered = eps - eps.mean(axis=0)
    d2 = eps[:, :-2] - 2.0 * eps[:, 1:-1] + eps[:, 2:]
    sigma_hf = float(np.sqrt((d2**2).mean() / 6.0))
    return {
        "rmse": rmse,
        "rmse_over_sigma": rmse / NOISE_SIGMA,
        "systematic": float(np.sqrt(max(rmse**2 - NOISE_SIGMA**2, 0.0))),
        "bias": float(eps.mean()),
        "violation_rate": float(viol.mean()),
        "max_abs": float(np.abs(eps).max()),
        "channel_rmse": ch_rmse,
        "channel_violation": viol.mean(axis=0),
        "channel_rmse_ratio": float(ch_rmse.max() / ch_rmse.min()),
        "thickness_violation": bins,
        "rho1": float((centered[:, 1:] * centered[:, :-1]).mean() / (centered**2).mean()),
        "sigma_hf": sigma_hf,
        "rmse_over_sigma_hf": rmse / sigma_hf,
    }


def invert_thickness(
    model: torch.nn.Module,
    x: np.ndarray,
    d_true: np.ndarray,
    *,
    iters: int = 30,
    step_nm: float = 1e-3,
) -> dict[str, Any]:
    """디코더를 두께로 역해한다 — 배치 Levenberg–Marquardt, d_true에서 출발.

    d_true에서 출발하는 이유: 여기서 재려는 것은 전역 탐색 난이도가 아니라 **디코더의
    내재 편향**이다 (forward 모델 오차가 두께 추정을 얼마나 밀어내는가). 야코비안은
    중앙차분 — float64에서 상대오차 ~1e-10이고 4층뿐이라 autograd보다 단순하다.

    Args:
        model: 캘리브레이션된 디코더. x: (M, W) R_obs. d_true: (M, 4) [nm].
        iters: LM 반복 수. step_nm: 중앙차분 보폭 [nm].

    Returns:
        {"mae", "mae_per_layer", "bias_per_layer", "rmse_nm"} — 전부 nm 단위.
    """
    d = torch.from_numpy(d_true.astype(np.float64)).clone()
    obs = torch.from_numpy(x.astype(np.float64))
    lam_damp = torch.full((len(d), 1, 1), 1e-3, dtype=torch.float64)
    with torch.no_grad():
        resid = model(d) - obs  # (M, W)
        cost = (resid**2).sum(dim=1)
        for _ in range(iters):
            jac = torch.stack(
                [
                    (
                        model(d + step_nm * torch.eye(4, dtype=torch.float64)[j])
                        - model(d - step_nm * torch.eye(4, dtype=torch.float64)[j])
                    )
                    / (2.0 * step_nm)
                    for j in range(4)
                ],
                dim=-1,
            )  # (M, W, 4)
            jtj = jac.transpose(1, 2) @ jac  # (M, 4, 4)
            jtr = (jac.transpose(1, 2) @ resid.unsqueeze(-1)).squeeze(-1)  # (M, 4)
            eye = torch.eye(4, dtype=torch.float64).expand_as(jtj)
            damped = jtj + lam_damp * eye * jtj.diagonal(dim1=1, dim2=2).mean()
            delta = torch.linalg.solve(damped, -jtr.unsqueeze(-1)).squeeze(-1)
            cand = (d + delta).clamp(1.0, 400.0)
            resid_c = model(cand) - obs
            cost_c = (resid_c**2).sum(dim=1)
            better = cost_c < cost
            d = torch.where(better.unsqueeze(-1), cand, d)
            resid = torch.where(better.unsqueeze(-1), resid_c, resid)
            cost = torch.where(better, cost_c, cost)
            lam_damp = torch.where(
                better.reshape(-1, 1, 1), (lam_damp * 0.3).clamp(min=1e-9), lam_damp * 3.0
            )
    err = (d - torch.from_numpy(d_true.astype(np.float64))).numpy()
    return {
        "mae": float(np.abs(err).mean()),
        "mae_per_layer": [float(v) for v in np.abs(err).mean(axis=0)],
        "bias_per_layer": [float(v) for v in err.mean(axis=0)],
        "rmse_nm": float(np.sqrt((err**2).mean())),
    }


def figure(results: list[dict[str, Any]], run_names: list[str]) -> None:
    """분산 곡선(문헌 대조) + 잔차 구조 4패널.

    상단은 최종 모델(자유도 최대)의 물성 곡선을 문헌과 겹쳐 그린다 (CLAUDE.md가
    요구하는 육안 확인 기록). 거친 표 대조군을 함께 그려 표·보간 품질의 영향을 보인다.
    """
    best = min(results, key=lambda r: r["struct"]["rmse"])
    coarse = next((r for r in results if "coarse" in r["name"]), None)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.45, wspace=0.28)

    lam, order = best["lam"], np.argsort(best["lam"])
    with torch.no_grad():
        n_lit, k_lit = TabulatedNK("Si_nk_Aspnes.yml")(torch.from_numpy(lam[order]))

    ax = axes[0, 0]
    _style_axes(ax)
    ax.plot(lam[order], si3n4_n(lam)[order], color=C_SIN, lw=1.3, ls="--", alpha=0.65)
    ax.plot(lam[order], best["n_sin"][order], color=C_SIN, lw=2)
    ax.plot(lam[order], sio2_n(lam)[order], color=C_SIO2, lw=2)
    mid = order[len(order) // 2]
    ax.annotate(
        "SiN fitted (dashed: Luke 2015)",
        (lam[mid], best["n_sin"][mid]),
        xytext=(0, 10),
        textcoords="offset points",
        color=C_SIN,
        fontsize=8,
    )
    ax.annotate(
        "SiO2 frozen (gauge)",
        (lam[mid], sio2_n(lam)[mid]),
        xytext=(0, 10),
        textcoords="offset points",
        color=C_SIO2,
        fontsize=8,
    )
    ax.set_xlabel("λ [nm]")
    ax.set_ylabel("n")
    _title(ax, "Layer n(λ)", "solid = calibrated, dashed = literature")

    for col, (vals, lit, label, logy) in enumerate(
        [
            (("n_si"), n_lit.numpy(), "Si substrate n(λ)", False),
            (("k_si"), k_lit.numpy(), "Si substrate k(λ)", True),
        ],
        start=1,
    ):
        ax = axes[0, col]
        _style_axes(ax)
        ax.plot(lam[order], lit, color=C_SI, lw=1.3, ls="--", alpha=0.65)
        ax.plot(lam[order], best[vals][order], color=C_SI, lw=2)
        if coarse is not None:
            oc = np.argsort(coarse["lam"])
            ax.plot(coarse["lam"][oc], coarse[vals][oc], color=INK_MUTED, lw=1.3)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("λ [nm]")
        ax.set_ylabel("k (log)" if logy else "n")
        _title(ax, label, "dashed = Aspnes 1983, gray = coarse-table control")

    ax = axes[1, 0]
    _style_axes(ax)
    for res in results:
        ax.plot(res["struct"]["channel_rmse"], lw=1.6, label=res["name"], color=res["color"])
    ax.axhline(NOISE_SIGMA, color=INK_MUTED, lw=1.2, ls="--")
    ax.annotate(
        "noise σ",
        (4, NOISE_SIGMA),
        xytext=(0, 5),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
    )
    ax.set_xlabel("channel index (de-identified)")
    ax.set_ylabel("residual RMSE")
    _title(ax, "Channel profile", "short λ is at the right end")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    _style_axes(ax)
    for res in results:
        ax.plot(100 * res["struct"]["channel_violation"], lw=1.6, color=res["color"])
    ax.set_xlabel("channel index (de-identified)")
    ax.set_ylabel("violation rate [%]")
    _title(ax, f"Bounded-noise violations (|ε| > {NOISE_BOUND})", "0% ⟺ model consistent")

    ax = axes[1, 2]
    _style_axes(ax)
    grid = np.unique(np.round(np.linspace(10, 300, 30)))
    for j in range(4):
        rate = 100 * np.array(best["struct"]["thickness_violation"][f"layer_{j + 1}"])
        ax.plot(grid, rate, color=LAYER_COLORS[j], lw=1.8)
        ax.annotate(
            LAYER_LABELS[j],
            (grid[-1], rate[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=LAYER_COLORS[j],
            fontsize=8,
            va="center",
        )
    ax.set_xlim(0, 385)
    ax.set_xlabel("thickness [nm]")
    ax.set_ylabel("violation rate [%]")
    _title(ax, "Violations by thickness", f"best run: {best['name']}")

    fig.suptitle("Stage A — physically constrained calibration", y=0.99, fontsize=12)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def write_report(results: list[dict[str, Any]], run_names: list[str], n_diag: int) -> None:
    """게이트 표를 reports/stage_a_gate.md 에 쓴다 (스크립트 산출, 덮어씀)."""
    lines = [
        "# Stage A 게이트 진단 (`scripts/diagnose_calibration.py` 산출 — 손으로 고치지 말 것)",
        "",
        f"- 진단 표본 {n_diag:,}행 × 226채널 (전 run **동일 표본**, 피팅과 분리)",
        f"- 노이즈 σ = {NOISE_SIGMA} (채널축 고차 차분, m=5~8 수렴) / "
        f"유계 상한 |ε| ≤ {NOISE_BOUND} / 게이트 (a) 임계 {GATE_A_RMSE:.6f} = 1.2σ",
        "",
        "## 게이트 종합",
        "",
        "| run | 자유도 | RMSE | RMSE/σ | 계통오차 | (a) | 위반율 | max ε | (b) | 역해 MAE [nm] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res, name in zip(results, run_names, strict=True):
        s = res["struct"]
        inv = res.get("invert")
        inv_cell = "—" if inv is None else format(inv["mae"], ".4f")
        gate_a = "✓" if s["rmse"] < GATE_A_RMSE else "✗"
        gate_b = "✓" if s["violation_rate"] == 0 else "✗"
        lines.append(
            f"| `{name.split('/')[-1]}` | {res['n_free']} | {s['rmse']:.6f} | "
            f"{s['rmse_over_sigma']:.3f} | {s['systematic']:.6f} | {gate_a} | "
            f"{s['violation_rate']:.2%} | {s['max_abs']:.5f} | {gate_b} | {inv_cell} |"
        )
    lines += [
        "",
        "## 잔차 백색성 (참고)",
        "",
        "| run | bias | 채널 RMSE max/min | lag-1 | RMSE/σ_hf |",
        "|---|---|---|---|---|",
    ]
    for res, name in zip(results, run_names, strict=True):
        s = res["struct"]
        lines.append(
            f"| `{name.split('/')[-1]}` | {s['bias']:+.5f} | {s['channel_rmse_ratio']:.3f} | "
            f"{s['rho1']:+.4f} | {s['rmse_over_sigma_hf']:.3f} |"
        )
    lines += [
        "",
        "## 층별 역해 오차 [nm]",
        "",
        "| run | layer_1 | layer_2 | layer_3 | layer_4 | 전체 MAE |",
        "|---|---|---|---|---|---|",
    ]
    for res, name in zip(results, run_names, strict=True):
        inv = res.get("invert")
        if inv is None:
            continue
        per = " | ".join(f"{v:.4f}" for v in inv["mae_per_layer"])
        lines.append(f"| `{name.split('/')[-1]}` | {per} | **{inv['mae']:.4f}** |")
    lines += ["", f"그림: `figures/{FIG_PATH.name}`", ""]
    GATE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A 게이트 진단")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--invert-rows", type=int, default=3000, help="역해 MAE에 쓸 행 수")
    args = parser.parse_args()

    data = load_split(fit_rows=8000)
    x_diag, d_diag = data["x_diag"], data["d_diag"].astype(np.float64)
    palette = [INK_MUTED, C_SIO2, C_SIN, C_SI, INK_SECONDARY]

    # 없는 run은 건너뛴다. 조용히 빠지면 "전부 비교했다"로 읽히므로 명시적으로 알린다.
    requested = list(args.runs)
    runs = [r for r in requested if (REPO_ROOT / r / "model.pt").exists()]
    for missing in [r for r in requested if r not in runs]:
        print(f"[skip] {missing} — model.pt 없음 (비교에서 제외)")
    if not runs:
        raise SystemExit("진단할 run이 없다")

    results: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        run_dir = REPO_ROOT / run
        model, metrics, n_free = load_run(run_dir)
        eps = residual_array(model, x_diag, d_diag)
        with torch.no_grad():
            lam, n_layers, ns = model.spectra()
        res: dict[str, Any] = {
            "name": run.split("/")[-1],
            "n_free": n_free,
            "struct": structure_metrics(eps, d_diag),
            "lam": lam.numpy(),
            "n_sin": n_layers[0].real.numpy(),
            "n_si": ns.real.numpy(),
            "k_si": (-ns.imag).numpy(),
            "color": palette[i % len(palette)],
        }
        rows = args.invert_rows
        res["invert"] = invert_thickness(model, x_diag[:rows], d_diag[:rows])
        s, inv = res["struct"], res["invert"]
        label = run.split("/")[-1]
        print(
            f"{label:26s} P={n_free:4d} RMSE {s['rmse']:.6f} ({s['rmse_over_sigma']:.3f}σ)"
            f"  계통 {s['systematic']:.6f}  위반 {s['violation_rate']:6.2%}"
            f"  역해 MAE {inv['mae']:.4f} nm  층별 "
            + "/".join(f"{v:.3f}" for v in inv["mae_per_layer"])
        )
        results.append(res)

    figure(results, runs)
    write_report(results, runs, len(x_diag))
    print(f"\n산출물: {GATE_PATH}\n         {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
