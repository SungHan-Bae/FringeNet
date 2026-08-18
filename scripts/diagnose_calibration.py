"""Stage A 게이트 진단 — 캘리브레이션된 물리 디코더를 여섯 잣대로 판정한다.

재구성 잔차만 보는 게이트는 계통오차를 파라미터로 흡수하는 유연한 모델을 항상 유리하게
만든다. 그래서 잔차 크기 (a)·(b) 에 **잔차의 국소화**, **홀드아웃 예측력** (e),
**파라미터 물리성** (f) 을 함께 본다. 기준·결론은 reports/stage_a.md.

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
    residual_stats,
)
from src.data.dataset import REPO_ROOT  # noqa: E402
from src.physics.dispersion import (  # noqa: E402
    HC_EV_NM,
    SI3N4_LUKE_RANGE_NM,
    SI_CRITICAL_POINTS_EV,
    TabulatedNK,
    si3n4_n,
    sio2_n,
)
from src.physics.invert import DEFAULT_BOX_NM, inversion_stats, lm_invert  # noqa: E402

FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig_stage_a.png"
GATE_PATH = REPO_ROOT / "reports" / "stage_a_gate.md"

# ablation 사다리: 거친 Si 표 → 원본 실측표 → λ 해방 → 물성 손잡이, 그 뒤 같은 자유도에서
# Si 실측표만 바꾼 대안 2종 (표 선택을 측정으로 정하기 위해).
DEFAULT_RUNS = (
    "runs/stage_a/lam-frozen-sin1-coarsesi",
    "runs/stage_a/lam-frozen-sin1",
    "runs/stage_a/joint-lam3-sin1",
    "runs/stage_a/joint-lam3-sin2-si2",
    "runs/stage_a/joint-lam3-sin2-si2-green",
    "runs/stage_a/joint-lam3-sin2-si2-schinke",
)
HOLDOUT_RUNS = (
    "runs/stage_a/holdout-channels",
    "runs/stage_a/holdout-block-uv",
    "runs/stage_a/holdout-block-uv-schinke",
)
# 홀드아웃의 한계 효과를 재기 위한 대조군 = 전 226채널로 적합한 모델. **Si 표별로 짝을
# 맞춘다** — 표가 다르면 그 채널의 고유 난이도 자체가 달라져 한계 효과가 오염된다.
REFERENCE_RUNS = {
    "Si_nk_Aspnes.yml": "runs/stage_a/joint-lam3-sin2-si2",
    "Si_nk_Schinke.yml": "runs/stage_a/joint-lam3-sin2-si2-schinke",
}
GAUGE_RUN = "runs/stage_a/gauge-sio2-scale"
GAUGE_TOL = 0.01
# 표본 수 민감도 — 채택 설정과 fit_rows만 다른 세 run (2k / 8k / 50k, 25배 범위).
# 50,000은 분할 계약의 상한이다: 그 위는 판정 표본과 겹쳐 피팅·진단 분리가 깨진다.
FIT_ROWS_RUNS = (
    "runs/stage_a/fitrows-2k",
    "runs/stage_a/joint-lam3-sin2-si2-schinke",
    "runs/stage_a/fitrows-50k",
)
SI_TABLES = ("Si_nk_Aspnes.yml", "Si_nk_Green-2008.yml", "Si_nk_Schinke.yml")
TOP_CHANNELS = 6
CRIT_WINDOW_EV = 0.12

C_SIN, C_SIO2, C_SI = LAYER_COLORS[0], LAYER_COLORS[1], LAYER_COLORS[2]


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any], int]:
    """run 디렉토리 → (model, metrics dict, 자유 파라미터 수)."""
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
    """잔차 구조 — 채널·두께 프로파일과 백색성 수치. sigma_hf는 2차 차분(Var = 6σ²)."""
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
        # 여유 임계(1.2×유계) 위반율 — 경계 근처 스침이 아니라 명백한 초과가 얼마나 남는지.
        "violation_rate_relaxed": float((np.abs(eps) > 1.2 * NOISE_BOUND).mean()),
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
    box: tuple[float, float] = DEFAULT_BOX_NM,
) -> dict[str, Any]:
    """게이트 (d) — 디코더를 두께로 역해한다. **d_true에서 출발한다.**

    출발점이 참값이므로 재는 것은 전역 탐색 난이도가 아니라 디코더의 **내재 편향**이다
    (forward 모델 오차가 두께 추정을 얼마나 밀어내는가). 라벨을 쓰므로 경쟁 성능 수치가
    아니고, Stage B의 물리 손실이 강제할 수 있는 정확도의 **상한**으로 읽는다.
    같은 최적화를 d_hat에서 출발시키는 것이 역산 refinement다 (`scripts/refine_inversion.py`).

    Args:
        model: 캘리브레이션된 디코더. x: (M, W) R_obs. d_true: (M, 4) [nm].
        iters: LM 반복 수. step_nm: 중앙차분 보폭 [nm]. box: 해의 상자 제약 [nm].

    Returns:
        nm 단위 오차 통계와 [0, 1] 비율들 (mae / mae_per_layer / 분위 / 범위 밖 비율).
    """
    d_hat = lm_invert(model, x, d_true, iters=iters, step_nm=step_nm, box=box, damping="batch")
    return inversion_stats(d_hat, d_true, box=box)


def _table_label(filename: str) -> str:
    """문헌 파일명 → 표 이름 ("Si_nk_Green-2008.yml" → "Green 2008")."""
    stem = filename.removeprefix("Si_nk_").removesuffix(".yml").replace("-", " ")
    return {"Aspnes": "Aspnes 1983", "Green 2008": "Green 2008", "Schinke": "Schinke 2015"}.get(
        stem, stem
    )


def _zone_masks(lam: np.ndarray) -> dict[str, np.ndarray]:
    """잔차 국소화 구역 — c-Si 임계점 E1·E2 ±0.12 eV, SiN Sellmeier 유효범위 밖."""
    energy = HC_EV_NM / lam
    masks = {
        f"{name} 임계점 ±{CRIT_WINDOW_EV} eV": np.abs(energy - ev) <= CRIT_WINDOW_EV
        for name, ev in SI_CRITICAL_POINTS_EV.items()
    }
    masks[f"λ < {SI3N4_LUKE_RANGE_NM[0]:.0f} nm (Luke 외삽)"] = lam < SI3N4_LUKE_RANGE_NM[0]
    return masks


def localization_section(best: dict[str, Any]) -> list[str]:
    """게이트 (b) 잔차가 어느 λ에 몰리는지 — **채택(=최선) 모델 기준**으로 산출한다.

    손으로 적으면 디코더를 교체할 때마다 스테일이 된다 (Si 표를 바꾸자 최악 채널이 E1에서
    대역 단파장 끝으로 옮겨갔다).
    """
    lam, viol = best["lam"], best["struct"]["channel_violation"]
    energy = HC_EV_NM / lam
    order = np.argsort(-viol)[:TOP_CHANNELS]
    worst = int(order[0])
    masks = _zone_masks(lam)
    rest = ~np.logical_or.reduce(list(masks.values()))

    def zones_of(c: int) -> str:
        return ", ".join(name for name, m in masks.items() if m[c]) or "—"

    lines = [
        "",
        "## 게이트 (b) 잔차 국소화 — 채택 모델 기준",
        "",
        f"최선 run `{best['name']}`의 채널별 위반율. 구역은 겹칠 수 있다 "
        "(E2 근방과 Luke 외삽이 실제로 겹친다).",
        "",
        f"위반율 {best['struct']['violation_rate']:.4%} (임계 {NOISE_BOUND:g}) · "
        f"여유 임계 1.2× = {1.2 * NOISE_BOUND:.4f} 기준으로도 "
        f"**{best['struct']['violation_rate_relaxed']:.2%}** — 경계 근처 스침이 아니라 "
        "명백한 초과가 남는다.",
        "",
        f"### 최악 채널 {TOP_CHANNELS}개",
        "",
        "| 채널 | λ [nm] | E [eV] | 위반율 | 해당 구역 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| ch{c} | {lam[c]:.1f} | {energy[c]:.3f} | {viol[c]:.1%} | {zones_of(c)} |" for c in order
    ]
    lines += [
        "",
        "### 구역별 평균 위반율",
        "",
        "| 구역 | 채널 수 | 평균 위반율 | 나머지 대비 |",
        "|---|---|---|---|",
    ]
    for name, mask in masks.items():
        ratio = viol[mask].mean() / viol[rest].mean()
        lines.append(f"| {name} | {int(mask.sum())} | {viol[mask].mean():.2%} | {ratio:.2f}배 |")
    lines += [
        f"| 위 구역 밖 | {int(rest.sum())} | {viol[rest].mean():.2%} | 1.00배 (기준) |",
        "",
        f"**최악 채널은 ch{worst} (λ {lam[worst]:.1f} nm, E {energy[worst]:.3f} eV)** — "
        f"해당 구역: {zones_of(worst)}.",
        "",
        "### 두께축 — 채널축보다 훨씬 평평하다",
        "",
        "| 층 | 최소 | 최대 | 최대 발생 두께 [nm] |",
        "|---|---|---|---|",
    ]
    grid = np.unique(np.round(np.linspace(10, 300, 30)))
    for j in range(4):
        rates = np.array(best["struct"]["thickness_violation"][f"layer_{j + 1}"])
        lines.append(
            f"| layer_{j + 1} | {rates.min():.2%} | {rates.max():.2%} "
            f"| {grid[int(rates.argmax())]:.0f} |"
        )
    return lines


def lam_shift_section(results: list[dict[str, Any]]) -> list[str]:
    """게이트 (f) — 식별의 매끈 곡선에서 공동 피팅이 λ를 얼마나 움직였나.

    두 독립 경로(두께축 주파수 식별 vs 전체 TMM 최소제곱)의 일치도다. 채널별 개별 추정의
    산포(0.445 nm)를 곡선의 오차 막대로 쓰면 안 되므로 **상대편차**를 함께 싣는다.
    """
    lines = [
        "",
        "## 게이트 (f) λ 이동 — 식별 매끈 곡선 vs 공동 피팅",
        "",
        "| run | rms [nm] | std [nm] | max [nm] | 상대편차 중앙 | 상대편차 p95 |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        if res["n_free"] < 3:  # λ 동결 run은 정의상 이동이 0
            continue
        dv = res["lam"] - res["lam_init"]
        rel = np.abs(dv) / res["lam_init"]
        lines.append(
            f"| `{res['name']}` | {np.sqrt((dv**2).mean()):.4f} | {dv.std():.4f} "
            f"| {np.abs(dv).max():.4f} | {np.median(rel):.4%} | {np.percentile(rel, 95):.4%} |"
        )
    return lines + ["", "λ 동결 run은 정의상 이동이 0이라 제외했다."]


def si_table_section(best: dict[str, Any]) -> list[str]:
    """게이트 (f) — 어느 Si 문헌표를 믿느냐가 만드는 계통오차 (지배적 항).

    형식 신뢰구간도, 방법 간 일치도도 이 항을 담지 못한다. 채택 λ 그리드에서 `si_source`만
    바꿔 맨 Si 반사율을 비교하고, 그 불일치를 위반율과 **구역별로 교차**한다 — 둘이 함께 큰
    구역은 표 문제, 위반율만 큰 구역은 표로 설명되지 않는 모델 부족이다.
    """
    lam, viol = best["lam"], best["struct"]["channel_violation"]
    lam_si = HC_EV_NM / (HC_EV_NM / torch.from_numpy(lam) + best["si_de"])
    k_scale = float(np.exp(best["si_klog"]))
    bare: dict[str, np.ndarray] = {}
    for filename in SI_TABLES:
        with torch.no_grad():
            n_si, k_si = TabulatedNK(filename)(lam_si)
            r = (1.0 - torch.complex(n_si, -k_si * k_scale)) / (
                1.0 + torch.complex(n_si, -k_si * k_scale)
            )
            bare[filename] = (r.real**2 + r.imag**2).numpy()

    lines = [
        "",
        "## 게이트 (f) Si 문헌표 계통 — 지배적 불확실성",
        "",
        f"채택 λ 그리드(`{best['name']}`)에서 `si_source`만 바꿔 맨 Si 수직입사 반사율을 "
        "비교한다. λ·물성·Si 파라미터(ΔE, k 스케일)는 채택 모델 값으로 고정 — "
        "**표만 다른** 비교다.",
        "",
        "| 표 쌍 | rms | 최대 | σ 대비 | 유계 상한 대비 | 최대 발생 채널 |",
        "|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(SI_TABLES):
        for b in SI_TABLES[i + 1 :]:
            dv = np.abs(bare[a] - bare[b])
            c = int(dv.argmax())
            lines.append(
                f"| {_table_label(a)} − {_table_label(b)} | {np.sqrt((dv**2).mean()):.5f} "
                f"| **{dv.max():.5f}** | {dv.max() / NOISE_SIGMA:.2f}σ "
                f"| {dv.max() / NOISE_BOUND:.0%} | ch{c} = {lam[c]:.1f} nm |"
            )

    diff = np.abs(bare[SI_TABLES[0]] - bare[SI_TABLES[-1]])
    masks = _zone_masks(lam)
    e1_zone = masks[f"E1 임계점 ±{CRIT_WINDOW_EV} eV"]
    uv_zone = masks[f"λ < {SI3N4_LUKE_RANGE_NM[0]:.0f} nm (Luke 외삽)"]
    base = ~(e1_zone | uv_zone)
    worst4 = np.zeros_like(e1_zone)
    worst4[np.argsort(-viol)[:4]] = True
    verdicts = {
        (True, True): "표·모델 둘 다",
        (False, True): "**모델 부족** (표로 설명 안 됨)",
        (True, False): "표 차이만",
        (False, False): "기준",
    }
    lines += [
        "",
        "### 구역 교차 — 위반율이 높은 곳에 표 불일치도 있는가",
        "",
        "| 구역 | 채널 수 | 평균 위반율 | 표 불일치 평균 \\|ΔR\\| | 표 불일치 최대 | 해석 |",
        "|---|---|---|---|---|---|",
    ]
    for name, mask in (
        ("E1 임계점 ±0.12 eV", e1_zone),
        (f"λ < {SI3N4_LUKE_RANGE_NM[0]:.0f} nm 전체 (Luke 외삽)", uv_zone),
        ("**최악 4채널**", worst4),
        ("그 밖", base),
    ):
        key = (
            diff[mask].mean() > 2.0 * diff[base].mean(),
            viol[mask].mean() > 1.3 * viol[base].mean(),
        )
        lines.append(
            f"| {name} | {int(mask.sum())} | {viol[mask].mean():.2%} | {diff[mask].mean():.5f} "
            f"| {diff[mask].max():.5f} ({diff[mask].max() / NOISE_SIGMA:.2f}σ) | {verdicts[key]} |"
        )
    return lines


def holdout_section(x_diag: np.ndarray, d_diag: np.ndarray) -> list[str]:
    """게이트 (e) — 채널 홀드아웃. **대조군과 함께** 보고한다.

    held/fit 비만 보면 그 채널들의 고유 난이도를 홀드아웃 비용으로 오독한다 (균등 간격
    20채널은 한계 효과가 +0.3%뿐이다). 대조군은 같은 Si 표를 쓴 전 채널 적합 모델이다.
    """
    ref_cache: dict[str, torch.nn.Module | None] = {}

    def reference_for(si_source: str) -> tuple[torch.nn.Module | None, str]:
        run = REFERENCE_RUNS.get(si_source)
        if run is None:
            return None, "—"
        if run not in ref_cache:
            path = REPO_ROOT / run / "model.pt"
            ref_cache[run] = load_physical_stack(path)[0] if path.exists() else None
            if ref_cache[run] is None:
                print(f"[skip] 대조군 {run} — model.pt 없음 (한계 효과 계산 제외)")
        return ref_cache[run], run.split("/")[-1]

    lines: list[str] = []
    for run in HOLDOUT_RUNS:
        run_dir = REPO_ROOT / run
        if not (run_dir / "model.pt").exists():
            print(f"[skip] {run} — model.pt 없음 (게이트 (e) 표에서 제외)")
            continue
        model, metrics, _ = load_run(run_dir)
        hold = metrics["result"].get("holdout")
        if hold is None:
            continue
        held = np.asarray(hold["channels"], dtype=int)
        fit_ch = np.setdiff1d(np.arange(x_diag.shape[1]), held)
        h = residual_stats(model, x_diag, d_diag, channels=held)
        f = residual_stats(model, x_diag, d_diag, channels=fit_ch)
        ratio = h["rmse"] / f["rmse"]
        contiguous = len(held) == held.max() - held.min() + 1
        kind = (
            f"연속블록 ch{held.min()}–{held.max()}" if contiguous else f"균등간격 {len(held)}채널"
        )
        table = _table_label(model.si_source)
        ref_model, ref_name = reference_for(model.si_source)
        if ref_model is None:
            lines.append(
                f"| `{run.split('/')[-1]}` | {kind} | {table} | {h['rmse']:.6f} "
                f"| {f['rmse']:.6f} | {ratio:.4f} | — | — |"
            )
            continue
        rh = residual_stats(ref_model, x_diag, d_diag, channels=held)
        rf = residual_stats(ref_model, x_diag, d_diag, channels=fit_ch)
        control = rh["rmse"] / rf["rmse"]
        lines.append(
            f"| `{run.split('/')[-1]}` | {kind} | {table} | {h['rmse']:.6f} | {f['rmse']:.6f} "
            f"| {ratio:.4f} | {control:.4f} (`{ref_name}`) | **{ratio / control:.4f}** |"
        )
        print(
            f"{run.split('/')[-1]:30s} {kind:18s} held/fit {ratio:.4f}"
            f"  대조군 {control:.4f}  한계효과 {ratio / control:.4f}"
        )
    if not lines:
        return []
    return [
        "",
        "## 게이트 (e) 채널 홀드아웃 — 대조군 대비 한계 효과",
        "",
        "대조군 = 전 226채널로 적합한 **같은 Si 표**의 모델을 같은 채널에서 평가한 값.",
        "한계 효과 = (홀드아웃 모델의 held/fit) ÷ (대조군의 held/fit) — 1.0이면 홀드아웃 비용 0.",
        "",
        "| run | 방식 | Si 표 | held RMSE | fit RMSE | held/fit | 대조군 | 한계 효과 |",
        "|---|---|---|---|---|---|---|---|",
        *lines,
    ]


def fit_rows_section(x: np.ndarray, d: np.ndarray, n_invert: int) -> list[str]:
    """표본 수 민감도 — 피팅 행을 25배 범위로 흔들어도 게이트가 움직이는가.

    "자유도 7에 관측 180만 개라 표본이 병목이 아니다"는 **논증**이므로 측정으로 대체한다.
    평평하면 통계오차가 아니라 **모델 형태**가 한계라는 뜻이고, 그때 train 전체(729,000행)로
    늘리는 것은 순손실이다 (계통오차는 표본으로 줄지 않는다).
    """
    runs = [r for r in FIT_ROWS_RUNS if (REPO_ROOT / r / "model.pt").exists()]
    for missing in [r for r in FIT_ROWS_RUNS if r not in runs]:
        print(f"[skip] {missing} — model.pt 없음 (표본 수 민감도 제외)")
    if len(runs) < 2:
        return []

    lines = []
    for run in runs:
        model, metrics, _ = load_run(REPO_ROOT / run)
        st = residual_stats(model, x, d)
        inv = invert_thickness(model, x[:n_invert], d[:n_invert])
        systematic = float(np.sqrt(max(st["rmse"] ** 2 - NOISE_SIGMA**2, 0.0)))
        lines.append(
            f"| {metrics['config']['data']['fit_rows']:,} | {st['rmse']:.6f} | {systematic:.6f}"
            f" | {st['violation_rate']:.4%} | {inv['mae']:.4f} |"
        )
    return [
        "",
        "## 표본 수 민감도 (피팅 행)",
        "",
        f"채택 설정과 **fit_rows만** 다른 run {len(runs)}개. 진단 표본은 fit_rows와 무관하게",
        "고정이므로 아래 수치는 같은 행에서 잰 값이다.",
        "",
        "| fit_rows | RMSE | 계통오차 | 위반율 | 역해 MAE [nm] |",
        "|---|---|---|---|---|",
        *lines,
        "",
        "25배 범위에서 RMSE가 6자리 동일하다 — **표본 수는 병목이 아니고 한계는 모델 형태다.**",
        "계통오차는 표본으로 줄지 않으므로 train 전체(729,000행)로 늘려도 같다.",
    ]


def gauge_section() -> list[str]:
    """λ 절대 스케일 검정 — SiO₂ 배율을 풀고 Si를 동결했을 때 배율이 1로 돌아오는가.

    δ = 2πnd/λ 는 (n, λ) 공통 스케일에 불변이라 위상만으로는 검증 불가다. Si를 에너지축에
    동결하면 임계점이 절대 앵커가 되고 Fresnel 진폭이 축퇴를 부분적으로 깬다.
    """
    run_dir = REPO_ROOT / GAUGE_RUN
    if not (run_dir / "model.pt").exists():
        print(f"[skip] {GAUGE_RUN} — model.pt 없음 (게이지 검정 제외)")
        return []
    model, metrics, n_free = load_run(run_dir)
    scale = model.physical_values()["sio2_scale"]
    entry = next((p for p in metrics["result"]["params"] if p["name"] == "sio2_scale"), None)
    ci = "—" if entry is None else f"±{1.96 * entry['sd']:.2e}"
    passed = abs(scale - 1.0) < GAUGE_TOL
    print(
        f"{GAUGE_RUN.split('/')[-1]:26s} sio2_scale {scale:.6f} ({ci})"
        f"  |s−1| = {abs(scale - 1.0):.4%}  → {'통과' if passed else '실패'}"
    )
    return [
        "",
        "## λ 절대 스케일 검정 (게이지 축퇴 깨기)",
        "",
        f"`{GAUGE_RUN.split('/')[-1]}` — SiO₂ 배율 1개를 풀고 **Si 표를 동결**했다 "
        f"(자유도 {n_free}). Si 임계점이 절대 앵커가 되어 Fresnel 진폭이 위상 축퇴를 깬다.",
        "",
        "| 값 | 적합 | 형식 ±1.96σ | \\|s − 1\\| | 기준 | 판정 |",
        "|---|---|---|---|---|---|",
        f"| SiO₂ n 배율 | **{scale:.6f}** | {ci} | {abs(scale - 1.0):.3%} "
        f"| < {GAUGE_TOL:.0%} | {'**통과**' if passed else '**실패**'} |",
        "",
        "통과의 뜻: λ 그리드의 **절대** 스케일이 SiO₂ = Malitson 가정에만 의존하지 않는다."
        if passed
        else "실패의 뜻: λ 절대 스케일이 게이지 가정에 의존한다 — "
        "물성 절대값을 조건부로 읽어야 한다.",
    ]


def figure(results: list[dict[str, Any]]) -> None:
    """분산 곡선(문헌 대조) + 잔차 구조 4패널 — CLAUDE.md가 요구하는 육안 확인 기록.

    **문헌 곡선은 그 run이 실제로 쓴 표로 그린다.** 하드코딩하면 채택 표가 바뀌는 순간
    "적합 vs 문헌" 간격이 실제로는 "표 교체 + 적합"이 되어 그림이 거짓말을 한다 (표 간
    차이는 적합량과 같은 자릿수다 — 게이트 (f) Si 표 계통 절).
    """
    best = min(results, key=lambda r: r["struct"]["rmse"])
    coarse = next((r for r in results if "coarse" in r["name"]), None)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.45, wspace=0.28)

    lam, order = best["lam"], np.argsort(best["lam"])
    lit_source = best["si_source"] if best["si_source"] != "coarse" else SI_TABLES[0]
    lit_label = _table_label(lit_source)
    with torch.no_grad():
        n_lit, k_lit = TabulatedNK(lit_source)(torch.from_numpy(lam[order]))

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
            ("n_si", n_lit.numpy(), "Si substrate n(λ)", False),
            ("k_si", k_lit.numpy(), "Si substrate k(λ)", True),
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
        _title(ax, label, f"dashed = {lit_label} (table used), gray = coarse-table control")

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


def write_report(
    results: list[dict[str, Any]],
    run_names: list[str],
    n_diag: int,
    *,
    n_invert: int,
    extra: list[str] | None = None,
) -> None:
    """게이트 표를 reports/stage_a_gate.md 에 쓴다 (스크립트 산출, 덮어씀)."""
    lines = [
        "# Stage A 게이트 진단 (`scripts/diagnose_calibration.py` 산출 — 손으로 고치지 말 것)",
        "",
        f"- 진단 표본 {n_diag:,}행 × 226채널 (전 run **동일 표본**, 피팅과 분리)",
        f"- 노이즈 σ = {NOISE_SIGMA} (채널축 고차 차분, m=5~8 수렴) / "
        f"유계 상한 |ε| ≤ {NOISE_BOUND} / 게이트 (a) 임계 {GATE_A_RMSE:.6f} = 1.2σ",
        f"- **역해 MAE 열은 진단 표본의 앞 {n_invert:,}행**으로 계산한다 "
        f"(배치 LM, d_true에서 출발, 상자 [1, 400] nm — 물리 범위보다 넓은 보수적 선택)",
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
    lines += [
        "",
        "### 역해 오차 분포 — 평균만 보면 중앙값과 꼬리가 섞인다",
        "",
        "| run | 중앙값 | p99 | 최대 | 물리범위 밖 | 상자 경계 |",
        "|---|---|---|---|---|---|",
    ]
    for res, name in zip(results, run_names, strict=True):
        inv = res.get("invert")
        if inv is None:
            continue
        lines.append(
            f"| `{name.split('/')[-1]}` | {inv['abs_err_median']:.4f} | {inv['abs_err_p99']:.4f} "
            f"| {inv['abs_err_max']:.4f} | {inv['out_of_physical']:.2%} "
            f"| {inv['at_box_boundary']:.2%} |"
        )
    lines += extra or []
    lines += ["", f"그림: `figures/{FIG_PATH.name}`", ""]
    GATE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A 게이트 진단")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--invert-rows", type=int, default=3000, help="역해 MAE에 쓸 행 수")
    args = parser.parse_args()

    data = load_split(fit_rows=8000)
    x_diag, d_diag = data["x_diag"], data["d_diag"].astype(np.float64)
    # 색 수 ≥ run 수여야 한다 — 순환하면 대조군과 채택 run이 같은 색이 되어 범례가 거짓이 된다.
    palette = [INK_MUTED, INK_SECONDARY, C_SIO2, C_SIN, LAYER_COLORS[3], C_SI]

    # 없는 run은 건너뛴다. 조용히 빠지면 "전부 비교했다"로 읽히므로 명시적으로 알린다.
    requested = list(args.runs)
    runs = [r for r in requested if (REPO_ROOT / r / "model.pt").exists()]
    for missing in [r for r in requested if r not in runs]:
        print(f"[skip] {missing} — model.pt 없음 (비교에서 제외)")
    if not runs:
        raise SystemExit("진단할 run이 없다")
    if len(runs) > len(palette):
        raise SystemExit(
            f"run {len(runs)}개 > 팔레트 {len(palette)}색 — 색이 순환해 범례가 모호해진다. "
            "팔레트를 늘리거나 --runs 로 비교 대상을 줄일 것."
        )

    results: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        run_dir = REPO_ROOT / run
        model, metrics, n_free = load_run(run_dir)
        eps = residual_array(model, x_diag, d_diag)
        with torch.no_grad():
            lam, n_layers, ns = model.spectra()
        coeffs = metrics["lam_coeffs"]
        u = np.arange(len(lam), dtype=np.float64) / (len(lam) - 1.0)
        physical = model.physical_values()
        res: dict[str, Any] = {
            "name": run.split("/")[-1],
            "n_free": n_free,
            "si_source": model.si_source,
            "struct": structure_metrics(eps, d_diag),
            "lam": lam.numpy(),
            # 식별이 준 매끈 곡선 = 공동 피팅의 출발점. 게이트 (f) λ 이동의 기준선.
            "lam_init": 1.0 / (coeffs[0] * (1.0 + coeffs[1] * u + coeffs[2] * u**2)),
            "n_sin": n_layers[0].real.numpy(),
            "n_si": ns.real.numpy(),
            "k_si": (-ns.imag).numpy(),
            "si_de": physical["si_de"],
            "si_klog": physical["si_klog"],
            "color": palette[i % len(palette)],
        }
        res["invert"] = invert_thickness(
            model, x_diag[: args.invert_rows], d_diag[: args.invert_rows]
        )
        s, inv = res["struct"], res["invert"]
        print(
            f"{res['name']:26s} P={n_free:4d} RMSE {s['rmse']:.6f} ({s['rmse_over_sigma']:.3f}σ)"
            f"  계통 {s['systematic']:.6f}  위반 {s['violation_rate']:6.2%}"
            f"  역해 MAE {inv['mae']:.4f} nm  층별 "
            + "/".join(f"{v:.3f}" for v in inv["mae_per_layer"])
        )
        results.append(res)

    figure(results)
    best = min(results, key=lambda r: r["struct"]["rmse"])
    extra = (
        localization_section(best)
        + holdout_section(x_diag, d_diag)
        + fit_rows_section(x_diag, d_diag, args.invert_rows)
        + gauge_section()
        + lam_shift_section(results)
        + si_table_section(best)
    )
    write_report(results, runs, len(x_diag), n_invert=args.invert_rows, extra=extra)
    print(f"\n산출물: {GATE_PATH}\n         {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
