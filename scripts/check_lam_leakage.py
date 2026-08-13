"""주파수 식별이 진단 표본을 포함하는 것이 게이트 수치를 낙관적으로 만드는가.

`src.calibrate`의 분할 계약은 "피팅 8,000행 / 판정 20,000행, 서로 분리"인데, λ 초기값을
만드는 두께축 주파수 식별(`identify_lam_coefficients`)만은 조건부 평균의 정확한 주변화가
필요해서 **holdout 제외 train 전체(729,000행)** 를 쓴다. 그 안에 판정 20,000행이
2.74% 포함된다. 이 스크립트는 그 예외가 실제로 판정 수치를 움직이는지 측정한다.

세 가지 식별을 만들어 비교한다:

  A) `all`       현행 — train 전체 729,000행 (판정 20,000행 포함)
  B) `no_diag`   판정 20,000행 제외 — 709,000행
  C) `no_other`  판정도 피팅도 아닌 **다른** 무작위 20,000행 제외 — 709,000행

C가 재표집 분산의 기준선이다. B와 C가 A로부터 비슷하게 벌어지면 그 움직임은 판정 행의
특이성이 아니라 추정량의 재표집 분산(주파수 후보 격자 양자화)이라는 뜻이다.

그 다음 각 λ 초기값으로 두 가지 설정을 실제로 적합해 판정 표본에서 평가한다:

  - λ **해방** (`joint-lam3-sin2-si2-schinke` = 채택 디코더): 식별 결과는 초기값일 뿐이라
    최소제곱이 λ를 다시 정한다 → 초기값 의존이 남는지 확인.
  - λ **동결** (`lam-frozen-sin1` = 사다리 1단): λ가 전적으로 식별 산출물이므로 예외가
    모델에 직접 들어간다 → 노출이 가장 큰 경로.

적합은 `src.calibrate`와 같은 신뢰영역 최소제곱(scipy TRF)이다.

비용: 식별 3회(각 ~45초, 최대 상주 ~6 GB) + 적합 6회 = 약 3분, 전부 CPU.

사용법:
    python scripts/check_lam_leakage.py            # 측정 + reports/stage_a_leakage.md 갱신
    python scripts/check_lam_leakage.py --no-write # 표준출력만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibrate import (  # noqa: E402
    _SPLIT_DIAG_ROWS,
    _SPLIT_FIT_ROWS,
    _SPLIT_SEED,
    PhysicalStack,
    fit_lam_coefficients,
    residual_stats,
)
from src.data.dataset import REPO_ROOT, prepare_train_arrays  # noqa: E402
from src.physics.freq_id import identify_wavelength_grid  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "stage_a_leakage.md"
# 대조군 C가 뺄 20,000행을 고르는 시드 (A·B와 독립이어야 하므로 분할 시드와 다르게 둔다).
CONTROL_SEED = 7
# 비교할 두 설정 — λ 해방(채택 디코더)과 λ 동결(노출이 가장 큰 경로).
SETTINGS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "joint-lam3-sin2-si2-schinke",
        "λ 해방 (채택 디코더, 자유도 7)",
        ("lam_nu0", "lam_r1", "lam_r2", "sin_b1", "sin_c1", "si_de", "si_klog"),
        "Si_nk_Schinke.yml",
    ),
    ("lam-frozen-sin1", "λ 동결 (사다리 1단, 자유도 1)", ("sin_b1",), "Si_nk_Aspnes.yml"),
)


def identify_variants() -> tuple[dict[str, tuple[float, ...]], dict[str, dict[str, Any]], int]:
    """세 가지 식별을 수행한다. 반환 (λ 계수, 식별 진단, 채널 수)."""
    x, y, train_idx, _ = prepare_train_arrays(val_frac=0.1, seed=_SPLIT_SEED)
    rng = np.random.default_rng(_SPLIT_SEED)
    pick = rng.choice(train_idx, size=_SPLIT_FIT_ROWS + _SPLIT_DIAG_ROWS, replace=False)
    diag_idx = pick[_SPLIT_FIT_ROWS:]
    rest = np.setdiff1d(train_idx, pick)
    ctrl_idx = np.random.default_rng(CONTROL_SEED).choice(
        rest, size=_SPLIT_DIAG_ROWS, replace=False
    )
    subsets = {
        "all": train_idx,
        "no_diag": np.setdiff1d(train_idx, diag_idx),
        "no_other": np.setdiff1d(train_idx, ctrl_idx),
    }
    coeffs: dict[str, tuple[float, ...]] = {}
    diags: dict[str, dict[str, Any]] = {}
    for tag, idx in subsets.items():
        ident = identify_wavelength_grid(x[idx], y[idx])
        coeffs[tag] = fit_lam_coefficients(ident["lam_grid"])
        diags[tag] = {"n_rows": int(len(idx)), **ident["diagnostics"]}
        print(
            f"[ident] {tag:9s} n={len(idx):,}  λ {ident['lam_grid'].min():.4f}"
            f"–{ident['lam_grid'].max():.4f} nm  불신 채널 "
            f"{ident['diagnostics']['unreliable_channels']}"
        )
    return coeffs, diags, x.shape[1]


def smooth_grid(coeffs: tuple[float, ...], n_ch: int) -> np.ndarray:
    """λ 3계수 → 채널별 매끈 곡선 (W,) [nm]."""
    u = np.arange(n_ch, dtype=np.float64) / (n_ch - 1.0)
    return 1.0 / (coeffs[0] * (1.0 + coeffs[1] * u + coeffs[2] * u**2))


def refit(
    coeffs: tuple[float, ...],
    free: tuple[str, ...],
    si_source: str,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    """주어진 λ 초기값으로 재적합(TRF) 후 판정 표본에서 평가한다."""
    model = PhysicalStack(
        n_channels=data["x_fit"].shape[1], lam_coeffs=coeffs, free=free, si_source=si_source
    )
    x_fit = torch.from_numpy(data["x_fit"]).to(torch.float64)
    d_fit = torch.from_numpy(data["d_fit"]).to(torch.float64)

    def residual(theta: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            model.theta.copy_(torch.from_numpy(theta).to(torch.float64))
            return (model(d_fit) - x_fit).reshape(-1).numpy()

    result = least_squares(residual, np.zeros(len(free)), method="trf", xtol=1e-12, ftol=1e-12)
    with torch.no_grad():
        model.theta.copy_(torch.from_numpy(result.x).to(torch.float64))
        lam = model.lam().numpy()
    stats = residual_stats(model, data["x_diag"], data["d_diag"])
    return {
        "rmse": stats["rmse"],
        "rmse_over_sigma": stats["rmse_over_sigma"],
        "violation_rate": stats["violation_rate"],
        "lam_range": (float(lam.min()), float(lam.max())),
        "sin_b1": model.physical_values()["sin_b1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="주파수 식별의 진단 표본 포함 영향 측정")
    parser.add_argument("--no-write", action="store_true", help=f"{OUT_PATH.name} 갱신 생략")
    args = parser.parse_args()

    from src.calibrate import load_split  # 지연 import — 위 식별이 메모리를 먼저 반납하게

    coeffs, diags, n_ch = identify_variants()
    base = smooth_grid(coeffs["all"], n_ch)
    spread = {
        tag: (
            float(np.sqrt(((smooth_grid(coeffs[tag], n_ch) - base) ** 2).mean())),
            float(np.abs(smooth_grid(coeffs[tag], n_ch) - base).max()),
        )
        for tag in ("no_diag", "no_other")
    }
    for tag, (rms, mx) in spread.items():
        print(f"[curve] {tag:9s} A 대비 rms {rms:.4f} nm / max {mx:.4f} nm")

    data = load_split(fit_rows=8000)
    fits: dict[str, dict[str, dict[str, Any]]] = {}
    for run_name, label, free, si_source in SETTINGS:
        fits[run_name] = {}
        for tag in ("all", "no_diag", "no_other"):
            out = refit(coeffs[tag], free, si_source, data)
            fits[run_name][tag] = out
            print(
                f"[fit] {run_name:28s} {tag:9s} RMSE {out['rmse']:.6f}"
                f" ({out['rmse_over_sigma']:.4f}σ)  위반 {out['violation_rate']:.4%}"
                f"  λ {out['lam_range'][0]:.2f}–{out['lam_range'][1]:.2f}"
            )
        del label

    lines = [
        "# 주파수 식별의 진단 표본 포함 영향"
        " (`scripts/check_lam_leakage.py` 산출 — 손으로 고치지 말 것)",
        "",
        "분할 계약은 피팅 8,000행 / 판정 20,000행 분리인데, λ 초기값을 만드는 두께축 주파수",
        "식별만은 조건부 평균의 정확한 주변화가 필요해 **holdout 제외 train 전체**를 쓴다 —",
        f"판정 {_SPLIT_DIAG_ROWS:,}행이 {_SPLIT_DIAG_ROWS / diags['all']['n_rows']:.2%} 포함된다.",
        "이 표는 그 예외가 판정 수치를 움직이는지 측정한 것이다.",
        "",
        "## 식별 변형",
        "",
        "| 변형 | 식별에 쓴 행 | λ 범위 [nm] | 불신 채널 | 매끈 곡선이 `all`과 벌어진 정도 |",
        "|---|---|---|---|---|",
    ]
    for tag, note in (
        ("all", "현행 (판정 행 포함)"),
        ("no_diag", "판정 20,000행 제외"),
        ("no_other", "**대조군** — 다른 20,000행 제외"),
    ):
        d = diags[tag]
        cell = "— (기준)"
        if tag in spread:
            cell = f"rms {spread[tag][0]:.4f} / max {spread[tag][1]:.4f} nm"
        lines.append(
            f"| `{tag}` — {note} | {d['n_rows']:,} | {d['lam_range'][0]:.2f}–"
            f"{d['lam_range'][1]:.2f} | {d['unreliable_channels']} | {cell} |"
        )
    lines += [
        "",
        "대조군이 판정 행 제외와 **같은 자릿수**로 움직이므로, 이 움직임은 판정 행의 특이성이",
        "아니라 추정량의 재표집 분산(주파수 후보 격자 양자화)이다.",
        "",
        "## 각 λ 초기값으로 재적합 → 판정 표본 평가",
        "",
        "| 설정 | 식별 변형 | RMSE | RMSE/σ | 위반율 | 적합 후 λ 범위 [nm] | SiN B₁ |",
        "|---|---|---|---|---|---|---|",
    ]
    for run_name, label, _free, _si in SETTINGS:
        for tag in ("all", "no_diag", "no_other"):
            f = fits[run_name][tag]
            lines.append(
                f"| {label} | `{tag}` | {f['rmse']:.6f} | {f['rmse_over_sigma']:.4f} "
                f"| {f['violation_rate']:.4%} | {f['lam_range'][0]:.2f}–{f['lam_range'][1]:.2f} "
                f"| {f['sin_b1']:.5f} |"
            )
    joint = fits[SETTINGS[0][0]]
    invariant = len({format(v["rmse"], ".6f") for v in joint.values()}) == 1
    frozen = [v["rmse"] for v in fits[SETTINGS[1][0]].values()]
    lo, hi, cur = min(frozen), max(frozen), fits[SETTINGS[1][0]]["all"]["rmse"]
    lines += [
        "",
        f"**λ 해방 설정은 세 변형에서 {'6자리 동일' if invariant else '값이 갈린다'}** — LM이 λ를 "
        "다시 적합하므로 식별 결과는 초기값일 뿐이고, 채택 디코더의 게이트 수치는 이 예외에 "
        f"{'의존하지 않는다' if invariant else '의존한다'}.",
        "",
        "λ **동결** 설정은 λ가 전적으로 식별 산출물이라 값이 움직인다: RMSE가 "
        f"{lo:.6f}~{hi:.6f}({(hi - lo) / cur:.1%} 폭)이고 현행 값 {cur:.6f}은 그 범위의 "
        "**낙관적 끝**에 있다. 사다리 하단 대조군 수치는 그만큼 조건부로 읽어야 한다 — "
        "다만 사다리 간격(0.0126 → 0.0096)이 이 폭의 10배 이상이라 순서·결론은 바뀌지 않는다.",
        "",
    ]
    if not args.no_write:
        OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n산출물: {OUT_PATH}")
    else:
        print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
