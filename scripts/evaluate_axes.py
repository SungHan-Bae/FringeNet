"""평가 축 — 노이즈 강건성(README §3.5-4)과 라벨 없는 신뢰도 지표(§3.4).

random split(격자 위 조합 보간) 하나만 보면 β ablation이 "차이 없음"으로 끝날 위험이 크다.
이 스크립트는 **학습 없이 체크포인트 위에서** 도는 두 축을 잰다.

1. **노이즈 강건성** — 입력 R에 노이즈를 주입했을 때의 MAE 열화. 데이터에 이미
   σ = 0.008658이 있으므로 주입은 **추가분**이고, 기본은 데이터와 같은 종류인
   균등 ±0.015다 (가우시안은 별개 질문으로 병기). 모든 run이 **같은 노이즈 실현**을
   보므로 run 사이 차이는 난수가 아니라 모델에 귀속된다.
2. **신뢰도 지표** — 행별 물리 잔차 L1(R_dec(d̂), R_obs)와 실제 오차의 순위상관.
   라벨을 쓰지 않으므로 test·실계측에도 적용할 수 있다. 지표가 노이즈 바닥
   E|ε| = 0.0075에 붙어 무력해지는지도 함께 본다.

체크포인트는 git에 없다 (runs/CHECKPOINTS.md) — Drive 미러에서 복사하거나
`git show <원본 커밋>:runs/<실험>/<run>/model.pt > model.pt`로 되살린 뒤 실행한다.

산출물:
  reports/<실험>_axes.md    (재실행 시 덮어씀)

사용법:
    python scripts/evaluate_axes.py --run runs/stage_b/beta0 --run runs/stage_b/beta100
    python scripts/evaluate_axes.py --run runs/level1_cnn/flatten-dilated-bound --rows 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibrate import NOISE_SIGMA  # noqa: E402
from src.data.dataset import REPO_ROOT, prepare_from_config  # noqa: E402
from src.evaluate import load_model_checkpoint, mae_per_layer, predict  # noqa: E402
from src.losses import DEFAULT_DECODER, FrozenDecoder  # noqa: E402

# 균등 ±a 노이즈의 평균 절대값 = a/2. 데이터의 a = 0.014996이므로 물리 항의 하한이 0.0075다.
NOISE_MEAN_ABS = 0.0075
# 주입 수준 — 균등은 데이터와 같은 종류(기본 ±0.015), 가우시안은 σ의 배수로 별개 병기.
UNIFORM_LEVELS = (0.0, 0.0075, 0.015, 0.030)
GAUSSIAN_LEVELS = (0.5, 1.0, 2.0)


def load_run(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """run 디렉토리에서 모델과 metrics.json을 읽는다."""
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json이 없다: {metrics_path}")
    if not ckpt_path.exists():
        rel = ckpt_path.relative_to(REPO_ROOT) if ckpt_path.is_relative_to(REPO_ROOT) else ckpt_path
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt_path}\n"
            f"  runs/는 텍스트 산출물만 git 추적한다 — Drive 미러에서 복사하거나\n"
            f"  git show <원본 커밋>:{rel} > {rel} 로 되살릴 것 (runs/CHECKPOINTS.md)"
        )
    return load_model_checkpoint(ckpt_path), json.loads(metrics_path.read_text())


def holdout_of(cfg: dict[str, Any], cache: dict[tuple, tuple]) -> tuple[np.ndarray, np.ndarray]:
    """config가 정의하는 holdout (x, y). 같은 분할은 한 번만 읽는다 (train 전체가 0.7 GB)."""
    data_cfg = cfg.get("data") or {}
    key = (
        float(data_cfg.get("val_frac", 0.1)),
        int(cfg["seed"]),
        data_cfg.get("subset"),
        tuple(data_cfg.get("holdout_thickness") or ()),
    )
    if key not in cache:
        x, y, _, holdout_idx = prepare_from_config(cfg)
        cache[key] = (x[holdout_idx], y[holdout_idx])
    return cache[key]


def inject(x: np.ndarray, kind: str, scale: float, seed: int) -> np.ndarray:
    """관측 R에 노이즈를 더한다 (기존 노이즈 위 **추가분**). 같은 인자면 항상 같은 실현."""
    if scale == 0.0:
        return x
    rng = np.random.default_rng(seed)
    if kind == "uniform":  # ±scale
        noise = rng.uniform(-scale, scale, size=x.shape)
    elif kind == "gaussian":  # 표준편차 scale
        noise = rng.normal(0.0, scale, size=x.shape)
    else:
        raise ValueError(f"모르는 노이즈 종류: {kind!r}")
    return (x + noise).astype(np.float32)


def total_sigma(kind: str, scale: float) -> float:
    """주입 후 총 노이즈 σ — 데이터의 σ와 독립이라 제곱합으로 더한다."""
    added = scale / np.sqrt(3.0) if kind == "uniform" else scale
    return float(np.hypot(NOISE_SIGMA, added))


def noise_curve(
    runs: list[dict[str, Any]], cache: dict[tuple, tuple], seed: int, rows: int | None
) -> list[dict[str, Any]]:
    """주입 수준별 holdout MAE. 노이즈 실현은 수준마다 고정이라 run 사이 비교가 짝지어진다."""
    levels = [("uniform", a) for a in UNIFORM_LEVELS]
    levels += [("gaussian", m * NOISE_SIGMA) for m in GAUSSIAN_LEVELS]

    out: list[dict[str, Any]] = []
    for i, (kind, scale) in enumerate(levels):
        row: dict[str, Any] = {"kind": kind, "scale": scale, "sigma": total_sigma(kind, scale)}
        for run in runs:
            x, y = holdout_of(run["cfg"], cache)
            if rows is not None:
                x, y = x[:rows], y[:rows]
            # 수준마다 seed를 달리해 서로 다른 실현을 쓰되, run 사이에는 동일하게 유지한다
            noisy = inject(x, kind, scale, seed + i)
            row[run["name"]] = mae_per_layer(predict(run["model"], noisy), y)["overall"]
        out.append(row)
    return out


def confidence_metrics(
    run: dict[str, Any], decoder: FrozenDecoder, cache: dict[tuple, tuple], rows: int | None
) -> dict[str, Any]:
    """행별 물리 잔차 ↔ 실제 오차. 라벨 없이 계산되는 지표가 오차를 얼마나 짚어내는가."""
    x, y = holdout_of(run["cfg"], cache)
    if rows is not None:
        x, y = x[:rows], y[:rows]
    pred = predict(run["model"], x)
    x_t = torch.from_numpy(x)
    residual = decoder.residual_l1(torch.from_numpy(pred), x_t).numpy()
    error = np.abs(pred - y).mean(axis=1)  # 행별 MAE [nm]

    order = np.argsort(residual)
    deciles = np.array_split(order, 10)
    decile_mae = [float(error[idx].mean()) for idx in deciles]
    # 잔차 상위 10%가 실제 오차 상위 10%를 얼마나 잡아내는가 (무작위면 10%)
    worst_by_residual = set(deciles[-1].tolist())
    worst_by_error = set(np.argsort(error)[-len(deciles[-1]) :].tolist())
    capture = len(worst_by_residual & worst_by_error) / len(worst_by_residual)

    return {
        "name": run["name"],
        "spearman": float(spearmanr(residual, error).statistic),
        "decile_mae": decile_mae,
        "capture_top10": float(capture),
        "residual_median": float(np.median(residual)),
        # 참 두께를 넣었을 때의 잔차 — 이 지표가 도달할 수 있는 바닥 (모델 오차 0에 해당)
        "residual_at_truth": float(decoder.residual_l1(torch.from_numpy(y), x_t).mean()),
    }


def render(
    runs: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    confidence: list[dict[str, Any]],
    decoder: FrozenDecoder,
    rows: int | None,
) -> list[str]:
    names = [run["name"] for run in runs]
    n_rows = rows if rows is not None else len(runs[0]["holdout_y"])
    lines = [
        f"# 평가 축 — 노이즈 강건성 · 신뢰도 지표 ({runs[0]['experiment']})",
        "",
        "`scripts/evaluate_axes.py` 산출 — 재실행 시 덮어쓴다. 해석은 리포트 본문에서 한다.",
        "",
        f"- 대상 run {len(runs)}개, holdout {n_rows:,}행",
        f"- 디코더 `{decoder.provenance['decoder']}` (Stage A 확정, 동결)",
        f"- 데이터에 이미 있는 노이즈 σ = {NOISE_SIGMA} — 아래 주입은 **그 위의 추가분**이다",
        "",
        "## 1. 노이즈 강건성 (holdout MAE [nm])",
        "",
        "주입 수준마다 노이즈 실현을 고정해 모든 run이 **같은 입력**을 본다 — run 사이 차이는",
        "난수가 아니라 모델에 귀속된다. 균등이 데이터와 같은 종류이고, 가우시안은 별개 질문이다.",
        "",
        "| 주입 | 총 σ | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for row in curve:
        if row["scale"] == 0.0:
            label = "없음 (원본)"
        elif row["kind"] == "uniform":
            label = f"균등 ±{row['scale']:.4f}"
        else:
            label = f"가우시안 σ={row['scale']:.6f} ({row['scale'] / NOISE_SIGMA:.1f}×σ)"
        cells = " | ".join(f"{row[name]:.4f}" for name in names)
        lines.append(f"| {label} | {row['sigma']:.6f} | {cells} |")

    lines += [
        "",
        "## 2. 신뢰도 지표 — 행별 물리 잔차 vs 실제 오차",
        "",
        "잔차는 라벨을 쓰지 않는다 (예측 두께를 디코더로 되비춰 관측과 비교). 순위상관이 높고",
        "잔차 상위 10%의 실제 오차가 하위 10%보다 크면 계측 이상 감지로 쓸 수 있다. 바닥은",
        f"참 두께에서의 잔차이며 균등 노이즈만 남으면 E|ε| = {NOISE_MEAN_ABS}이다.",
        "",
        "| run | Spearman ρ | 잔차 최저 10% MAE | 최고 10% MAE | 상위 10% 포착률 | 잔차 중앙값 |",
        "|---|---|---|---|---|---|",
    ]
    for c in confidence:
        lines.append(
            f"| {c['name']} | {c['spearman']:.4f} | {c['decile_mae'][0]:.3f} |"
            f" {c['decile_mae'][-1]:.3f} | {c['capture_top10']:.1%} | {c['residual_median']:.6f} |"
        )

    sharpest = max(confidence, key=lambda c: c["spearman"])
    lines += [
        "",
        f"참 두께에서의 잔차(지표 바닥) = {confidence[0]['residual_at_truth']:.6f}",
        f" — 순수 노이즈 하한 {NOISE_MEAN_ABS}보다 큰 만큼이 Stage A forward 모델의 잔여",
        "계통오차를 반영한다 (노이즈와 합성된 값이라 계통오차 자체와 같지는 않다).",
        "",
        f"### 잔차 십분위별 실제 MAE — `{sharpest['name']}` (순위상관이 가장 높은 run)",
        "",
        "| 십분위 | " + " | ".join(str(i) for i in range(1, 11)) + " |",
        "|---|" + "---|" * 10,
        "| MAE [nm] | " + " | ".join(f"{v:.3f}" for v in sharpest["decile_mae"]) + " |",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="평가 축 — 노이즈 강건성·신뢰도 지표")
    parser.add_argument(
        "--run", action="append", required=True, help="runs/<실험>/<변형> (여러 번 지정 가능)"
    )
    parser.add_argument("--out", default=None, help="리포트 경로 (기본 reports/<실험>_axes.md)")
    parser.add_argument("--seed", type=int, default=0, help="노이즈 주입 시드")
    parser.add_argument(
        "--rows", type=int, default=None, help="holdout 앞에서 N행만 (빠른 확인용, 기본 전체)"
    )
    parser.add_argument("--decoder", default=DEFAULT_DECODER, help="동결 디코더 체크포인트")
    args = parser.parse_args()

    cache: dict[tuple, tuple] = {}
    runs: list[dict[str, Any]] = []
    for run_path in args.run:
        run_dir = Path(run_path)
        model, metrics = load_run(run_dir)
        cfg = metrics["config"]
        x_h, y_h = holdout_of(cfg, cache)
        runs.append(
            {
                "name": metrics["run_name"],
                "experiment": metrics["experiment"],
                "model": model,
                "cfg": cfg,
                "holdout_y": y_h,
            }
        )
        recorded = metrics["model"]["val_mae"]
        again = mae_per_layer(predict(model, x_h), y_h)["overall"]
        print(f"{metrics['run_name']:24s} 기록 {recorded:.4f} / 재추론 {again:.4f} nm")
        if abs(again - recorded) > 1e-3:
            raise SystemExit(
                f"체크포인트가 기록된 holdout MAE를 재현하지 못한다 ({run_dir}) — 손상 의심"
            )

    decoder = FrozenDecoder(args.decoder)
    curve = noise_curve(runs, cache, args.seed, args.rows)
    confidence = [confidence_metrics(run, decoder, cache, args.rows) for run in runs]

    out_path = (
        Path(args.out) if args.out else REPO_ROOT / "reports" / f"{runs[0]['experiment']}_axes.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(render(runs, curve, confidence, decoder, args.rows)) + "\n")
    print(f"\n산출물: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
