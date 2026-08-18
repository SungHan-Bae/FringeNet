"""예측 오차 구조 진단 — 상관·예측 std·범위 밖 비율·두께 구간별 MAE를 run별로 나란히.

`reports/level1_cnn.md` §3·§4가 인용하는 진단 수치의 산출 경로다 — 리포트에 손으로 적힌
수치는 재현도 갱신도 안 되므로, 정본은 이 스크립트의 산출물로 둔다 (기록 규약).

산출물:
  reports/level1_cnn_diagnostics.md   (재실행 시 덮어씀)

사용법:
    python scripts/diagnose_predictions.py                       # 기본 3 run (아래 DEFAULT_RUNS)
    python scripts/diagnose_predictions.py --run runs/<실험>/<변형> [--run ...]

체크포인트가 로컬에 없으면 git 히스토리에서 복구한다 (`runs/CHECKPOINTS.md`):
    git show 2a2ba56:runs/<실험>/<run>/model.pt > runs/<실험>/<run>/model.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_axes import holdout_of, load_run  # noqa: E402

from src.data.dataset import REPO_ROOT  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "level1_cnn_diagnostics.md"
DEFAULT_RUNS = (
    "runs/mlp_baseline/dropout0.0",
    "runs/level1_cnn/flatten-dilated",
    "runs/level1_cnn/flatten-dilated-bound",
)
# 물리 범위와 격자 [nm] — train 격자는 {10, 20, …, 300}이므로 끝값은 10·300이다.
PHYS_LO, PHYS_HI = 10.0, 300.0
THIN = (10.0, 60.0)  # 가장 어려운 얇은 구간
INNER_BAND = (70.0, 240.0)  # 구간별 표의 내부
INNER_ENDS = (20.0, 290.0)  # 격자 끝 대비의 내부


def diagnose_one(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """한 run의 오차 구조. x: (N, W) 관측, y: (N, L) 참 두께 [nm].

    Returns:
        corr/std는 층별 (L,), MAE류는 (참값, 층) 쌍 단위 스칼라.
    """
    device = next((t.device for t in (*model.parameters(), *model.buffers())), torch.device("cpu"))
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)

    err = np.abs(pred - y)  # (N, L)
    corr = np.array([float(np.corrcoef(pred[:, j], y[:, j])[0, 1]) for j in range(y.shape[1])])
    return {
        "corr": corr,
        "pred_std": pred.std(axis=0),
        "below_lo": float((pred < PHYS_LO).mean()),
        "above_hi": float((pred > PHYS_HI).mean()),
        "mae": float(err.mean()),
        "mae_at_lo": float(err[y == PHYS_LO].mean()),
        "mae_at_hi": float(err[y == PHYS_HI].mean()),
        "mae_inner_ends": float(err[(y >= INNER_ENDS[0]) & (y <= INNER_ENDS[1])].mean()),
        "mae_thin": float(err[(y >= THIN[0]) & (y <= THIN[1])].mean()),
        "mae_inner_band": float(err[(y >= INNER_BAND[0]) & (y <= INNER_BAND[1])].mean()),
    }


def render(rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    lines = [
        "# level1_cnn 오차 구조 진단 — 상관·범위 밖·두께 구간별 MAE",
        "",
        "`scripts/diagnose_predictions.py` 산출 — 재실행 시 덮어쓴다."
        " 해석은 `reports/level1_cnn.md`.",
        "",
        f"- holdout 전체 {meta['rows']:,}행 · 모든 run이 같은 행을 본다 · 장치 {meta['device']}"
        f" ({meta['device_name']})",
        f"- 범위 밖 = 예측값 < {PHYS_LO:g} 또는 > {PHYS_HI:g} nm (값 단위 비율)."
        " 구간별 MAE는 (참값, 층) 쌍 단위다.",
        "",
        "| run | val MAE | 상관 min~max | 예측 std min~max | <10 nm | >300 nm |"
        f" d=10 | d=300 | 내부 {INNER_ENDS[0]:g}~{INNER_ENDS[1]:g} |"
        f" 얇은 {THIN[0]:g}~{THIN[1]:g} | 내부 {INNER_BAND[0]:g}~{INNER_BAND[1]:g} |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['val_mae']:.4f} | {r['corr'].min():.4f}~{r['corr'].max():.4f} |"
            f" {r['pred_std'].min():.1f}~{r['pred_std'].max():.1f} |"
            f" {r['below_lo']:.2%} | {r['above_hi']:.2%} |"
            f" {r['mae_at_lo']:.2f} | {r['mae_at_hi']:.2f} | {r['mae_inner_ends']:.2f} |"
            f" {r['mae_thin']:.2f} | {r['mae_inner_band']:.2f} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="예측 오차 구조 진단 (run별 나란히)")
    parser.add_argument("--run", action="append", help="run 디렉토리 (반복 가능, 기본 3 run)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help=f"리포트 경로 (기본 {OUT_PATH})")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in (args.run or DEFAULT_RUNS)]
    dev = torch.device(args.device)
    loaded = [(path, *load_run(path)) for path in run_dirs]
    blocks = {json.dumps(m["config"].get("data"), sort_keys=True) for _, _, m in loaded}
    if len(blocks) > 1:
        raise ValueError(f"run마다 split이 다르다 — 나란히 비교할 수 없다: {sorted(blocks)}")

    x, y = holdout_of(loaded[0][2]["config"], {})
    y = y.astype(np.float64)

    rows: list[dict[str, Any]] = []
    for _path, model, metrics in loaded:
        stats = diagnose_one(model.to(dev), x, y)
        rows.append(
            {
                "run": f"{metrics['experiment']}/{metrics['run_name']}",
                "val_mae": metrics["model"]["val_mae"],
                **stats,
            }
        )
        print(
            f"  {rows[-1]['run']:44s} MAE {stats['mae']:.4f}  범위 밖 "
            f"{stats['below_lo']:.2%}/{stats['above_hi']:.2%}"
        )

    meta = {"rows": len(x), "device": str(dev), "device_name": platform.machine()}
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.write_text("\n".join(render(rows, meta)) + "\n")
    print(f"\n산출물: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
