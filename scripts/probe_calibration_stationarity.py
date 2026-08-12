"""캘리브레이션 해의 1차 정류점 검사 — "옵티마이저 한계 vs 모델족 한계" 판별의 최종 근거.

주어진 run의 해에서 fit 표본 전체(run과 동일한 시드·표집)의 전배치 기울기를 재고,
−grad 방향의 손실 단면을 훑는다. |g| ≈ 0 이고 어떤 보폭도 손실을 의미 있게 못 내리면
그 해는 해당 파라미터화(모델족)의 국소 최적값이다 — 남은 오차를 옵티마이저 탓으로
돌릴 수 없다 (sio2-freeze-adachi의 0.01442 정체 판별에 사용, 2026-08-12).

사용:
    python scripts/probe_calibration_stationarity.py --run runs/stage_a/sio2-freeze-adachi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import mse_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibrate import load_calibrated_stack  # noqa: E402
from src.data.dataset import REPO_ROOT, prepare_train_arrays  # noqa: E402

PROBE_STEPS = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0)
CHUNK = 8192


def main() -> int:
    parser = argparse.ArgumentParser(description="캘리브레이션 해의 정류점 검사")
    parser.add_argument("--run", default="runs/stage_a/sio2-freeze-adachi", help="run 디렉토리")
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    metrics = json.loads((run_dir / "metrics.json").read_text())
    seed = int(metrics["seed"])
    data_cfg = metrics["config"]["data"]
    n_fit = int(metrics["rows"]["fit"])
    n_diag = int(metrics["rows"]["diag"])

    # run과 동일한 표집 재현 (calibrate.main과 같은 시드·순서) — fit 표본에서 잰다.
    x, y, train_idx, _ = prepare_train_arrays(val_frac=float(data_cfg["val_frac"]), seed=seed)
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=n_fit + n_diag, replace=False)
    x_fit = torch.from_numpy(x[pick[:n_fit]]).to(torch.float64)
    d_fit = torch.from_numpy(y[pick[:n_fit]]).to(torch.float64)
    del x, y

    model, ckpt = load_calibrated_stack(run_dir / "model.pt")
    params = [p for p in model.parameters() if p.requires_grad]
    n_pix = float(x_fit.numel())
    print(f"run {metrics['run_name']}: best step {ckpt['step']}, fit {n_fit:,}행에서 검사")

    for p in params:
        p.grad = None
    f0 = torch.zeros((), dtype=torch.float64)
    for s in range(0, n_fit, CHUNK):
        loss = mse_loss(model(d_fit[s : s + CHUNK]), x_fit[s : s + CHUNK], reduction="sum") / n_pix
        loss.backward()
        f0 += loss.detach()
    f0_val = float(f0)
    g = torch.cat([p.grad.reshape(-1) for p in params]).clone()
    print(f"f0 = {f0_val:.6e} (RMSE {f0_val**0.5:.7f})")
    print(f"|g|_2 = {float(g.norm()):.4e} / |g|_inf = {float(g.abs().max()):.4e}")
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        print(f"  {name:14s} |g|_inf = {float(g[offset : offset + n].abs().max()):.3e}")
        offset += n

    saved = [p.detach().clone() for p in params]
    best_gain = 0.0
    print(f"{'t':>10} {'f(t)':>16} {'Δf':>12} {'RMSE':>11}")
    for t in PROBE_STEPS:
        with torch.no_grad():
            offset = 0
            for p, s0 in zip(params, saved, strict=True):
                n = p.numel()
                p.copy_(s0 - t * g[offset : offset + n].view(p.shape))
                offset += n
            f_t = 0.0
            for s in range(0, n_fit, CHUNK):
                f_t += float(
                    mse_loss(model(d_fit[s : s + CHUNK]), x_fit[s : s + CHUNK], reduction="sum")
                )
            f_t /= n_pix
        best_gain = max(best_gain, f0_val - f_t)
        print(f"{t:10.1e} {f_t:16.10e} {f_t - f0_val:+12.3e} {f_t**0.5:11.7f}")
    with torch.no_grad():  # 원상 복구
        for p, s0 in zip(params, saved, strict=True):
            p.copy_(s0)

    d_rmse = f0_val**0.5 - max(f0_val - best_gain, 0.0) ** 0.5
    print(
        f"\n−grad 방향 최선 개선: Δf = {-best_gain:+.3e} (ΔRMSE ≈ {-d_rmse:+.2e})"
        f" → {'정류점 (모델족 국소 최적)' if d_rmse < 1e-5 else '미수렴 — 추가 최적화 여지'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
