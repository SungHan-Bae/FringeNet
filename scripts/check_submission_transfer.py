"""제출 전 관문 — raw CNN 예측의 격자 밖 전이를 라벨 없이 점검한다.

배경 (Task 8 격자 밖 반전, `reports/task8.md` 발견 ②③): train/holdout 타깃이 10 nm 격자
위에 있어 정확도가 격자 간격보다 좋아진 raw CNN은 격자 보간을 흡수한다 — holdout에서
최강(0.2954)이던 d2-se raw가 test에서 0.5461로 붕괴했다(+85%). 이 붕괴는 제출 전에 라벨
없이 보였다: 두 신호를 이 스크립트가 잰다.

  ① 격자 거리 |pred − 최근접 10nm 격자|: holdout은 참값이 격자라 작은 게 정상.
     test는 참값이 연속 — 편향 없는 모델이면 [0, 5] 균등(평균 ~2.5).
  ② 재구성 잔차 |R_dec(pred) − R_obs| 행 평균: holdout↔test **분포가 같아야** 전이가
     깨끗하다. 꼬리(p90·p99)가 2배 이상 벌어지면 격자 과적합 신호 — 제출을 중단하고
     물리 보정 파이프라인(`--submission --refine`)을 쓴다.

실측 기준점 (2026-08-20): d2-se raw p90 2.3배 → test +85% 붕괴 / d2-fft raw p90 2.0배
(미제출) / d2-fft+LM 파이프라인은 p99.9까지 일치 → test 전이 −0.02%.

사용법:
    python scripts/check_submission_transfer.py --run runs/task8/d2-fft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import REPO_ROOT, prepare_from_config, subsample_indices  # noqa: E402
from src.evaluate import load_model_checkpoint, load_test, predict  # noqa: E402
from src.losses import DEFAULT_DECODER, FrozenDecoder  # noqa: E402

TAIL_RATIO_WARN = 2.0  # test/holdout 잔차 꼬리 배수가 이 이상이면 격자 과적합 신호
GRID_NM = 10.0


def grid_dist(pred: np.ndarray) -> np.ndarray:
    """(N, L) 예측의 최근접 격자 거리 [nm] — (N, L) 반환, 값 범위 [0, 5]."""
    return np.abs(pred - GRID_NM * np.round(pred / GRID_NM))


def residual_row_l1(decoder: FrozenDecoder, x: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """행별 재구성 잔차 |R_dec(pred) − x| 평균. x (N, W), pred (N, L) → (N,)."""
    with torch.no_grad():
        r_hat = decoder(torch.from_numpy(pred.astype(np.float64))).numpy()
    return np.abs(r_hat - x).mean(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="raw CNN 제출 전 격자 밖 전이 관문")
    parser.add_argument("--run", required=True, help="run 디렉토리 (model.pt + metrics.json)")
    parser.add_argument("--rows", type=int, default=10_000, help="holdout 표본 행 수")
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.run
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    model = load_model_checkpoint(run_dir / "model.pt")

    x, y, _, holdout_idx = prepare_from_config(cfg)
    sub = subsample_indices(len(holdout_idx), args.rows, seed=0)
    x_hold, y_hold = x[holdout_idx][sub], y[holdout_idx][sub]
    del x, y
    _, x_test = load_test()

    pred_hold = predict(model, x_hold)
    pred_test = predict(model, x_test)
    mae = np.abs(pred_hold - y_hold).mean()
    print(f"{args.run} — holdout {len(x_hold):,}행 표본 CNN MAE {mae:.4f} nm\n")

    print("① 격자 거리 [nm] — 편향 없는 모델이면 test 평균 ~2.5 (참값이 연속이므로)")
    for name, p in (("holdout", pred_hold), ("test", pred_test)):
        d = grid_dist(p).ravel()
        print(
            f"   {name:7s} 평균 {d.mean():.3f}  중앙값 {np.median(d):.3f}"
            f"  p10 {np.percentile(d, 10):.3f}  p90 {np.percentile(d, 90):.3f}"
        )

    decoder = FrozenDecoder(args.decoder, dtype=torch.complex128)
    stats = {}
    print("\n② 재구성 잔차 — holdout↔test 분포가 같아야 전이가 깨끗하다")
    for name, xx, pp in (("holdout", x_hold, pred_hold), ("test", x_test, pred_test)):
        res = residual_row_l1(decoder, xx, pp)
        stats[name] = {q: float(np.percentile(res, q)) for q in (50, 90, 99)}
        print(
            f"   {name:7s} 중앙값 {stats[name][50]:.6f}  p90 {stats[name][90]:.6f}"
            f"  p99 {stats[name][99]:.6f}"
        )

    ratios = {q: stats["test"][q] / stats["holdout"][q] for q in (90, 99)}
    print(f"\n   꼬리 배수 (test/holdout): p90 {ratios[90]:.2f}배 · p99 {ratios[99]:.2f}배")
    if max(ratios.values()) >= TAIL_RATIO_WARN:
        print(
            f"   → **격자 과적합 신호** (기준 {TAIL_RATIO_WARN}배) — raw 제출을 중단하고"
            " 물리 보정 파이프라인(--submission --refine)을 쓸 것"
        )
        return 1
    print("   → 꼬리가 유지된다 — 전이 신호 정상 (그래도 정본 확인은 리더보드다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
