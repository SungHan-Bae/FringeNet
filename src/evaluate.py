"""평가·추론 — 저장된 run의 체크포인트로 holdout MAE 리포트와 제출 파일을 만든다.

사용:
    python -m src.evaluate --run runs/mlp_baseline                  # holdout MAE 재계산
    python -m src.evaluate --run runs/mlp_cv5 --submission          # 앙상블 test 추론 -> csv
    python -m src.evaluate --run runs/mlp_cv5 --submission --snap   # + 격자 스냅본(분리 보고)

체크포인트가 여러 개면(fold*.pt) 예측 평균 앙상블로 추론한다.

규약(CLAUDE.md 평가 규약): 기본 리포트는 raw 예측 MAE다. 격자 스냅(--snap)은 타깃이
10 nm 격자 위에 있다는 **생성 방식의 누설**을 이용하는 것이라 주 결과로 쓰지 않으며,
항상 별도 파일·별도 행으로 분리 표기한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.data.dataset import LAYER_COLS, RAW_DIR, load_test, prepare_train_arrays
from src.models import build_model


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """배치 추론. x (N, 226) float32 반사율 원값 -> 예측 (N, 4) float32 [nm].

    입력 표준화는 하지 않는다 — 모델이 자체 norm 층(batchnorm/layernorm)으로 처리한다.
    """
    model.eval()
    outs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size])
        outs.append(model(xb).numpy())
    return np.concatenate(outs, axis=0)


def mae_per_layer(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """(N, 4) 예측·정답의 층별 MAE와 overall(층 평균) [nm]."""
    err = np.abs(pred - target).mean(axis=0)
    out = {col: float(e) for col, e in zip(LAYER_COLS, err, strict=True)}
    out["overall"] = float(err.mean())
    return out


def format_mae(metrics: dict[str, float]) -> str:
    """mae_per_layer 결과를 로그 한 줄로 만든다."""
    layers = "  ".join(f"{col}={metrics[col]:.3f}" for col in LAYER_COLS)
    return f"MAE {metrics['overall']:.4f} nm  ({layers})"


def load_model_checkpoint(path: Path | str) -> nn.Module:
    """train.py가 저장한 체크포인트에서 모델을 복원한다 (eval 모드)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def find_checkpoints(run_dir: Path) -> list[Path]:
    """run 디렉토리의 체크포인트 목록 (단일 model.pt 또는 fold*.pt, 이름순)."""
    paths = sorted(run_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"{run_dir} 에 체크포인트(*.pt)가 없다")
    return paths


def snap_to_grid(
    pred: np.ndarray, lo: float = 10.0, hi: float = 300.0, step: float = 10.0
) -> np.ndarray:
    """예측을 두께 격자로 반올림한다 — 생성 방식 **누설**이라 분리 보고 전용."""
    return np.clip(np.round(pred / step) * step, lo, hi)


def build_submission_frame(ids: np.ndarray, pred: np.ndarray, sample_path: Path) -> pd.DataFrame:
    """sample_submission 형식(컬럼·id 순서)에 예측을 맞춘 DataFrame을 만든다.

    Args:
        ids: (N,) test id — pred의 행과 같은 순서.
        pred: (N, 4) 두께 예측 [nm].
        sample_path: sample_submission.csv 경로 (id 순서·컬럼 순서의 기준).
    """
    sample = pd.read_csv(sample_path)
    frame = pd.DataFrame({"id": ids})
    frame[LAYER_COLS] = pred.astype(np.float64)
    out = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    if out[LAYER_COLS].isna().to_numpy().any():
        raise ValueError("sample_submission의 id 중 예측이 없는 것이 있다")
    return out[list(sample.columns)]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FringeNet 평가·추론")
    parser.add_argument("--run", required=True, help="runs/<이름> 디렉토리 경로")
    parser.add_argument("--submission", action="store_true", help="test 추론 후 제출 csv 생성")
    parser.add_argument(
        "--snap", action="store_true", help="격자 스냅본도 생성/보고 (누설 — 분리 보고 전용)"
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run)
    cfg: dict[str, Any] = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt_paths = find_checkpoints(run_dir)
    models = [load_model_checkpoint(p) for p in ckpt_paths]
    print(f"run {run_dir.name}: 체크포인트 {len(models)}개 — {[p.name for p in ckpt_paths]}")

    # 학습 때와 동일한 seed/subset/val_frac으로 holdout을 재현한다.
    data_cfg = cfg.get("data", {})
    x, y, _, holdout_idx = prepare_train_arrays(
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        seed=int(cfg["seed"]),
        subset=data_cfg.get("subset"),
    )
    x_hold, y_hold = x[holdout_idx], y[holdout_idx]
    preds = [predict(m, x_hold, args.batch_size) for m in models]
    for path, pred in zip(ckpt_paths, preds, strict=True):
        print(f"  {path.name:10s} holdout {format_mae(mae_per_layer(pred, y_hold))}")
    ensemble = np.mean(preds, axis=0)
    if len(preds) > 1:
        print(f"  {'ensemble':10s} holdout {format_mae(mae_per_layer(ensemble, y_hold))}")
    if args.snap:
        snapped = mae_per_layer(snap_to_grid(ensemble), y_hold)
        print(f"  [분리 보고·누설] 격자 스냅 holdout {format_mae(snapped)} — 주 결과 아님")

    if args.submission:
        ids, x_test = load_test()
        test_pred = np.mean([predict(m, x_test, args.batch_size) for m in models], axis=0)
        sample_path = RAW_DIR / "sample_submission.csv"
        out_path = run_dir / f"submission_{run_dir.name}.csv"
        build_submission_frame(ids, test_pred, sample_path).to_csv(out_path, index=False)
        print(f"제출 파일(raw): {out_path}")
        if args.snap:
            snap_path = run_dir / f"submission_{run_dir.name}_snap.csv"
            snapped_pred = snap_to_grid(test_pred)
            build_submission_frame(ids, snapped_pred, sample_path).to_csv(snap_path, index=False)
            print(f"제출 파일(격자 스냅 — 누설, 분리 보고 전용): {snap_path}")


if __name__ == "__main__":
    main()
