"""학습 엔트리포인트 — Task 4 baseline(holdout 단일) + k-fold 앙상블 학습.

프로토콜 (CLAUDE.md 평가 규약과의 관계):
  - seed 고정 holdout(기본 10%)이 프로젝트 공통 검증셋이다. 어떤 모델의 학습에도
    쓰지 않으며, 실험 간 비교(ablation)는 전부 이 셋의 raw MAE로 한다.
  - k-fold(``--folds k``)는 holdout을 뺀 나머지 90% 안에서만 접는다. 그래야
    fold 모델 누구도 보지 않은 holdout으로 앙상블을 공정하게 평가할 수 있고,
    단일 모델 대비 앙상블의 이득이 같은 잣대로 측정된다. fold 모델의 best
    체크포인트 선택은 자기 fold의 OOF 조각으로만 한다 (holdout 미사용).
  - 리포트 MAE는 raw 예측 기준. 격자 스냅은 여기서 하지 않는다
    (분리 보고가 필요하면 evaluate.py --snap).

사용:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --folds 5 --run-name mlp_cv5
    python -m src.train --config configs/baseline.yaml --subset 20000 --epochs 2 \
        --run-name smoke   # 스모크 테스트
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.nn.functional import l1_loss
from torch.optim.lr_scheduler import LambdaLR

from src.data.dataset import REPO_ROOT, kfold_indices, prepare_train_arrays
from src.evaluate import (
    format_mae,
    load_model_checkpoint,
    mae_per_layer,
    predict,
)
from src.models import build_model
from src.utils.seed import set_seed

RUNS_DIR = REPO_ROOT / "runs"


def log_line(run_dir: Path, message: str) -> None:
    """콘솔에 출력하고 같은 내용을 run_dir/train.log에도 append한다.

    학습이 중간에 죽어도 진행 기록이 run 디렉토리에 남도록 에폭마다 즉시 쓴다.
    """
    print(message, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR | None:
    """스텝 단위 LR 스케줄러를 만든다.

    Args:
        schedule: "cosine" = linear warmup 후 cosine 감쇠(끝에서 0), "none" = 고정 lr.
        warmup_steps: warmup 스텝 수. 첫 스텝이 lr=0이 되지 않도록 (step+1)/warmup을 쓴다.
        total_steps: 전체 학습 스텝 수 (= 에폭당 스텝 × 에폭).

    Raises:
        ValueError: 모르는 schedule 이름이거나 warmup_steps >= total_steps인 경우.
    """
    if schedule == "none":
        return None
    if schedule != "cosine":
        raise ValueError(f'lr_schedule은 "cosine" | "none" 이어야 한다 (받은 값: {schedule!r})')
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps는 [0, total_steps) 범위여야 한다"
            f" (받은 값: {warmup_steps}, total_steps={total_steps})"
        )

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)


def train_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    tag: str,
) -> dict[str, Any]:
    """모델 하나를 학습하고 best(val MAE 기준) 체크포인트를 저장한다.

    Args:
        x_train: (N, 226) float32 — 이 모델이 학습하는 행.
        y_train: (N, 4) float32 두께 [nm].
        x_val / y_val: best 선택용 검증 행 — holdout 모드에서는 공통 holdout,
            k-fold 모드에서는 해당 fold의 OOF 조각.
        cfg: 전체 config (model/train 섹션 사용).
        seed: 이 모델의 시드 (fold마다 다르게 주면 앙상블 다양성이 생긴다).
        run_dir: 체크포인트({tag}.pt)·history(history_{tag}.csv)·train.log 저장 위치.
            history와 로그는 에폭마다 즉시 기록된다 (중단 시에도 남는다).
        tag: 파일명 태그 ("model" 또는 "fold0" 등).

    Returns:
        {"tag", "seed", "ckpt_path", "best_epoch", "val_mae", "val_mae_per_layer",
         "val_pred" (M, 4) np.ndarray — best 모델의 x_val 예측, "wall_sec"}
    """
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    if epochs < 1:
        raise ValueError(f"epochs는 1 이상이어야 한다 (받은 값: {epochs})")

    set_seed(seed)

    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train)
    model = build_model(cfg["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    n = len(x_t)
    steps_per_epoch = math.ceil(n / batch_size)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=str(train_cfg.get("lr_schedule", "cosine")),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        total_steps=steps_per_epoch * epochs,
    )
    best_mae = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_pred: np.ndarray | None = None
    history: list[dict[str, float]] = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        perm = torch.randperm(n)
        loss_sum = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = l1_loss(model(x_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:  # warmup+cosine은 스텝 단위로 움직인다
                scheduler.step()
            loss_sum += loss.item() * len(idx)

        val_pred = predict(model, x_val)
        val_metrics = mae_per_layer(val_pred, y_val)
        row = {
            "epoch": epoch,
            "train_l1": loss_sum / n,
            "val_mae": val_metrics["overall"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.perf_counter() - t_epoch,
        }
        history.append(row)
        # 중간에 죽어도 기록이 남도록 history를 에폭마다 덮어쓴다.
        pd.DataFrame(history).to_csv(run_dir / f"history_{tag}.csv", index=False)
        marker = ""
        if val_metrics["overall"] < best_mae:
            best_mae = val_metrics["overall"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = val_metrics
            best_pred = val_pred
            marker = " *"
        log_line(
            run_dir,
            f"[{tag}] epoch {epoch:3d}/{epochs}  train_l1 {row['train_l1']:.4f}  "
            f"val_mae {row['val_mae']:.4f}  lr {row['lr']:.2e}  {row['sec']:.1f}s{marker}",
        )

    if best_state is None or best_pred is None:  # epochs >= 1 이므로 도달 불가
        raise RuntimeError("best 체크포인트가 만들어지지 않았다")

    ckpt_path = run_dir / f"{tag}.pt"
    torch.save(
        {
            "model_cfg": dict(cfg["model"]),
            "state_dict": best_state,
            "seed": seed,
            "tag": tag,
            "best_epoch": best_epoch,
            "val_mae": best_mae,
        },
        ckpt_path,
    )
    return {
        "tag": tag,
        "seed": seed,
        "ckpt_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "val_mae": best_mae,
        "val_mae_per_layer": best_metrics,
        "val_pred": best_pred,
        "wall_sec": time.perf_counter() - t_start,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FringeNet 학습")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--epochs", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--batch-size", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--lr", type=float, default=None, help="config 덮어씀")
    parser.add_argument("--weight-decay", type=float, default=None, help="config 덮어씀")
    parser.add_argument(
        "--hidden-dims",
        default=None,
        help='은닉 블록 폭, 쉼표 구분 (예: "256,256,256") — config 덮어씀',
    )
    parser.add_argument(
        "--folds", type=int, default=None, help="0=holdout 단일, k>=2 = k-fold (config 덮어씀)"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="시드 고정 무작위 서브셋 크기 (스모크용)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.hidden_dims is not None:
        cfg["model"]["hidden_dims"] = [int(w) for w in args.hidden_dims.split(",")]
    if args.folds is not None:
        cfg["train"]["num_folds"] = args.folds
    if args.subset is not None:
        cfg.setdefault("data", {})["subset"] = args.subset

    seed = int(cfg["seed"])
    set_seed(seed)
    data_cfg = cfg.get("data", {})
    x, y, train_idx, holdout_idx = prepare_train_arrays(
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        seed=seed,
        subset=data_cfg.get("subset"),
    )
    run_dir = RUNS_DIR / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # evaluate.py가 같은 분할을 재현할 수 있도록 CLI 오버라이드가 반영된 config를 남긴다.
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    x_hold, y_hold = x[holdout_idx], y[holdout_idx]
    n_folds = int(cfg["train"].get("num_folds", 0))
    mode = "kfold" if n_folds >= 2 else "holdout"
    log_line(
        run_dir,
        f"run {cfg['run_name']}: 행 {len(x):,} = 학습 {len(train_idx):,}"
        f" + holdout {len(holdout_idx):,} / mode={mode}"
        + (f" (k={n_folds})" if mode == "kfold" else ""),
    )

    metrics: dict[str, Any] = {
        "run_name": cfg["run_name"],
        "seed": seed,
        "mode": mode,
        "rows": {"train": int(len(train_idx)), "holdout": int(len(holdout_idx))},
        "config": cfg,
    }

    if mode == "holdout":
        result = train_one_model(
            x[train_idx], y[train_idx], x_hold, y_hold, cfg, seed, run_dir, "model"
        )
        result.pop("val_pred")
        metrics["model"] = result  # 이 모드에서 val == holdout
        log_line(
            run_dir,
            f"\n[model] holdout {format_mae(result['val_mae_per_layer'])}"
            f" (best epoch {result['best_epoch']})",
        )
    else:
        oof_pred = np.full((len(x), 4), np.nan, dtype=np.float32)
        fold_rows: list[dict[str, Any]] = []
        holdout_preds: list[np.ndarray] = []
        for i, (fold_train, fold_val) in enumerate(kfold_indices(train_idx, n_folds, seed)):
            result = train_one_model(
                x[fold_train],
                y[fold_train],
                x[fold_val],
                y[fold_val],
                cfg,
                seed + i,
                run_dir,
                f"fold{i}",
            )
            oof_pred[fold_val] = result.pop("val_pred")
            model = load_model_checkpoint(result["ckpt_path"])
            hold_pred = predict(model, x_hold)
            holdout_preds.append(hold_pred)
            result["holdout_mae"] = mae_per_layer(hold_pred, y_hold)
            fold_rows.append(result)
            log_line(
                run_dir,
                f"[fold{i}] oof {format_mae(result['val_mae_per_layer'])}"
                f" / holdout {format_mae(result['holdout_mae'])}",
            )

        ensemble_pred = np.mean(holdout_preds, axis=0)
        metrics["folds"] = fold_rows
        metrics["oof_mae"] = mae_per_layer(oof_pred[train_idx], y[train_idx])
        metrics["singles_holdout_overall"] = [r["holdout_mae"]["overall"] for r in fold_rows]
        metrics["ensemble_holdout_mae"] = mae_per_layer(ensemble_pred, y_hold)
        singles_mean = float(np.mean(metrics["singles_holdout_overall"]))
        log_line(run_dir, f"\nOOF(단일 모델, 90% 전체) {format_mae(metrics['oof_mae'])}")
        log_line(run_dir, f"단일 모델 holdout 평균 MAE {singles_mean:.4f} nm")
        log_line(
            run_dir, f"앙상블(k={n_folds}) holdout {format_mae(metrics['ensemble_holdout_mae'])}"
        )
        log_line(run_dir, f"제출 파일 생성: python -m src.evaluate --run {run_dir} --submission")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    log_line(run_dir, f"\n산출물: {run_dir}/ (metrics.json, history_*.csv, train.log, *.pt)")


if __name__ == "__main__":
    main()
