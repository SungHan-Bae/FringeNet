"""GPU(CUDA) 학습 엔트리포인트 — Colab 등 GPU 런타임용. CPU 파이프라인과 디커플.

src/train.py는 baseline(Task 4)을 만든 CPU 검증 경로 그대로 보존하고 수정하지 않는다.
이 모듈은 같은 학습 프로토콜·같은 산출물 계약을 따르는 별도 GPU 경로다:

- 산출물: runs/<experiment>/<run_name>/{model.pt, train.log, metrics.json} — CPU와 동일.
  metrics.json에 "device" 필드가 추가된다는 것만 다르다.
- 데이터 전체를 GPU에 상주시킨다 (train x 810k×226 float32 ≈ 0.69 GB — T4 16 GB에 여유).
  DataLoader 없이 GPU 텐서 인덱싱으로 배치를 뽑아 호스트-디바이스 복사를 없앤다.
- 체크포인트 state_dict는 **CPU 텐서로 변환해 저장**한다 — 로컬(CPU 전용)에서
  evaluate.py / load_model_checkpoint로 바로 분석할 수 있게.
- holdout 단일 모드만 지원한다. k-fold가 필요해지면 그때 추가한다.
- 재현성: 같은 시드·같은 GPU 기종에서는 cudnn deterministic(set_seed)으로 재현된다.
  단 CPU와 GPU는 부동소수 연산 순서가 달라 bit 단위로 같지 않다 — CPU baseline과의
  비교는 MAE 수준에서 한다.

사용 (Colab 노트북 notebooks/colab_train.ipynb가 이 함수를 호출한다):
    from src.train_gpu import run_config
    metrics = run_config("configs/level1_cnn/single-scale.yaml")

    # CLI로도 동작한다
    python -m src.train_gpu --config configs/level1_cnn/single-scale.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.functional import l1_loss

from src.data.dataset import REPO_ROOT, prepare_train_arrays
from src.evaluate import format_mae, mae_per_layer
from src.models import build_model
from src.train import RUNS_DIR, build_lr_scheduler, log_line
from src.utils.seed import set_seed


def resolve_device(device: str | None = None) -> torch.device:
    """device 문자열을 torch.device로 푼다. None이면 cuda 우선 자동 선택."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda를 요청했지만 사용할 수 없다 — 런타임에 GPU가 붙어 있는지 확인")
    return resolved


@torch.no_grad()
def predict_on_device(model: torch.nn.Module, x: Tensor, batch_size: int = 8192) -> np.ndarray:
    """디바이스 상주 텐서 x (N, 226)를 배치 추론해 (N, 4) float32 numpy로 돌려준다."""
    model.eval()
    outs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        outs.append(model(x[start : start + batch_size]).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_one_model_gpu(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    tag: str,
    device: torch.device,
) -> dict[str, Any]:
    """모델 하나를 device에서 학습하고 best(val MAE) 체크포인트를 저장한다.

    src/train.py의 train_one_model과 같은 입출력 계약(반환 키·체크포인트 포맷·로그
    형식)을 따른다. 차이는 (1) 데이터·모델이 device에 상주, (2) 저장 전 state_dict를
    CPU로 옮긴다는 것뿐.

    Returns:
        {"tag", "seed", "ckpt_path", "best_epoch", "val_mae", "val_mae_per_layer",
         "val_pred" (M, 4) np.ndarray, "wall_sec"}
    """
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    if epochs < 1:
        raise ValueError(f"epochs는 1 이상이어야 한다 (받은 값: {epochs})")

    set_seed(seed)

    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    x_v = torch.from_numpy(x_val).to(device)
    model = build_model(cfg["model"]).to(device)
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
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        perm = torch.randperm(n, device=device)
        loss_sum = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = l1_loss(model(x_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_sum += loss.item() * len(idx)

        val_pred = predict_on_device(model, x_v)
        val_metrics = mae_per_layer(val_pred, y_val)
        row = {
            "epoch": epoch,
            "train_l1": loss_sum / n,
            "val_mae": val_metrics["overall"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.perf_counter() - t_epoch,
        }
        marker = ""
        if val_metrics["overall"] < best_mae:
            best_mae = val_metrics["overall"]
            # copy=True: CPU 텐서여도 참조가 아닌 복사본을 남긴다 (이후 학습이 덮어쓰지 않게)
            state = model.state_dict()
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in state.items()}
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
    if ckpt_path.is_relative_to(REPO_ROOT):  # metrics.json에 로컬 절대경로가 남지 않게
        ckpt_path = ckpt_path.relative_to(REPO_ROOT)
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


def run_config(
    config_path: str | Path,
    *,
    device: str | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    subset: int | None = None,
) -> dict[str, Any]:
    """config 하나를 GPU(가능하면)에서 학습하고 metrics dict를 돌려준다.

    노트북에서 호출하는 진입점. 키워드 인자는 config 값을 덮어쓴다
    (src/train.py의 CLI 오버라이드와 같은 의미).

    Raises:
        ValueError: config가 k-fold 모드(num_folds >= 2)인 경우 —
            GPU 경로는 holdout 단일 모드만 지원한다.
    """
    cfg: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
    if run_name is not None:
        cfg["run_name"] = run_name
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if batch_size is not None:
        cfg["train"]["batch_size"] = batch_size
    if lr is not None:
        cfg["train"]["lr"] = lr
    if weight_decay is not None:
        cfg["train"]["weight_decay"] = weight_decay
    if subset is not None:
        cfg.setdefault("data", {})["subset"] = subset

    if int(cfg["train"].get("num_folds", 0)) >= 2:
        raise ValueError("GPU 경로는 holdout 단일 모드만 지원한다 — k-fold는 src/train.py 참조")

    experiment = cfg.get("experiment")
    if not experiment:
        raise ValueError(
            'config에 "experiment" 키(대실험 주제)가 필요하다 — runs/<experiment>/<run_name> 구조'
        )

    dev = resolve_device(device)
    seed = int(cfg["seed"])
    set_seed(seed)
    data_cfg = cfg.get("data", {})
    x, y, train_idx, holdout_idx = prepare_train_arrays(
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        seed=seed,
        subset=data_cfg.get("subset"),
    )

    run_dir = RUNS_DIR / str(experiment) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_line(
        run_dir,
        f"run {experiment}/{cfg['run_name']}: 행 {len(x):,} = 학습 {len(train_idx):,}"
        f" + holdout {len(holdout_idx):,} / mode=holdout / device={dev}",
    )

    metrics: dict[str, Any] = {
        "experiment": experiment,
        "run_name": cfg["run_name"],
        "seed": seed,
        "mode": "holdout",
        "device": str(dev),
        "rows": {"train": int(len(train_idx)), "holdout": int(len(holdout_idx))},
        "config": cfg,
    }
    # 시작 시점 설정 스냅샷 (중단돼도 남는다) — 완료 시 결과를 합쳐 덮어쓴다
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    result = train_one_model_gpu(
        x[train_idx],
        y[train_idx],
        x[holdout_idx],
        y[holdout_idx],
        cfg,
        seed,
        run_dir,
        "model",
        dev,
    )
    result.pop("val_pred")
    metrics["model"] = result
    log_line(
        run_dir,
        f"\n[model] holdout {format_mae(result['val_mae_per_layer'])}"
        f" (best epoch {result['best_epoch']})",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    log_line(run_dir, f"\n산출물: {run_dir}/ (metrics.json, train.log, model.pt)")
    return metrics


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FringeNet GPU 학습 (holdout 단일 모드)")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--device", default=None, help='"cuda" | "cpu" (기본: cuda 우선 자동)')
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--epochs", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--batch-size", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--lr", type=float, default=None, help="config 덮어씀")
    parser.add_argument("--weight-decay", type=float, default=None, help="config 덮어씀")
    parser.add_argument(
        "--subset", type=int, default=None, help="시드 고정 무작위 서브셋 크기 (스모크용)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_config(
        args.config,
        device=args.device,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
