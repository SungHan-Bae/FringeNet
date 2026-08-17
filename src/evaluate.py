"""평가·추론 — 저장된 run의 체크포인트로 holdout MAE 리포트와 제출 파일을 만든다.

사용:
    python -m src.evaluate --run runs/mlp_baseline/dropout0.0            # holdout MAE 재계산
    python -m src.evaluate --run runs/mlp_baseline/cv5 --submission      # 앙상블 test 추론 -> csv
    python -m src.evaluate --run runs/mlp_baseline/cv5 --submission --snap  # + 격자 스냅본
    python -m src.evaluate --run runs/cnn_recipe/budget100 --submission --refine  # 물리 보정 제출

run 디렉토리에는 train.py가 남긴 metrics.json(설정 스냅샷 포함)과 체크포인트(*.pt)가 있어야 한다.

체크포인트가 여러 개면(fold*.pt) 예측 평균 앙상블로 추론한다.

규약(CLAUDE.md 평가 규약): 기본 리포트는 raw 예측 MAE다. 격자 스냅(--snap)은 타깃이
10 nm 격자 위에 있다는 **생성 방식의 누설**을 이용하는 것이라 주 결과로 쓰지 않으며,
항상 별도 파일·별도 행으로 분리 표기한다. **제출에는 절대 쓰지 않는다** — test 두께는 격자
밖이라 MAE가 약 +1.2 nm 나빠진다.

`--refine`은 동결 TMM 디코더로 추론 후 물리 보정을 하고(LM 역해 + 라벨 없는 되돌림 규칙)
그 결과로 제출 파일을 만든다 — `src.physics.invert.refine_with_fallback`이 정의이고 판정
스크립트(`scripts/judge_recipe.py`)와 **같은 함수를 쓴다**. 라벨을 안 쓰므로 test에 적용
가능하다는 것이 이 경로의 요점이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.dataset import LAYER_COLS, RAW_DIR, load_test, prepare_from_config
from src.models import build_model
from src.physics.invert import PHYSICAL_RANGE_NM, refine_with_fallback


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
        "--snap",
        action="store_true",
        help="격자 스냅본도 생성/보고 (holdout에서는 누설 — 분리 보고 전용. "
        "**제출에는 쓰지 말 것**: test 두께는 격자 밖이라 MAE가 약 +1.2 nm 나빠진다)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="추론 후 물리 보정 (LM 역해 + 라벨 없는 되돌림 규칙). holdout에 리포트하고 "
        "--submission과 함께 쓰면 보정본 제출 파일도 만든다",
    )
    parser.add_argument(
        "--refine-k-sigma",
        type=float,
        default=5.0,
        help="되돌림 문턱 배수 — **사전등록 값이다** (성능을 보고 고치지 말 것)",
    )
    parser.add_argument(
        "--refine-clip",
        action="store_true",
        help=f"보정 결과를 물리 범위 {PHYSICAL_RANGE_NM} nm로 클리핑한 판본도 만든다. "
        "LM 상자를 넓게 두는 것은 라벨 사전지식을 역해에 넣지 않으려는 결정이므로 "
        "**기본은 클리핑 없음**이고, 클리핑본은 범위 가정을 쓴다는 사실과 함께 별도 보고한다",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args(argv)


def _refine_report(
    decoder: nn.Module,
    x: np.ndarray,
    pred: np.ndarray,
    *,
    k_sigma: float,
    label: str,
    y: np.ndarray | None = None,
    clip: bool = False,
) -> np.ndarray:
    """물리 보정을 돌리고 보고한다. y가 없으면(test) 라벨 없는 지표만 낸다.

    잔차 분포를 함께 찍는 이유: holdout(격자 위)과 test(격자 밖)의 분포를 나란히 놓으면
    **라벨 없이** 전이 여부를 볼 수 있다. 중앙값이 같으면 LM이 test 관측도 똑같이 잘 설명한다는
    뜻이다 — 단 잔차는 "관측을 설명하는가"만 재므로 **MAE 주장의 근거는 되지 못한다**
    (관측을 같게 설명하는 등가 분지가 있다). 그 확인은 리더보드뿐이다.
    """
    refined, info = refine_with_fallback(decoder, x, pred, k_sigma=k_sigma)
    res = info["residual"]
    tail = "  ".join(f"p{q}={np.percentile(res, q):.6f}" for q in (90, 99, 99.9))
    flag = f"지목 {info['flagged']}행 ({info['flagged_frac']:.2%}, 문턱 {info['threshold']:.6f})"
    out_of_range = float(
        ((refined < PHYSICAL_RANGE_NM[0]) | (refined > PHYSICAL_RANGE_NM[1])).mean()
    )
    if y is not None:
        print(
            f"  [{label}] post-LM {format_mae(mae_per_layer(info['d_lm'], y))}\n"
            f"  [{label}] 되돌림 후 {format_mae(mae_per_layer(refined, y))}"
        )
    print(f"  [{label}] 물리 보정 — {flag} · 범위 밖 값 {out_of_range:.2%}")
    print(f"  [{label}] post-LM 잔차 중앙값 {np.median(res):.6f}  {tail}")
    if clip:
        clipped = np.clip(refined, *PHYSICAL_RANGE_NM)
        if y is not None:
            print(f"  [{label}] [범위 가정 사용] 클리핑본 {format_mae(mae_per_layer(clipped, y))}")
        return clipped
    return refined


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run)
    # 학습 시점 설정 스냅샷 — train.py가 metrics.json의 "config" 키에 남긴다.
    cfg: dict[str, Any] = json.loads((run_dir / "metrics.json").read_text())["config"]
    ckpt_paths = find_checkpoints(run_dir)
    models = [load_model_checkpoint(p) for p in ckpt_paths]
    print(f"run {run_dir.name}: 체크포인트 {len(models)}개 — {[p.name for p in ckpt_paths]}")

    # 학습 때와 동일한 분할로 holdout을 재현한다 (split 옵션은 config 스냅샷이 정본).
    x, y, _, holdout_idx = prepare_from_config(cfg)
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

    decoder = None
    if args.refine:
        # 판정 수치는 complex128이다 (CLAUDE.md). 디코더는 Stage A 확정본을 쓴다.
        from src.losses import FrozenDecoder

        decoder = FrozenDecoder(dtype=torch.complex128)
        _refine_report(
            decoder,
            x_hold,
            ensemble,
            k_sigma=args.refine_k_sigma,
            label="holdout",
            y=y_hold,
            clip=args.refine_clip,
        )

    if args.submission:
        ids, x_test = load_test()
        test_pred = np.mean([predict(m, x_test, args.batch_size) for m in models], axis=0)
        sample_path = RAW_DIR / "sample_submission.csv"
        out_path = run_dir / f"submission_{run_dir.name}.csv"
        build_submission_frame(ids, test_pred, sample_path).to_csv(out_path, index=False)
        print(f"제출 파일(raw): {out_path}")
        if decoder is not None:
            # test는 라벨이 없으므로 지목 통계만 나온다 — 문턱은 **test 행 집합의 잔차 분포**에서
            # 만들어진다 (transductive). 그것이 이 규칙이 라벨 없이 작동하는 방식이다.
            refined = _refine_report(
                decoder,
                x_test,
                test_pred,
                k_sigma=args.refine_k_sigma,
                label="test",
                clip=args.refine_clip,
            )
            suffix = "_refined_clipped" if args.refine_clip else "_refined"
            ref_path = run_dir / f"submission_{run_dir.name}{suffix}.csv"
            build_submission_frame(ids, refined, sample_path).to_csv(ref_path, index=False)
            print(
                f"제출 파일(물리 보정{'  + 범위 클리핑' if args.refine_clip else ''}): {ref_path}"
            )
        if args.snap:
            snap_path = run_dir / f"submission_{run_dir.name}_snap.csv"
            snapped_pred = snap_to_grid(test_pred)
            build_submission_frame(ids, snapped_pred, sample_path).to_csv(snap_path, index=False)
            print(f"제출 파일(격자 스냅 — 누설, 분리 보고 전용): {snap_path}")


if __name__ == "__main__":
    main()
