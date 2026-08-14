"""역해 LM의 추론 비용 — 장치·dtype·반복수별 실측.

역산 refinement는 holdout MAE를 크게 낮추지만 **행당 270회의 TMM forward**를 쓴다
(야코비안 중앙차분 8 + 시험 스텝 1) × 30회. 그래서 "작은 모델 + 물리"가 추론 지연에서도
싼지는 따로 물어야 하는 질문이다 — 213M skip-MLP의 forward 한 번과 같은 조건에서 잰다.

시간과 **MAE를 함께** 낸다. float32로 빨라진 대신 정확도가 무너지면 그건 개선이 아니고,
반복수를 줄여 빨라진 것도 마찬가지다 (조기 종료 여지가 있는지가 이 표에서 읽힌다).

산출물:
  reports/inversion_bench.md   (재실행 시 덮어씀)

사용법:
    python scripts/bench_invert.py                      # 현재 장치 자동 (CUDA 있으면 포함)
    python scripts/bench_invert.py --rows 2048 --devices cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_axes import SUBSAMPLE_SEED  # noqa: E402

from src.data.dataset import REPO_ROOT, prepare_from_config, subsample_indices  # noqa: E402
from src.evaluate import load_model_checkpoint  # noqa: E402
from src.losses import DEFAULT_DECODER, FrozenDecoder  # noqa: E402
from src.models import build_model  # noqa: E402
from src.physics.invert import lm_invert  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "inversion_bench.md"
CNN_RUN = "runs/level1_cnn/flatten-dilated-bound"
# 비교 대상 = 리더보드 1등 단일 모델. **시간은 가중치 값에 무관**하므로 무작위 초기화로 잰다
# (813 MB 체크포인트가 Drive에만 있어도 이 표는 만들 수 있다). 설정은 커밋된 스냅샷에서 읽는다.
STRONG_RUN = "runs/strong_baseline/winner-repro-asis"
# (dtype, iters) 조합 — complex128 30회가 리포트가 쓰는 설정이다.
LM_VARIANTS = ((torch.complex128, 30), (torch.complex64, 30), (torch.complex64, 10))


def timed(fn: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    """벽시계 초와 결과. CUDA는 비동기라 동기화하지 않으면 0초로 찍힌다."""
    fn()  # 워밍업 — 커널 컴파일·캐시·지연 초기화를 시간에서 뺀다
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def model_of(run: str, device: torch.device, *, weights: bool) -> tuple[torch.nn.Module, int]:
    """run의 모델을 올린다. weights=False면 설정만 읽어 무작위 초기화한다 (**시간 전용**).

    weights=True인데 체크포인트가 없으면 **에러를 낸다** — 조용히 무작위 가중치로 넘어가면
    MAE 열이 의미 없는 값으로 채워지고 표는 정상으로 보인다.
    """
    metrics = json.loads((REPO_ROOT / run / "metrics.json").read_text())
    ckpt = REPO_ROOT / run / "model.pt"
    if weights and not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt}\n"
            "  MAE 열이 필요하므로 무작위 가중치로 대체하지 않는다. Drive 미러에서 복사하거나\n"
            "  git show <원본 커밋>:<경로> 로 되살릴 것 (runs/CHECKPOINTS.md)"
        )
    model = load_model_checkpoint(ckpt) if weights else build_model(metrics["config"]["model"])
    model = model.eval().to(device)
    return model, sum(p.numel() for p in model.parameters())


def mae_nm(d: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(d.astype(np.float64) - y.astype(np.float64)).mean())


def bench_device(
    device: torch.device, x: np.ndarray, y: np.ndarray, decoder_path: str
) -> list[dict[str, Any]]:
    """한 장치에서 신경망 forward 2종 + LM 변형들을 잰다. 반환 행마다 ms/행과 MAE."""
    rows = len(x)
    xt = torch.from_numpy(x).to(device)
    out: list[dict[str, Any]] = []

    @torch.no_grad()
    def forward(model: torch.nn.Module) -> torch.Tensor:
        return model(xt)

    cnn, cnn_params = model_of(CNN_RUN, device, weights=True)
    dt, pred = timed(lambda m=cnn: forward(m), device)
    d_hat = pred.detach().cpu().numpy().astype(np.float64)
    out.append(
        {
            "what": "CNN forward",
            "params": cnn_params,
            "ms": dt / rows * 1e3,
            "mae": mae_nm(d_hat, y),
        }
    )

    strong, strong_params = model_of(STRONG_RUN, device, weights=False)
    dt, _ = timed(lambda m=strong: forward(m), device)
    out.append(
        {"what": "skip-MLP forward", "params": strong_params, "ms": dt / rows * 1e3, "mae": None}
    )
    strong = None
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for dtype, iters in LM_VARIANTS:
        decoder = FrozenDecoder(decoder_path, dtype=dtype).to(device)
        dt, d_ref = timed(
            lambda dec=decoder, it=iters: lm_invert(dec, x, d_hat, iters=it, damping="row"),
            device,
        )
        name = "complex128" if dtype == torch.complex128 else "complex64"
        out.append(
            {
                "what": f"LM {iters}회 ({name})",
                "params": 7,
                "ms": dt / rows * 1e3,
                "mae": mae_nm(d_ref, y),
            }
        )
        decoder = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def render(results: dict[str, list[dict[str, Any]]], meta: dict[str, Any]) -> list[str]:
    lines = [
        "# 역해 LM 추론 비용 — 장치 · dtype · 반복수",
        "",
        "`scripts/bench_invert.py` 산출 — 재실행 시 덮어쓴다. 해석은 리포트 본문에서 한다.",
        "",
        f"- 표본 {meta['rows']:,}행 (holdout 무작위) · 디코더 `{meta['decoder']}`",
        f"- torch {meta['torch']} · CPU 스레드 {meta['threads']} · {meta['cpu']}",
        "- **skip-MLP는 시간만 잰다** — 가중치 값은 지연에 무관하므로 무작위 초기화다"
        " (813 MB 체크포인트는 Drive 전용). MAE는 `reports/strong_baseline.md`가 정본이다.",
        "- LM은 CNN 예측 `d_hat`에서 출발한다 (실제 워크로드와 같다).",
        "",
    ]
    for device_label, rowset in results.items():
        base = next((r["ms"] for r in rowset if r["what"] == "skip-MLP forward"), None)
        lines += [
            f"## {device_label}",
            "",
            "| 무엇 | 파라미터 | ms/행 | skip-MLP 대비 | holdout MAE [nm] |",
            "|---|---|---|---|---|",
        ]
        for r in rowset:
            ratio = f"{r['ms'] / base:.2f}×" if base else "—"
            mae = f"{r['mae']:.4f}" if r["mae"] is not None else "—"
            params = f"{r['params'] / 1e6:.2f}M" if r["params"] > 1000 else str(r["params"])
            lines.append(f"| {r['what']} | {params} | {r['ms']:.3f} | {ratio} | {mae} |")
        cnn = next(r for r in rowset if r["what"] == "CNN forward")
        for r in rowset:
            if r["what"].startswith("LM 30회 (complex128)"):
                total = cnn["ms"] + r["ms"]
                lines += [
                    "",
                    f"**cnn + LM 30회(complex128) 합계 = {total:.3f} ms/행**"
                    + (f" — skip-MLP의 {total / base:.2f}배" if base else ""),
                    "",
                ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="역해 LM 추론 비용 벤치마크")
    parser.add_argument("--rows", type=int, default=4096, help="측정 표본 행 수")
    parser.add_argument("--devices", nargs="*", default=None, help="기본: cpu (+ 있으면 cuda)")
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    cfg = json.loads((REPO_ROOT / CNN_RUN / "metrics.json").read_text())["config"]
    x_all, y_all, _, holdout_idx = prepare_from_config(cfg)
    idx = subsample_indices(len(holdout_idx), args.rows, seed=SUBSAMPLE_SEED)
    x = x_all[holdout_idx][idx]
    y = y_all[holdout_idx][idx].astype(np.float64)
    del x_all, y_all

    results: dict[str, list[dict[str, Any]]] = {}
    for name in devices:
        device = torch.device(name)
        label = f"{name} — {torch.cuda.get_device_name(device)}" if name == "cuda" else "CPU"
        print(f"\n[{label}]")
        rowset = bench_device(device, x, y, args.decoder)
        for r in rowset:
            mae = f"  MAE {r['mae']:.4f} nm" if r["mae"] is not None else ""
            print(f"  {r['what']:28s} {r['ms']:8.3f} ms/행{mae}")
        results[label] = rowset

    meta = {
        "rows": len(x),
        "decoder": args.decoder,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
    }
    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(render(results, meta)) + "\n")
    print(f"\n산출물: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
