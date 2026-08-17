"""레시피 라운드 판정 — post-LM 정확도와 분지 실패율을 run별로 나란히 잰다.

**val MAE로 모델을 고르면 틀린다.** 라운드 1이 실측으로 보여준 것이 이것이다: 노이즈 증강은
pre-LM 최고인데 post-LM에서 세 번째고, 꼬리 가중 손실은 pre-LM 최악인데 post-LM 최저다.
이유는 `reports/inversion_refine.md`의 십분위에 있다 — 1~9분위가 이미 0.25~0.47 nm로 평평해서
**LM이 정밀도를 회복해 주므로 pre-LM 정밀도 개선은 post-LM에 남지 않는다.** 남는 것은 꼬리다.

그래서 판정 지표는 두 개다.

    post-LM MAE     추론 후 물리 보정까지 끝낸 정확도 = 실제 배포 성능
    분지 실패율      행 평균 오차 > 5 nm 비율 — 오차 구조가 post-LM ≈ 0.33 + 실패율 × 약 40 nm

분지 실패 정의는 `refine_inversion.py`와 **같은 것을 쓴다** (`arm_stats`) — 두 곳에서 따로
정의하면 조용히 갈라진다.

계약 셋:

1. **같은 행·같은 디코더·같은 장치로 전부 잰다.** CPU↔GPU는 bit가 아니라 MAE 수준에서
   일치하므로(같은 표본 약 1.8% 차) 장치를 섞은 표는 읽을 수 없다. 기준 팔도 여기서 함께 잰다.
2. **표본 수치다.** 기본 5,000행이고 정본(81,000행)보다 어려울 수 있다 — 라운드 1 표본에서
   옛 CNN 실패율이 0.78%인데 전체는 0.67%다. 가지를 고르는 용도이고 최종 수치는
   `refine_inversion.py`로 전체 holdout에서 낸다.
3. **야코비안은 해석적 + 조기 종료.** 정확도는 중앙차분과 같고 약 10배 빠르다
   (`reports/inversion_bench.md`, 근거는 tests/test_tmm.py §8).

산출물:
  reports/cnn_recipe_judge.md   (재실행 시 덮어씀)

사용법:
    python scripts/judge_recipe.py --run runs/cnn_recipe/budget100
    python scripts/judge_recipe.py --run runs/cnn_recipe/budget100 --rows 20000 --device cpu

체크포인트는 git에 없다 (`runs/`는 텍스트만 추적) — Drive 미러에서 받아온 뒤 실행할 것
(`runs/CHECKPOINTS.md`). 없으면 복구 방법을 알려주고 멈춘다.
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

from evaluate_axes import SUBSAMPLE_SEED, holdout_of, load_run  # noqa: E402
from refine_inversion import BRANCH_FAIL_NM  # noqa: E402 — 분지 실패 정의 단일 출처

from src.data.dataset import REPO_ROOT, subsample_indices  # noqa: E402
from src.losses import DEFAULT_DECODER, FrozenDecoder  # noqa: E402
from src.physics.invert import lm_invert  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "cnn_recipe_judge.md"
# 라운드 1의 기준 팔 — 예산을 늘리기 전의 확정 백본. 판정 표의 원점이다.
REFERENCE_RUN = "runs/level1_cnn/flatten-dilated-bound"
DEFAULT_ROWS = 5_000
# 조기 종료 문턱 [nm] — 두께 정확도 0.3 nm 규모보다 세 자리 아래 (bench_invert와 같은 값).
EARLY_TOL_NM = 1e-4
LM_ITERS = 30


def judge_one(
    model: torch.nn.Module,
    decoder: FrozenDecoder,
    x: np.ndarray,
    y: np.ndarray,
    *,
    iters: int = LM_ITERS,
    tol_nm: float | None = EARLY_TOL_NM,
) -> dict[str, float]:
    """한 모델의 pre/post-LM 오차 요약. x (N, W) 관측, y (N, L) 참 두께 [nm].

    Returns:
        {"cnn", "lm", "fail", "median", "p99"} — MAE·분지 실패율·행 오차 분위.
    """
    device = next((t.device for t in (*model.parameters(), *model.buffers())), torch.device("cpu"))
    with torch.no_grad():
        d_hat = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)
    d_lm = lm_invert(
        decoder,
        x,
        d_hat,
        iters=iters,
        damping="row",
        chunk=4096,
        jacobian="analytic",
        tol_nm=tol_nm,
    )
    row_err = np.abs(d_lm - y).mean(axis=1)

    # 라벨 없는 재구성 잔차 — 이것이 분지 실패를 가리키면 실패 행만 골라 재시도할 수 있다.
    # 장치를 명시해서 옮긴다: 디코더가 GPU면 관측도 GPU로 올려야 한다.
    res = np.empty(len(d_lm), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(d_lm), 4096):
            recon = decoder(torch.from_numpy(d_lm[s : s + 4096]).to(device))
            obs = torch.from_numpy(x[s : s + 4096]).to(device=device, dtype=recon.dtype)
            res[s : s + 4096] = (recon - obs).abs().mean(dim=1).cpu().numpy()

    fail = row_err > BRANCH_FAIL_NM
    order = np.argsort(-res)
    detect: dict[str, float] = {}
    for pct in (0.5, 1.0):
        n = max(1, int(round(len(res) * pct / 100)))
        picked = np.zeros(len(res), dtype=bool)
        picked[order[:n]] = True
        caught = int((picked & fail).sum())
        detect[f"recall@{pct:g}"] = caught / max(int(fail.sum()), 1)
        detect[f"precision@{pct:g}"] = caught / n
    return {
        "cnn": float(np.abs(d_hat - y).mean()),
        "lm": float(np.abs(d_lm - y).mean()),
        "fail": float(fail.mean()),
        "median": float(np.median(row_err)),
        "p99": float(np.percentile(row_err, 99)),
        "res_fail": float(np.median(res[fail])) if fail.any() else float("nan"),
        "res_ok": float(np.median(res[~fail])),
        **detect,
    }


def judge_runs(
    run_dirs: list[str | Path],
    *,
    reference: str | Path | None = REFERENCE_RUN,
    rows: int = DEFAULT_ROWS,
    device: torch.device | None = None,
    decoder_path: str = DEFAULT_DECODER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """run들을 **같은 행·같은 디코더·같은 장치**로 나란히 잰다.

    Args:
        run_dirs: 판정할 run 디렉토리들. reference: 원점으로 함께 잴 run (None이면 생략).
        rows: holdout에서 무작위 표집할 행 수 (시드 고정 — 모든 run이 같은 행을 본다).

    Returns:
        (rows_out, meta) — rows_out은 표 한 줄씩, meta는 리포트 머리말용 설정 스냅샷.

    Raises:
        ValueError: run들의 `data` 블록이 다른 경우 (holdout이 달라 나란히 비교할 수 없다).
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = ([("기준", Path(reference))] if reference else []) + [
        (Path(r).name, Path(r)) for r in run_dirs
    ]

    loaded = [(label, *load_run(path)) for label, path in targets]
    blocks = {json.dumps(m["config"].get("data"), sort_keys=True) for _, _, m in loaded}
    if len(blocks) > 1:
        raise ValueError(f"run마다 split이 다르다 — 나란히 비교할 수 없다: {sorted(blocks)}")

    x_hold, y_hold = holdout_of(loaded[0][2]["config"], {})
    idx = subsample_indices(len(x_hold), rows, seed=SUBSAMPLE_SEED)
    x = x_hold[idx]
    y = y_hold[idx].astype(np.float64)
    del x_hold, y_hold

    decoder = FrozenDecoder(decoder_path, dtype=torch.complex128).to(dev)
    out: list[dict[str, Any]] = []
    for label, model, metrics in loaded:
        stats = judge_one(model.to(dev), decoder, x, y)
        out.append(
            {
                "label": label,
                "run": f"{metrics['experiment']}/{metrics['run_name']}",
                "val_mae": metrics["model"]["val_mae"],
                "epochs": metrics["config"]["train"]["epochs"],
                "l2_weight": float(metrics["config"]["train"].get("l2_weight", 0.0)),
                **stats,
            }
        )
    meta = {
        "rows": len(x),
        "decoder": decoder.provenance["decoder"],
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev)
        if dev.type == "cuda"
        else platform.machine(),
        "iters": LM_ITERS,
        "tol_nm": EARLY_TOL_NM,
        "branch_fail_nm": BRANCH_FAIL_NM,
        "seed": SUBSAMPLE_SEED,
    }
    return out, meta


def render(rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    """판정 표를 markdown으로. 해석은 리포트 본문(reports/cnn_recipe.md)에서 한다."""
    origin = rows[0]["lm"] if rows else None
    lines = [
        "# cnn_recipe 판정 — post-LM 정확도와 분지 실패율",
        "",
        "`scripts/judge_recipe.py` 산출 — 재실행 시 덮어쓴다. 해석은 `reports/cnn_recipe.md`.",
        "",
        f"- 표본 {meta['rows']:,}행 (holdout 무작위, 시드 {meta['seed']})"
        " — **모든 run이 같은 행을 본다**",
        f"- 디코더 `{meta['decoder']}` (Stage A 확정, 동결) · complex128",
        f"- LM {meta['iters']}회 · 해석적 야코비안 · 조기 종료 {meta['tol_nm']:g} nm · 감쇠 행별",
        f"- 장치 {meta['device']} ({meta['device_name']}) — **CPU↔GPU는 MAE 수준에서만 일치한다**"
        " (같은 표본 약 1.8% 차). 표 안의 비교만 유효하다.",
        f"- 분지 실패 = 행 평균 오차 > {meta['branch_fail_nm']:.0f} nm"
        " (`refine_inversion.py`와 같은 정의)",
        "- **표본 수치다.** 최종 정본은 `refine_inversion.py`로 전체 holdout에서 낸다.",
        "",
        "| run | 에폭 | λ | val MAE | CNN MAE | **post-LM** | 분지 실패 | 중앙값 | p99 | Δ원점 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lam = f"{r['l2_weight']:g}" if r["l2_weight"] else "—"
        delta = "—" if origin is None or r is rows[0] else f"{r['lm'] - origin:+.4f}"
        lines.append(
            f"| `{r['run']}` | {r['epochs']} | {lam} | {r['val_mae']:.4f} | {r['cnn']:.4f} |"
            f" **{r['lm']:.4f}** | {r['fail']:.2%} | {r['median']:.4f} | {r['p99']:.4f} | {delta} |"
        )
    if origin is not None:
        lines += [
            "",
            f"원점 = `{rows[0]['run']}` (표의 첫 행). `Δ원점`이 음수면 그만큼 좋아진 것이다.",
        ]
    # 오차 구조의 적합도 — 실패율만으로 설명되지 않는 몫이 있으면 그것 자체가 발견이다
    lines += [
        "",
        "## 오차 구조 대조 — `post-LM ≈ 0.33 + 실패율 × 40`",
        "",
        "실패 **건수**만으로 설명되는지 본다. 모형보다 좋은 run은 실패의 **심각도**를 줄인 것이다.",
        "",
        "| run | 실측 | 모형 | 차 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        model_val = 0.33 + r["fail"] * 40.0
        lines.append(
            f"| `{r['run']}` | {r['lm']:.4f} | {model_val:.4f} | {r['lm'] - model_val:+.4f} |"
        )

    lines += [
        "",
        "## 라벨 없는 실패 검출 — 재구성 잔차로 분지 실패를 가려낼 수 있는가",
        "",
        "잔차는 라벨을 쓰지 않으므로 **test에도 쓸 수 있다.** 상위 k%를 재시도 후보로 뽑을 때의",
        "포착률(실패 중 잡은 비율)과 정밀도(후보 중 실제 실패)를 함께 본다 —"
        " 포착률만 보면 후보를 넓히는 것으로 올릴 수 있다.",
        "",
        "| run | 잔차 중앙값 (실패) | (정상) | 비 | 포착@0.5% | 정밀도@0.5%"
        " | 포착@1% | 정밀도@1% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ratio = r["res_fail"] / r["res_ok"] if r["res_ok"] else float("nan")
        lines.append(
            f"| `{r['run']}` | {r['res_fail']:.6f} | {r['res_ok']:.6f} | {ratio:.1f}× |"
            f" {r['recall@0.5']:.1%} | {r['precision@0.5']:.1%} |"
            f" {r['recall@1']:.1%} | {r['precision@1']:.1%} |"
        )
    lines += [
        "",
        "**포착률이 후보를 넓혀도 오르지 않는 지점**이 이 지표의 상한이다 — 남는 실패는 잔차로",
        "구분되지 않는다(관측을 거의 같게 설명하는 등가 분지). 그 몫은 디코더 개선(게이트 (b))",
        "쪽 문제이고 재시도로는 잡히지 않는다.",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="레시피 run들의 post-LM·분지 실패율 판정")
    parser.add_argument(
        "--run", action="append", required=True, help="판정할 run 디렉토리 (반복 가능)"
    )
    parser.add_argument("--reference", default=REFERENCE_RUN, help="원점 run ('none'이면 생략)")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="표본 행 수")
    parser.add_argument("--device", default=None, help="기본: cuda 있으면 cuda")
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    parser.add_argument("--out", default=None, help=f"리포트 경로 (기본 {OUT_PATH})")
    args = parser.parse_args()

    reference = None if str(args.reference).lower() == "none" else args.reference
    rows, meta = judge_runs(
        args.run,
        reference=reference,
        rows=args.rows,
        device=torch.device(args.device) if args.device else None,
        decoder_path=args.decoder,
    )
    for r in rows:
        print(
            f"  {r['run']:44s} CNN {r['cnn']:7.4f}  post-LM {r['lm']:7.4f}  실패 {r['fail']:6.2%}"
        )

    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(render(rows, meta)) + "\n")
    print(f"\n산출물: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
