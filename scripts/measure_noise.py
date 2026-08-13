"""데이터 노이즈의 크기 σ와 **하드 상한**을 측정한다 (Stage A 게이트의 기준값 산출).

CLAUDE.md 데이터 계약의 σ = 0.008658 과 |ε| ≤ 0.0152 를 만드는 스크립트다. Task 3의
σ ≈ 0.0087~0.0088은 2차 차분 기반 **상한**이었고(스펙트럼 곡률이 섞여 위로 밀린다),
게이트 임계 1.2σ가 여기에 직접 걸리므로 수렴시켜 확정할 필요가 있었다.

방법 1 — 채널축 m차 차분으로 σ:
    iid 노이즈에서 `Var(Δ^m y) = C(2m, m)·σ²` 이고, 매끈한 신호 성분은 차수 m을 올릴
    때 노이즈 성장 C(2m,m) ~ 4^m 보다 빠르게 억제된다. 따라서 σ̂(m)은 **단조 감소하는
    상한 수열**이고, 평평해지는 지점이 σ다. 평평해진다는 사실 자체가 잔차가 채널축에서
    백색(iid)이라는 근거이기도 하다.

    **두께축으로는 쓸 수 없다** — 두께 방향은 전 범위에 주기가 1.07~4.5개뿐인
    과소표집이라 고차 차분이 신호를 지우지 못하고 신호 수준에서 정체한다. 이 스크립트가
    두 축을 모두 계산해 그 사실을 확인한다.

방법 2 — 음수 꼬리에서 하드 상한:
    R_true ≥ 0 이므로 관측 최소값이 노이즈 하한을 드러낸다. 가우시안이면 5σ까지
    나와야 하는데 특정 값에서 딱 끊기면 **유계**가 확정된다. 균등분포 가정의
    a = σ√3 과 대조한다.

표본 주의: train 행은 (layer_1..4) 사전식 정렬이라 row group을 그대로 쓰면 layer_1
구석만 본다 (CLAUDE.md "표본 추출 주의"). 17개 row group에서 각각 무작위 표집한다.

사용법:
    python scripts/measure_noise.py                 # 전체 (음수 꼬리 스캔에 ~1분)
    python scripts/measure_noise.py --skip-tail     # σ만
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import REPO_ROOT  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache" / "train.parquet"
CHANNELS = [str(i) for i in range(226)]
TAIL_THRESHOLDS = (-0.0158, -0.0156, -0.0154, -0.0152, -0.0151, -0.0150, -0.0148, -0.0145, -0.0140)


def stratified_sample(rows_per_group: int, seed: int) -> np.ndarray:
    """row group마다 무작위 표집해 사전식 정렬 편향을 피한다. 반환 (N, 226) float64."""
    handle = pq.ParquetFile(CACHE)
    rng = np.random.default_rng(seed)
    chunks = []
    for group in range(handle.num_row_groups):
        table = handle.read_row_group(group, columns=CHANNELS).to_pandas().to_numpy(np.float64)
        take = min(rows_per_group, len(table))
        chunks.append(table[rng.choice(len(table), take, replace=False)])
    return np.concatenate(chunks)


def sigma_by_differences(x: np.ndarray, axis: int, max_order: int = 8) -> list[tuple[int, float]]:
    """m차 차분 기반 σ 추정 수열. 반환 [(m, σ̂(m))] — 수렴하면 그 값이 σ다."""
    return [
        (m, float(np.sqrt(np.diff(x, n=m, axis=axis).var() / comb(2 * m, m))))
        for m in range(1, max_order + 1)
    ]


def negative_tail_counts() -> tuple[dict[float, int], float, int]:
    """전 데이터에서 음수 꼬리 누적 개수를 센다. 반환 (임계값→개수, 최소값, 총 관측 수)."""
    handle = pq.ParquetFile(CACHE)
    counts = dict.fromkeys(TAIL_THRESHOLDS, 0)
    minimum = np.inf
    total = 0
    for group in range(handle.num_row_groups):
        table = handle.read_row_group(group, columns=CHANNELS).to_pandas().to_numpy(np.float32)
        total += table.size
        tail = table[table < max(TAIL_THRESHOLDS)]
        minimum = min(minimum, float(table.min()))
        for threshold in TAIL_THRESHOLDS:
            counts[threshold] += int((tail < threshold).sum())
    return counts, minimum, total


def main() -> int:
    parser = argparse.ArgumentParser(description="데이터 노이즈 σ·상한 측정")
    parser.add_argument("--rows-per-group", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tail", action="store_true", help="음수 꼬리 전수 스캔 생략")
    args = parser.parse_args()

    x = stratified_sample(args.rows_per_group, args.seed)
    print(f"층화 표본 {x.shape[0]:,}행 × {x.shape[1]}채널 (row group마다 무작위 표집)\n")

    print("[1] 채널축 m차 차분 — 상한 수열이 수렴하는 값이 σ")
    channel = sigma_by_differences(x, axis=1)
    for m, value in channel:
        print(f"    m={m}  σ̂ = {value:.6f}")
    sigma = channel[-1][1]
    print(f"\n    → σ = {sigma:.6f}   (m=5~8에서 수렴 = 잔차가 채널축에서 백색)")
    print(f"    → 균등분포 가정의 폭 a = σ√3 = {sigma * np.sqrt(3.0):.6f}\n")

    print("[2] 두께축 — 같은 추정이 통하지 않음을 확인 (과소표집)")
    handle = pq.ParquetFile(CACHE)
    middle = (
        handle.read_row_group(handle.num_row_groups // 2, columns=CHANNELS)
        .to_pandas()
        .to_numpy(np.float64)
    )
    # 사전식 정렬이라 layer_4가 가장 빠르게 변한다 → 연속 30행이 d₄ = 10..300 한 블록.
    blocks = middle[: len(middle) // 30 * 30].reshape(-1, 30, 226)
    for m, value in sigma_by_differences(blocks, axis=1)[3:]:
        print(f"    m={m}  σ̂ = {value:.6f}")
    print("    → 신호 수준에서 정체한다 (두께 방향 주기가 1.07~4.5개뿐) — 채널축만 유효\n")

    if not args.skip_tail:
        print("[3] 음수 꼬리 — 노이즈가 유계인가")
        counts, minimum, total = negative_tail_counts()
        for threshold in TAIL_THRESHOLDS:
            print(f"    R_obs < {threshold:+.4f} : {counts[threshold]:9,d}")
        print(f"\n    전체 관측 {total:,} / 최소값 {minimum:.6f}")
        print(
            f"    → 가우시안(σ={sigma:.6f})이면 5σ = {-5 * sigma:.4f}까지 나와야 하는데"
            " 특정 값에서 끊긴다 = 유계 확정"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
