"""데이터 가설 검증 — CLAUDE.md §데이터 계약의 체크박스를 실제로 확인한다.

가설을 코드로 못박아 두는 이유: 이후 모든 설계(출력 bound 범위, split 전략,
격자 스냅 논의)가 "두께가 10 nm 격자 위 전수 조합"이라는 전제에 의존한다.
전제가 틀리면 조용히 틀린 모델이 나오므로, 통과/실패를 종료 코드로 낸다.

사용법:
    python scripts/verify_data.py

종료 코드: 모든 가설 통과 시 0, 하나라도 실패하면 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (  # noqa: E402
    CHANNEL_COLS,
    LAYER_COLS,
    N_CHANNELS,
    RAW_DIR,
    load_frame,
)

EXPECTED_TRAIN_ROWS = 810_000
EXPECTED_TEST_ROWS = 10_000
EXPECTED_THICKNESS_VALUES = np.arange(10, 301, 10, dtype=np.int64)  # 10..300, 10 nm 격자

# 반사율 범위 — 최초 가설 "R ⊂ [0, 1]" 은 검증 결과 **거짓**이었다.
# train 최소 -0.015117 / test 최소 -0.014998 로 음의 반사율이 나온다. 물리적으로
# 불가능하므로 참 스펙트럼에 가산 노이즈가 얹혀 있다는 뜻이다. 음수값이 -0.0151에서
# 잘리고(1퍼센타일 -0.0135) 2차 차분 기반 sigma 추정 0.0087 이 균등분포 ±0.015의
# sigma(0.015/sqrt(3)=0.00866)와 일치하는 것으로 보아 유계 노이즈에 가깝다.
# 따라서 검증하는 대상을 "확인된 사실"로 바꾼다: 위로는 1을 넘지 않고,
# 아래로는 노이즈 수준(-0.02) 이상이어야 한다.
R_UPPER_BOUND = 1.0
R_LOWER_BOUND = -0.02


class CheckLog:
    """가설 검증 결과를 모아 PASS/FAIL로 출력한다."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, passed: bool, name: str, detail: str = "") -> bool:
        mark = "PASS" if passed else "FAIL"
        suffix = f"  — {detail}" if detail else ""
        print(f"  [{mark}] {name}{suffix}")
        if not passed:
            self.failures.append(name)
        return passed

    def note(self, text: str) -> None:
        print(f"         {text}")


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_reflectance_range(log: CheckLog, spectra: np.ndarray, label: str) -> None:
    """반사율이 확인된 범위 안에 있는지 본다 (물리 범위 [0,1]이 아니다 — 위 상수 주석 참고)."""
    r_min, r_max = float(spectra.min()), float(spectra.max())
    log.check(
        r_max <= R_UPPER_BOUND,
        f"{label} 반사율 ≤ {R_UPPER_BOUND}",
        f"실제 최대 {r_max:.6f}",
    )
    log.check(
        r_min >= R_LOWER_BOUND,
        f"{label} 반사율 ≥ {R_LOWER_BOUND} (음수는 노이즈 수준까지만)",
        f"실제 최소 {r_min:.6f}",
    )
    negative = spectra < 0.0
    n_neg = int(negative.sum())
    frac_rows = float(negative.any(axis=1).mean())
    log.note(
        f"{label} 음의 반사율 {n_neg:,}개 ({100 * n_neg / spectra.size:.4f}%), "
        f"영향 행 {100 * frac_rows:.2f}% → 참 스펙트럼 위의 가산 노이즈 증거"
    )


def verify_train(log: CheckLog) -> None:
    """train.csv의 4개 가설을 검증한다."""
    _section("train.csv")
    frame = load_frame("train")
    mem_gb = frame.memory_usage(deep=True).sum() / 1e9
    print(f"  로드 완료: shape={frame.shape}, 메모리={mem_gb:.2f} GB")

    # 가설 1 — 행 수
    log.check(
        len(frame) == EXPECTED_TRAIN_ROWS,
        f"행 수 = {EXPECTED_TRAIN_ROWS:,}",
        f"실제 {len(frame):,}",
    )

    # 컬럼 구조 (가설의 전제)
    expected_cols = LAYER_COLS + CHANNEL_COLS
    log.check(
        list(frame.columns) == expected_cols,
        f"컬럼 = layer_1..4 + '0'..'{N_CHANNELS - 1}' ({len(expected_cols)}개, 순서 포함)",
        f"실제 {len(frame.columns)}개",
    )

    # 가설 2 — 각 층 고유값이 10 nm 격자 30개
    all_grid = True
    for col in LAYER_COLS:
        uniq = np.sort(frame[col].unique().astype(np.int64))
        ok = uniq.shape == EXPECTED_THICKNESS_VALUES.shape and bool(
            (uniq == EXPECTED_THICKNESS_VALUES).all()
        )
        all_grid &= ok
        if not ok:
            log.note(f"{col} 고유값 {len(uniq)}개: {uniq[:5]} ... {uniq[-3:]}")
    log.check(all_grid, "각 layer 고유값 = {10, 20, ..., 300} (30개, 10 nm 격자)")

    # 가설 3 — 반사율 범위·결측·중복
    spectra = frame[CHANNEL_COLS].to_numpy(dtype=np.float32)
    check_reflectance_range(log, spectra, "train")

    n_missing = int(frame.isna().to_numpy().sum())
    log.check(n_missing == 0, "결측 없음", f"결측 {n_missing}개")

    n_dup = int(frame[LAYER_COLS].duplicated().sum())
    log.check(n_dup == 0, "두께 조합 중복 없음", f"중복 {n_dup}행")

    # 전수 조합(30^4) 여부 — 무작위 split이 "조합 보간"만 재는 근거가 된다.
    n_unique = int(frame[LAYER_COLS].drop_duplicates().shape[0])
    log.check(
        n_unique == 30**4 == EXPECTED_TRAIN_ROWS,
        "30^4 전수 조합 (격자를 빠짐없이 채움)",
        f"고유 조합 {n_unique:,}",
    )

    _reference_stats(spectra)


def _reference_stats(spectra: np.ndarray) -> None:
    """가설은 아니지만 이후 설계에 쓰이는 참고 통계 (Task 3 EDA의 출발점)."""
    _section("참고 통계 (가설 아님)")
    print(f"  반사율 mean={spectra.mean():.6f}  std={spectra.std():.6f}")
    print(f"  행별 스펙트럼 범위 중앙값: {np.median(spectra.max(1) - spectra.min(1)):.6f}")

    # 2차 차분 노이즈 추정: y=s+e 이고 s가 국소 선형이면 Var(y[i-1]-2y[i]+y[i+1]) ~ 6*sigma^2.
    # 스펙트럼 곡률이 그대로 섞여 들어가므로 **상한**이다 — 무늬가 조밀할수록 부풀려진다.
    #
    # 표본을 무작위로 뽑아야 한다. 행이 (layer_1..layer_4) 사전식으로 정렬되어 있어
    # spectra[:N] 같은 앞머리 자르기는 layer_1 = 10 nm 인 구석만 보게 된다(평평한 스펙트럼
    # 쪽으로 치우쳐 추정치가 낮게 나온다).
    rng = np.random.default_rng(0)
    sample = spectra[rng.choice(len(spectra), size=20_000, replace=False)].astype(np.float64)
    second_diff = sample[:, :-2] - 2 * sample[:, 1:-1] + sample[:, 2:]
    sigma_upper = float(second_diff.std() / np.sqrt(6.0))

    # 곡률 오염이 가장 적은 행(스펙트럼 진폭 하위 10%)만 보면 상한이 참값에 가까워진다.
    spread = sample.max(axis=1) - sample.min(axis=1)
    flat = sample[spread <= np.percentile(spread, 10)]
    flat_diff = flat[:, :-2] - 2 * flat[:, 1:-1] + flat[:, 2:]
    sigma_flat = float(flat_diff.std() / np.sqrt(6.0))

    print(f"  노이즈 sigma 상한   = {sigma_upper:.6f}  (2차 차분, 무작위 20,000행)")
    print(f"  노이즈 sigma (평평)  = {sigma_flat:.6f}  (같은 식, 진폭 하위 10% 행만)")
    negatives = spectra[spectra < 0.0]
    if negatives.size:
        print(
            f"  음수 반사율 하한 = {negatives.min():.6f}  "
            f"→ 균등 노이즈 ±a 라면 sigma = a/sqrt(3) = {abs(negatives.min()) / np.sqrt(3):.6f}"
        )
    print("  ↑ 앞 둘은 같은 추정식(상한), 마지막이 독립 추정. 해석은 Task 3 EDA에서.")


def verify_test_and_submission(log: CheckLog) -> None:
    """test.csv와 sample_submission.csv의 형식을 검증한다."""
    _section("test.csv / sample_submission.csv")
    test = load_frame("test")
    log.check(
        len(test) == EXPECTED_TEST_ROWS,
        f"test 행 수 = {EXPECTED_TEST_ROWS:,}",
        f"실제 {len(test):,}",
    )
    log.check(
        list(test.columns) == ["id", *CHANNEL_COLS],
        f"test 컬럼 = id + '0'..'{N_CHANNELS - 1}'",
        f"실제 {len(test.columns)}개",
    )
    log.check(int(test.isna().to_numpy().sum()) == 0, "test 결측 없음")

    test_spectra = test[CHANNEL_COLS].to_numpy(dtype=np.float32)
    check_reflectance_range(log, test_spectra, "test")

    sub = pd.read_csv(RAW_DIR / "sample_submission.csv")
    log.check(
        list(sub.columns) == ["id", *LAYER_COLS],
        "sample_submission 컬럼 = id + layer_1..4",
        f"실제 {list(sub.columns)}",
    )
    log.check(
        len(sub) == len(test) and bool((sub["id"].to_numpy() == test["id"].to_numpy()).all()),
        "sample_submission id가 test와 일치",
    )


def main() -> int:
    print("=" * 70)
    print("FringeNet 데이터 가설 검증 (CLAUDE.md §데이터 계약)")
    print("=" * 70)

    log = CheckLog()
    verify_train(log)
    verify_test_and_submission(log)

    _section("결과")
    if log.failures:
        print(f"  {len(log.failures)}개 가설 실패:")
        for name in log.failures:
            print(f"    - {name}")
        print("\n  CLAUDE.md 체크박스와 README §2를 실제 결과에 맞춰 고쳐야 한다.")
        return 1

    print("  모든 가설 통과.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
