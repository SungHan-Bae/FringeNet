"""데이콘 박막 두께 데이터 로딩 — CSV -> parquet 캐시 -> numpy/torch.

train.csv는 810,000행 x 230컬럼(1.9GB)이라 매번 CSV를 파싱하면 느리다.
최초 1회 청크 단위로 parquet 캐시를 만들고(피크 메모리를 청크 크기로 묶는다),
이후에는 캐시에서 읽는다. 캐시는 `data/cache/` 아래 생기며 .gitignore 대상이다.

**데이터는 저장소에 커밋하지 않는다.** 이 모듈은 로컬 파일만 참조한다.

컬럼 규약
---------
    layer_1..layer_4 : 두께 [nm] (타깃)
    "0".."225"       : 비식별화된 파장 인덱스별 반사율.
                       실제 nm 값이 아니다. 다만 컬럼 순서는 연속 스펙트럼으로
                       물리적 의미를 가지므로 1D conv의 축으로 쓸 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "cache"

N_CHANNELS = 226
N_LAYERS = 4
LAYER_COLS: list[str] = [f"layer_{i}" for i in range(1, N_LAYERS + 1)]
CHANNEL_COLS: list[str] = [str(i) for i in range(N_CHANNELS)]

_CHUNK_ROWS = 50_000


def _csv_dtypes(*, with_layers: bool, with_id: bool) -> dict[str, str]:
    """CSV 파싱 dtype. 반사율은 float32, 두께는 int16으로 충분하다 (10~300 nm)."""
    dtypes: dict[str, str] = dict.fromkeys(CHANNEL_COLS, "float32")
    if with_layers:
        dtypes.update(dict.fromkeys(LAYER_COLS, "int16"))
    if with_id:
        dtypes["id"] = "int64"
    return dtypes


def _build_parquet_cache(csv_path: Path, parquet_path: Path, dtypes: dict[str, str]) -> None:
    """CSV를 청크 단위로 읽어 parquet으로 옮긴다 (피크 메모리 = 청크 크기)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")

    writer: pq.ParquetWriter | None = None
    try:
        reader = pd.read_csv(csv_path, dtype=dtypes, chunksize=_CHUNK_ROWS)
        for chunk in reader:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"{csv_path} 에서 읽은 행이 없다")
    # 중간에 죽은 캐시를 정상 캐시로 오인하지 않도록 마지막에 원자적으로 옮긴다.
    tmp_path.replace(parquet_path)


def load_frame(split: str, *, use_cache: bool = True) -> pd.DataFrame:
    """train/test 전체를 DataFrame으로 읽는다.

    Args:
        split: "train" 또는 "test".
        use_cache: True면 parquet 캐시를 쓰고, 없으면 만든다.
            주의 — 캐시는 존재 여부만 보고 원본 CSV의 변경을 감지하지 않는다.
            `data/raw/`의 파일을 교체했다면 `data/cache/`를 지워야 반영된다.

    Returns:
        DataFrame — train은 layer_1..4 + "0".."225", test는 id + "0".."225".

    Raises:
        FileNotFoundError: `data/raw/{split}.csv` 가 없는 경우.
        ValueError: split 이름이 잘못된 경우.
    """
    if split not in ("train", "test"):
        raise ValueError(f'split은 "train" 또는 "test" 여야 한다 (받은 값: {split})')

    csv_path = RAW_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} 가 없다. 데이콘 대회 페이지에서 받아 data/raw/ 에 두어야 한다."
        )

    dtypes = _csv_dtypes(with_layers=(split == "train"), with_id=(split == "test"))
    if not use_cache:
        return pd.read_csv(csv_path, dtype=dtypes)

    parquet_path = CACHE_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        _build_parquet_cache(csv_path, parquet_path, dtypes)
    return pq.read_table(parquet_path).to_pandas()


def load_train(*, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """학습셋을 (스펙트럼, 두께) 배열로 읽는다.

    Returns:
        (x, y): x는 (N, 226) float32 반사율, y는 (N, 4) float32 두께 [nm].
    """
    frame = load_frame("train", use_cache=use_cache)
    x = frame[CHANNEL_COLS].to_numpy(dtype=np.float32)
    y = frame[LAYER_COLS].to_numpy(dtype=np.float32)
    return x, y


def load_test(*, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """테스트셋을 (id, 스펙트럼)으로 읽는다.

    Returns:
        (ids, x): ids는 (N,) int64, x는 (N, 226) float32.
    """
    frame = load_frame("test", use_cache=use_cache)
    ids = frame["id"].to_numpy(dtype=np.int64)
    x = frame[CHANNEL_COLS].to_numpy(dtype=np.float32)
    return ids, x


class FringeDataset(Dataset):
    """반사율 스펙트럼 -> 두께 회귀용 Dataset.

    Args:
        x: (N, 226) 반사율.
        y: (N, 4) 두께 [nm]. None이면 추론용으로 스펙트럼만 낸다.

    Returns per item:
        y가 있으면 (spectrum (226,) float32, thickness (4,) float32),
        없으면 spectrum (226,) float32.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        if y is not None and len(x) != len(y):
            raise ValueError(f"x({len(x)})와 y({len(y)})의 행 수가 다르다")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = None if y is None else torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def random_split_indices(
    n: int, val_frac: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """시드 고정 무작위 train/val 인덱스 분할.

    주의: 전수 조합 격자 데이터에서 무작위 split은 모든 두께 값이 학습에 등장하므로
    "조합 보간" 성능만 측정한다. 진짜 외삽은 held-out 두께 값 split으로 봐야 한다
    (README §3.5).

    Args:
        n: 전체 행 수.
        val_frac: 검증셋 비율.
        seed: 난수 시드.

    Returns:
        (train_idx, val_idx): 겹치지 않는 인덱스 배열.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac은 (0, 1) 사이여야 한다 (받은 값: {val_frac})")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]


def kfold_indices(
    indices: np.ndarray, n_folds: int, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """주어진 인덱스 집합 안에서 k-fold (train, val) 분할 목록을 만든다.

    프로젝트 공통 holdout(random_split_indices의 val)을 **뺀 나머지** 안에서만 접는
    용도다 — 어떤 fold 모델도 holdout을 보지 않아야 fold 앙상블을 holdout으로
    공정하게 평가할 수 있다 (src/train.py 프로토콜 참조).

    Args:
        indices: 접을 인덱스 배열 (예: train_idx).
        n_folds: fold 수 (>= 2).
        seed: 셔플 시드.

    Returns:
        길이 n_folds의 [(train_idx, val_idx)]. val들은 서로소이고 합집합이 indices 전체.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds는 2 이상이어야 한다 (받은 값: {n_folds})")
    if len(indices) < n_folds:
        raise ValueError(f"인덱스 수({len(indices)})가 n_folds({n_folds})보다 적다")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(indices))
    chunks = np.array_split(perm, n_folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        train_part = np.concatenate([c for j, c in enumerate(chunks) if j != i])
        out.append((train_part, chunks[i]))
    return out


def prepare_train_arrays(
    *, val_frac: float = 0.1, seed: int = 42, subset: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """학습 배열과 (train_idx, holdout_idx)를 만든다 — train.py / evaluate.py 공용.

    subset을 주면 시드 고정 **무작위** 표본을 먼저 뽑는다. train 행이 (layer_1..4)
    사전식 정렬이라 `x[:N]` 앞머리 자르기는 표본이 아니기 때문이다 (CLAUDE.md
    "표본 추출 주의"). 같은 (seed, subset, val_frac)이면 항상 같은 분할이 나온다.

    Returns:
        (x, y, train_idx, holdout_idx):
            x (N, 226) float32 반사율, y (N, 4) float32 두께 [nm],
            인덱스 두 개는 x/y 기준 행 번호 (서로소, 합집합 = 전체).
    """
    x, y = load_train()
    if subset is not None:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(x), size=int(subset), replace=False)
        x, y = x[pick], y[pick]
    train_idx, holdout_idx = random_split_indices(len(x), val_frac=val_frac, seed=seed)
    return x, y, train_idx, holdout_idx
