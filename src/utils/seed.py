"""전역 시드 고정 유틸 — 실험 재현성을 위한 최소 구현."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """python / numpy / torch(CPU·CUDA)의 난수 생성기를 모두 고정한다.

    Args:
        seed: 시드 값.
        deterministic: True면 cuDNN 결정론 모드를 켜고 벤치마크 자동탐색을 끈다.
            속도는 다소 손해지만 동일 커맨드가 동일 결과를 낸다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # 전역 RNG를 의도적으로 고정한다: sklearn 등 외부 라이브러리가 np.random의
    # 전역 상태를 참조하므로 Generator 인스턴스로는 대체되지 않는다.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker용 시드 함수 (`DataLoader(worker_init_fn=seed_worker)`).

    각 워커가 서로 다른, 그러나 재현 가능한 시드를 갖도록 한다.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002  (set_seed와 동일 이유)
    random.seed(worker_seed)
