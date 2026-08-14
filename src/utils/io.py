"""산출물 저장 유틸.

`src/calibrate.py`(Stage A, CPU)와 `src/train_gpu.py`(학습, GPU)가 같이 쓰므로 여기에 둔다
— 전자가 후자에서 가져오면 의존 방향이 뒤집혀 순환 import이 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ["atomic_save"]


def atomic_save(obj: dict[str, Any], path: Path) -> None:
    """torch.save를 임시파일→교체로 수행 — 저장 도중 세션이 죽어도 파일이 깨지지 않는다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
