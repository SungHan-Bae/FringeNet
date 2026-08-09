"""출력 head 모듈 — 모델들이 공유하는 마지막 단(출력 bound 등)을 모아 둔다."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ThicknessBound(nn.Module):
    """sigmoid로 출력을 물리 두께 범위 [d_min, d_max] nm에 가두는 출력 bound.

    물리적으로 불가능한 두께 예측을 구조적으로 배제한다 (README §3.1).
    기본 범위 [10, 300]은 데이터 검증(Task 2)에서 확정된 두께 격자와 일치한다.
    경계값은 수학적으로는 점근적으로만 도달하지만(sigmoid ∈ (0, 1)), float 포화로는
    도달 가능하다. 격자 끝 두께(10, 300 nm)에서 생길 수 있는 잔차 편향은
    bound on/off ablation(Task 5)에서 확인한다.

    Shapes:
        z: (B, L) float 로짓 -> (B, L) float ∈ [d_min, d_max] [nm]
    """

    def __init__(self, d_min: float = 10.0, d_max: float = 300.0) -> None:
        super().__init__()
        if not d_min < d_max:
            raise ValueError(f"d_min < d_max 여야 한다 (받은 값: {d_min}, {d_max})")
        self.d_min = float(d_min)
        self.d_max = float(d_max)

    def forward(self, z: Tensor) -> Tensor:
        return self.d_min + torch.sigmoid(z) * (self.d_max - self.d_min)

    def extra_repr(self) -> str:
        return f"d_min={self.d_min}, d_max={self.d_max}"
