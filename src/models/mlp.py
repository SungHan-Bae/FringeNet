"""Baseline MLP — 226채널을 순서 없는 피처로 취급하는 대조군.

1D CNN(Level 1)이 쓰는 "채널 순서 = 연속 스펙트럼" 구조 bias를 일부러 쓰지 않는다.
Level 1과의 성능 차이가 곧 구조 bias의 기여가 되도록 하는 기준선이다 (CLAUDE.md 모델 스펙).
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from src.data.dataset import N_CHANNELS, N_LAYERS
from src.models.heads import ThicknessBound

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """반사율 스펙트럼 -> 4층 두께 회귀 baseline.

    Args:
        n_channels: 입력 스펙트럼 채널 수.
        n_outputs: 출력 두께 개수 (= 층 수).
        hidden_dims: 은닉층 폭. 빈 시퀀스면 선형 회귀로 퇴화한다.
        activation: 은닉층 활성화 — "relu" | "gelu" | "silu" | "tanh".
        dropout: 각 은닉층 뒤 dropout 확률. 0이면 층 자체를 넣지 않는다.
        output_bound: True면 sigmoid bound로 출력을 [d_min, d_max] nm에 가둔다.
            False면 선형 출력 그대로 두되, 마지막 bias를 범위 중앙 (d_min+d_max)/2로
            초기화한다 — bound 유무와 무관하게 학습 시작점이 "범위 중앙 예측"으로
            같아야 ablation이 공정하다. (bound가 있으면 로짓 0 → sigmoid 0.5가
            자동으로 중앙이지만, 선형 출력이 0 근처에서 출발하면 Adam의 스텝 크기가
            lr로 정규화되는 탓에 bias가 155까지 가는 데만 수만 스텝을 쓴다.)
        d_min / d_max: 물리 두께 범위 [nm] (데이터 격자 10~300).
            output_bound=False일 때도 bias 초기화 중앙값 계산에 쓰인다.

    Shapes:
        forward: x (B, 226) float 반사율 -> (B, 4) float 두께 [nm]
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_outputs: int = N_LAYERS,
        hidden_dims: Sequence[int] = (512, 512, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        output_bound: bool = True,
        d_min: float = 10.0,
        d_max: float = 300.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation은 {sorted(_ACTIVATIONS)} 중 하나여야 한다 (받은 값: {activation!r})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout은 [0, 1) 범위여야 한다 (받은 값: {dropout})")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims는 전부 양수여야 한다 (받은 값: {list(hidden_dims)})")
        if not d_min < d_max:
            raise ValueError(f"d_min < d_max 여야 한다 (받은 값: {d_min}, {d_max})")

        self.n_channels = int(n_channels)
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        width_in = self.n_channels
        for width in hidden_dims:
            layers.append(nn.Linear(width_in, width))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width_in = width

        head = nn.Linear(width_in, int(n_outputs))
        layers.append(head)
        if output_bound:
            layers.append(ThicknessBound(d_min, d_max))
        else:
            nn.init.constant_(head.bias, (d_min + d_max) / 2)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"입력은 (B, {self.n_channels}) 여야 한다 (받은 shape: {tuple(x.shape)})"
            )
        return self.net(x)
