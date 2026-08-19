"""ConvNeXt-1D — 현대적 conv 블록(depthwise + inverted bottleneck + 잔차)의 1D 이식.

Task 8의 구조 후보 2. 잔차 플래그를 얹은 기존 백본(CNN1D + residual)과 달리 블록
설계 자체를 ConvNeXt(Liu et al. 2022) 규약으로 바꾼다: depthwise conv(공간 혼합)와
pointwise MLP(채널 혼합)를 분리하고, 블록마다 identity 잔차 + layer scale을 쓴다.

이 태스크에 맞춘 원본과의 차이 두 가지 (근거는 level1_cnn 확정 결론):
- **stem이 stride 1이다** — 원본의 stride-4 patchify는 파장축 해상도를 입구에서 버리는데,
  이 태스크의 정보는 fringe의 파장축 절대 위치·위상에 실려 있다 (GAP가 flatten보다
  4배 나쁜 이유와 같은 축). 다운샘플은 스테이지 사이에서만 한다.
- **head는 flatten** — GAP 계열(원본의 global pool)은 위치 정보를 붕괴시켜 기각됐다.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.data.dataset import N_CHANNELS, N_LAYERS
from src.models.heads import ThicknessBound


class ChannelLayerNorm(nn.Module):
    """(B, C, W) 텐서의 채널축(C) LayerNorm — 파장 위치마다 독립 정규화.

    GroupNorm(1, C)는 C·W를 함께 정규화하므로 다르다 — ConvNeXt 규약은 위치별
    채널 정규화다 (원본의 channels_first LayerNorm).

    Shapes:
        forward: x (B, C, W) float -> (B, C, W) float
    """

    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt 블록의 1D 판: dwconv(k) -> LN -> Linear(x ratio) -> GELU -> Linear -> gamma -> +x.

    공간 혼합(depthwise conv)과 채널 혼합(위치별 MLP)을 분리한 inverted bottleneck.
    입출력 shape가 같아 잔차가 항상 identity다 (projection 없음 — 다운샘플은 스테이지
    사이의 별도 층이 맡는다). gamma(layer scale)는 블록 기여를 작게 시작시켜 깊은
    스택의 학습을 안정화한다.

    Shapes:
        forward: x (B, C, W) float -> (B, C, W) float
    """

    def __init__(self, dim: int, kernel_size: int, mlp_ratio: int, layer_scale_init: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        out = self.dwconv(x)
        out = out.transpose(1, 2)  # (B, C, W) -> (B, W, C): LN·Linear는 마지막 축에 작용
        out = self.pwconv2(self.act(self.pwconv1(self.norm(out))))
        out = (self.gamma * out).transpose(1, 2)
        return x + out


class ConvNeXt1D(nn.Module):
    """반사율 스펙트럼 -> 4층 두께 회귀. ConvNeXt-1D 백본 (Task 8 구조 후보).

    구조: (B, 226) -> stem Conv1d(1->dims[0], stride 1) -> [스테이지: 블록 x depth]
    -> (스테이지 사이 LN + Conv1d stride 2 다운샘플) -> LN -> flatten -> Linear -> (B, 4).

    수용영역은 스테이지가 내려갈수록 stride 배수만큼 빨리 자란다:
    RF = 1 + (k_stem - 1) + sum_i depth_i * (k - 1) * s_i + sum(다운샘플 (2-1) * s_i)
    (s_i = 스테이지 i 입장에서의 누적 stride). 기본 설정은 RF가 전 대역(226)을 덮도록
    고른다 — level1_cnn에서 RF가 대역을 못 덮으면 실패했다 (dilated 이전의 RF 97).

    Args:
        n_channels: 입력 스펙트럼 채널 수.
        n_outputs: 출력 두께 개수 (= 층 수).
        dims: 스테이지별 채널 폭.
        depths: 스테이지별 블록 수 (dims와 길이가 같아야 한다).
        kernel_size: depthwise conv 커널 (홀수 — 길이 보존 padding).
        mlp_ratio: pointwise MLP 확장비 (ConvNeXt 기본 4).
        layer_scale_init: gamma 초기값 (ConvNeXt 기본 1e-6).
        output_bound: True면 sigmoid bound로 출력을 [d_min, d_max] nm에 가둔다.
            False면 head bias를 범위 중앙으로 초기화한다 (다른 모델과 같은 규약).
        d_min / d_max: 물리 두께 범위 [nm].

    Shapes:
        forward: x (B, 226) float 반사율 -> (B, 4) float 두께 [nm]
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_outputs: int = N_LAYERS,
        dims: Sequence[int] = (40, 80, 120, 160),
        depths: Sequence[int] = (2, 2, 4, 2),
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        output_bound: bool = True,
        d_min: float = 10.0,
        d_max: float = 300.0,
    ) -> None:
        super().__init__()
        if len(dims) == 0 or any(d <= 0 for d in dims):
            raise ValueError(f"dims는 비어있지 않은 양수 목록이어야 한다 (받은 값: {list(dims)})")
        if len(depths) != len(dims):
            raise ValueError(f"depths 길이({len(depths)})는 dims 길이({len(dims)})와 같아야 한다")
        if any(d < 1 for d in depths):
            raise ValueError(f"depths는 전부 1 이상이어야 한다 (받은 값: {list(depths)})")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size는 홀수 양수여야 한다 (받은 값: {kernel_size})")
        if mlp_ratio < 1:
            raise ValueError(f"mlp_ratio는 1 이상이어야 한다 (받은 값: {mlp_ratio})")
        if not d_min < d_max:
            raise ValueError(f"d_min < d_max 여야 한다 (받은 값: {d_min}, {d_max})")

        self.n_channels = int(n_channels)
        dims = [int(d) for d in dims]
        depths = [int(d) for d in depths]

        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size, padding=kernel_size // 2),
            ChannelLayerNorm(dims[0]),
        )
        stages: list[nn.Module] = []
        w_last = self.n_channels
        for i, (dim, depth) in enumerate(zip(dims, depths, strict=True)):
            layers: list[nn.Module] = []
            if i > 0:
                layers += [ChannelLayerNorm(dims[i - 1]), nn.Conv1d(dims[i - 1], dim, 2, stride=2)]
                w_last //= 2  # k=2, stride 2, padding 0 -> floor(W/2)
            layers += [
                ConvNeXtBlock1D(dim, kernel_size, mlp_ratio, layer_scale_init) for _ in range(depth)
            ]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.final_norm = ChannelLayerNorm(dims[-1])

        self.head = nn.Linear(dims[-1] * w_last, int(n_outputs))
        self.bound: ThicknessBound | None = None
        if output_bound:
            self.bound = ThicknessBound(d_min, d_max)
        else:
            nn.init.constant_(self.head.bias, (d_min + d_max) / 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"입력은 (B, {self.n_channels}) 여야 한다 (받은 shape: {tuple(x.shape)})"
            )
        feat = self.final_norm(self.stages(self.stem(x.unsqueeze(1))))
        out = self.head(feat.flatten(1))
        if self.bound is not None:
            out = self.bound(out)
        return out
