"""Level 1 — 1D CNN: "채널 순서 = 연속 스펙트럼" 구조 bias를 쓰는 모델.

baseline MLP(src/models/mlp.py)와의 성능 차이가 곧 구조 bias의 기여가 되도록,
블록 구성(Conv -> Norm -> Activation -> Dropout)·활성화·정규화·출력 규약
(bare head + bias 중앙 초기화 / sigmoid bound)을 MLP와 같게 두고 **연결 패턴만**
바꾼다 — dense·순서 무시(Linear) -> local·순서 사용(Conv1d).

변인 통제 장치 (Task 5, MLP vs CNN 비교의 공정성):
- 기본 channels는 파라미터 수를 baseline MLP(647,172)와 ±10% 안에서 맞춘 값이다.
  용량(capacity)이 아니라 구조가 성능 차의 원인이라고 말할 수 있게 하기 위함.
- ``channel_shuffle_seed``: 고정 무작위 순열로 입력 채널 순서를 파괴하는 대조군 플래그.
  MLP는 입력 순열에 불변이므로(첫 Linear의 열 순서만 바뀜 — iid 초기화 분포 동일),
  순열로 성능이 떨어지는 것은 CNN뿐이고, 그 낙폭이 "스펙트럼 순서 정보"의 기여다.
- ``kernel_sizes``에 커널을 여러 개 주면 다중 스케일(병렬 분기 concat), 하나면 단일
  스케일 — 단일 vs 다중 스케일 ablation도 이 플래그 하나로 돈다.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.data.dataset import N_CHANNELS, N_LAYERS
from src.models.heads import ThicknessBound

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

_NORM_NAMES = ("batchnorm", "layernorm", "none")


def _make_norm(norm: str, n_channels: int) -> nn.Module | None:
    """conv 텐서 (B, C, W)용 정규화 층. layernorm은 GroupNorm(1, C)로 대응한다
    (채널+파장축 전체 정규화 — Linear 블록의 LayerNorm에 해당하는 conv 버전)."""
    if norm == "batchnorm":
        return nn.BatchNorm1d(n_channels)
    if norm == "layernorm":
        return nn.GroupNorm(1, n_channels)
    return None


class ConvBlock(nn.Module):
    """다중 커널 병렬 conv 블록: [Conv1d(k) | k in kernel_sizes] concat -> Norm -> Act -> Dropout.

    kernel_sizes가 하나면 평범한 단일 conv와 같다. 여러 개면 c_out을 커널 수로 균등
    분할해 분기별로 계산한 뒤 채널축으로 concat한다 — fringe 주기가 두께에 따라
    변하므로 여러 수용영역을 섞는 다중 스케일 장치 (CLAUDE.md 모델 스펙).
    dilation은 커널 파라미터 수를 바꾸지 않고 수용영역만 d배 넓힌다 — 다중 스케일의
    또 다른 축 (kernel 혼합과 독립적으로 조합 가능).
    모든 커널은 홀수여야 한다: 유효 span = d*(k-1)+1 이 항상 홀수가 되어
    padding = d*(k//2)로 분기 간 출력 길이가 정확히 같다 (= ceil(W/stride), 커널 무관).

    Shapes:
        forward: x (B, C_in, W) float -> (B, C_out, ceil(W/stride)) float
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_sizes: Sequence[int],
        stride: int,
        activation: str,
        norm: str,
        dropout: float,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        n_branch = len(kernel_sizes)
        if c_out % n_branch != 0:
            raise ValueError(
                f"c_out({c_out})은 커널 수({n_branch})로 나누어떨어져야 한다"
                f" (분기별 채널 균등 분할)"
            )
        c_branch = c_out // n_branch
        self.branches = nn.ModuleList(
            nn.Conv1d(
                c_in, c_branch, k, stride=stride, padding=dilation * (k // 2), dilation=dilation
            )
            for k in kernel_sizes
        )

        tail: list[nn.Module] = []
        norm_layer = _make_norm(norm, c_out)
        if norm_layer is not None:
            tail.append(norm_layer)
        tail.append(_ACTIVATIONS[activation]())
        if dropout > 0.0:
            tail.append(nn.Dropout1d(dropout))
        self.tail = nn.Sequential(*tail)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.tail(out)


class CNN1D(nn.Module):
    """반사율 스펙트럼 -> 4층 두께 회귀. 채널 순서를 쓰는 Level 1 모델.

    구조: (B, 226) -> (B, 1, 226) -> ConvBlock 스택 -> head(gap|flatten) -> Linear -> (B, 4).

    head 선택이 이 모델의 핵심 ablation 축이다 (level1_cnn 첫 결과의 교훈):
    - "gap": 파장축을 평균으로 붕괴 — 위치 불변 특징만 남는다. 첫 실험에서 MLP 대비
      4배 나쁜 18.16 nm, 심지어 채널 셔플 대조군(12.23 nm)보다도 나빴다.
      이 태스크의 정보가 fringe의 파장축 절대 위치·위상에 실려 있기 때문으로 해석.
    - "flatten": (C, W_last)를 펴서 위치 정보를 보존한 채 회귀 — 국소성(conv)은
      유지하고 위치 불변성만 제거하는 가설 검증용.

    Args:
        n_channels: 입력 스펙트럼 채널 수.
        n_outputs: 출력 두께 개수 (= 층 수).
        channels: 블록별 출력 채널 수. 기본값 (32, 64, 128, 200, 280)은 gap 헤드 기준
            파라미터 수 646,340으로 baseline MLP 512x3(646,660)과 -0.05% 차이 — 용량
            통제용. flatten 헤드는 662,020(+2.4%)으로 여전히 ±10% 안이다.
        strides: 블록별 stride (channels와 길이가 같아야 한다). 기본 (1, 2, 2, 2, 2):
            첫 블록은 전체 226 해상도에서 보고, 이후 절반씩 다운샘플 (226->113->57->29->15).
        kernel_sizes: conv 커널 크기 (전 블록 공유, 전부 홀수). 하나면 단일 스케일,
            여러 개면 병렬 분기 concat 다중 스케일 (ConvBlock 참조).
        dilations: 블록별 dilation (None이면 전부 1). 파라미터·연산량을 바꾸지 않고
            수용영역만 넓힌다 — 예: 기본 stride에서 (1,1,1,1,1) -> RF 97채널,
            (1,2,4,4,2) -> RF 259채널(전 대역). 블록 입력 길이보다 유효 span이
            커지지 않게 고를 것 (padding이 지배하면 낭비).
        activation / norm / dropout: 블록 구성 — MLP 블록과 같은 의미·같은 선택지.
            dropout은 채널 단위 Dropout1d.
        head: "gap"(파장축 평균) | "flatten"(파장축 보존). 위 설명 참조.
        output_bound: True면 sigmoid bound로 출력을 [d_min, d_max] nm에 가둔다.
            False면 선형 출력 그대로 두되 head bias를 범위 중앙으로 초기화한다
            (MLP와 같은 규약 — 학습 시작점을 "범위 중앙 예측"으로 통일해야 공정).
        d_min / d_max: 물리 두께 범위 [nm].
        channel_shuffle_seed: None이면 입력 그대로. 정수를 주면 그 시드의 고정 무작위
            순열로 채널 순서를 파괴한다 (buffer로 저장 — 체크포인트에 같이 남는다).
            "스펙트럼 순서 정보"의 기여를 분리하는 대조군용.

    Shapes:
        forward: x (B, 226) float 반사율 -> (B, 4) float 두께 [nm]
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_outputs: int = N_LAYERS,
        channels: Sequence[int] = (32, 64, 128, 200, 280),
        strides: Sequence[int] = (1, 2, 2, 2, 2),
        kernel_sizes: Sequence[int] = (7,),
        dilations: Sequence[int] | None = None,
        activation: str = "gelu",
        norm: str = "batchnorm",
        dropout: float = 0.0,
        head: str = "gap",
        output_bound: bool = True,
        d_min: float = 10.0,
        d_max: float = 300.0,
        channel_shuffle_seed: int | None = None,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation은 {sorted(_ACTIVATIONS)} 중 하나여야 한다 (받은 값: {activation!r})"
            )
        if norm not in _NORM_NAMES:
            raise ValueError(f"norm은 {sorted(_NORM_NAMES)} 중 하나여야 한다 (받은 값: {norm!r})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout은 [0, 1) 범위여야 한다 (받은 값: {dropout})")
        if len(channels) == 0 or any(c <= 0 for c in channels):
            raise ValueError(
                f"channels는 비어있지 않은 양수 목록이어야 한다 (받은 값: {list(channels)})"
            )
        if len(strides) != len(channels):
            raise ValueError(
                f"strides 길이({len(strides)})는 channels 길이({len(channels)})와 같아야 한다"
            )
        if any(s < 1 for s in strides):
            raise ValueError(f"strides는 전부 1 이상이어야 한다 (받은 값: {list(strides)})")
        if len(kernel_sizes) == 0 or any(k < 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError(
                f"kernel_sizes는 전부 홀수 양수여야 한다 (받은 값: {list(kernel_sizes)})"
            )
        if dilations is None:
            dilations = (1,) * len(channels)
        if len(dilations) != len(channels):
            raise ValueError(
                f"dilations 길이({len(dilations)})는 channels 길이({len(channels)})와 같아야 한다"
            )
        if any(d < 1 for d in dilations):
            raise ValueError(f"dilations는 전부 1 이상이어야 한다 (받은 값: {list(dilations)})")
        if head not in ("gap", "flatten"):
            raise ValueError(f'head는 "gap" | "flatten" 이어야 한다 (받은 값: {head!r})')
        if not d_min < d_max:
            raise ValueError(f"d_min < d_max 여야 한다 (받은 값: {d_min}, {d_max})")

        self.n_channels = int(n_channels)

        self.channel_perm: Tensor | None
        if channel_shuffle_seed is None:
            self.channel_perm = None
        else:
            perm = torch.randperm(
                self.n_channels,
                generator=torch.Generator().manual_seed(int(channel_shuffle_seed)),
            )
            self.register_buffer("channel_perm", perm)

        blocks: list[nn.Module] = []
        c_in = 1
        for c_out, stride, dilation in zip(channels, strides, dilations, strict=True):
            blocks.append(
                ConvBlock(
                    c_in,
                    int(c_out),
                    kernel_sizes,
                    int(stride),
                    activation,
                    norm,
                    dropout,
                    dilation=int(dilation),
                )
            )
            c_in = int(c_out)
        self.blocks = nn.Sequential(*blocks)

        self.head_mode = head
        w_last = self.n_channels  # 홀수 유효 span + padding = d*(k//2) 이므로 ceil(W/s)만 남는다
        for s in strides:
            w_last = -(-w_last // int(s))
        head_in = c_in if head == "gap" else c_in * w_last
        self.head = nn.Linear(head_in, int(n_outputs))
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
        if self.channel_perm is not None:
            x = x[:, self.channel_perm]
        feat = self.blocks(x.unsqueeze(1))  # (B, 1, 226) -> (B, C_last, W_last)
        # gap: 파장축 평균 (위치 정보 소실) / flatten: 파장축 보존
        pooled = feat.mean(dim=-1) if self.head_mode == "gap" else feat.flatten(1)
        out = self.head(pooled)
        if self.bound is not None:
            out = self.bound(out)
        return out
