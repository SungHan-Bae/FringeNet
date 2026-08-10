"""대회 1등 솔루션 skip-connection MLP — strong baseline(원본 충실 재현)용.

출처: [1등][Context_KKP] Skipconnection MLP with Ensemble
(https://dacon.io/competitions/official/235554/codeshare/651). 코드 원문은
위키북스 공식 저장소 https://github.com/wikibook/dacon 의 ch02/src/model.py
``SkipConnectionModel`` (메인 단일 모델, 보고 val MAE ≈ 0.42 nm).

원본과의 수식 동일성 (재현 근거):
- 원본의 손 구현 GELU는 tanh 근사식 그대로라 ``nn.GELU(approximate="tanh")``와 같고,
  손 구현 LayerNorm(TF식, eps=1e-5)도 ``nn.LayerNorm(dim, eps=1e-5)``와 같은 수식이다.
- 원본은 ``nn.Dropout(0.1)``을 정의만 하고 forward에서 호출하지 않는다 — 여기서는
  아예 넣지 않는다 (누락이 아니라 원본 동작의 충실 재현).
- 가중치 초기화는 원본과 같이 PyTorch 기본값 그대로 둔다.

구조 (기본 인자 = 원본, 파라미터 213,208,104개 — tests/test_models.py에서 고정):
- up 경로: 226 -> 2000 -> 4000 -> 7000 -> 10000. 블록 = Linear -> GELU -> BatchNorm1d.
- down 경로: 10000 -> 7000 -> 4000 -> 2000 -> 300. 각 down 블록 **입구에 LayerNorm**,
  출구에서 같은 폭의 up 출력과 덧셈 skip (마지막 down -> head_dim 300은 skip 없음).
- head: Linear(300, 4). output bound 없음(bare regression), 입력 표준화 없음.

학습 프로토콜 쪽 재현 항목(AdamW eps 1e-6, 셔플 1회 고정, 에폭 2부터 eval 모드 학습)은
src/train_gpu.py의 config 플래그가 담당한다 — configs/strong_baseline/ 참조.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from src.data.dataset import N_CHANNELS, N_LAYERS


class WinnerSkipMLP(nn.Module):
    """1등 솔루션의 U자형 skip-connection MLP.

    Args:
        n_channels: 입력 스펙트럼 채널 수.
        n_outputs: 출력 두께 개수 (= 층 수).
        up_dims: up 경로 폭 (마지막 값이 병목 폭). down 경로는 이를 역순으로 내려온다.
            최소 2개 — skip connection이 성립하려면 up 중간 출력이 있어야 한다.
        head_dim: 마지막 down 블록의 출력 폭 (head Linear의 입력 폭, 원본 300).

    Shapes:
        forward: x (B, 226) float 반사율 -> (B, 4) float 두께 [nm]
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_outputs: int = N_LAYERS,
        up_dims: Sequence[int] = (2000, 4000, 7000, 10000),
        head_dim: int = 300,
    ) -> None:
        super().__init__()
        dims = [int(d) for d in up_dims]
        if len(dims) < 2:
            raise ValueError(f"up_dims는 2개 이상이어야 한다 (받은 값: {dims})")
        if any(d <= 0 for d in dims) or head_dim <= 0:
            raise ValueError(f"폭은 전부 양수여야 한다 (받은 값: {dims}, head_dim={head_dim})")

        self.n_channels = int(n_channels)

        def block(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_out), nn.GELU(approximate="tanh"), nn.BatchNorm1d(d_out)
            )

        up_in = [self.n_channels, *dims[:-1]]
        self.ups = nn.ModuleList(
            block(d_in, d_out) for d_in, d_out in zip(up_in, dims, strict=True)
        )
        down_in = dims[::-1]  # 원본: [10000, 7000, 4000, 2000]
        down_out = [*dims[-2::-1], int(head_dim)]  # 원본: [7000, 4000, 2000, 300]
        self.norms = nn.ModuleList(nn.LayerNorm(d, eps=1e-5) for d in down_in)
        self.downs = nn.ModuleList(
            block(d_in, d_out) for d_in, d_out in zip(down_in, down_out, strict=True)
        )
        self.head = nn.Linear(int(head_dim), int(n_outputs))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"입력은 (B, {self.n_channels}) 여야 한다 (받은 shape: {tuple(x.shape)})"
            )
        up_outs: list[Tensor] = []
        h = x
        for up in self.ups:
            h = up(h)
            up_outs.append(h)
        # down_i 출력에 더할 up 출력들 — 원본: down1+up3, down2+up2, down3+up1, down4는 없음
        skips = up_outs[-2::-1]
        for i, (norm, down) in enumerate(zip(self.norms, self.downs, strict=True)):
            h = down(norm(h))
            if i < len(skips):
                h = h + skips[i]
        return self.head(h)
