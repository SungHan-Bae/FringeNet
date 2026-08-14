"""Stage B 물리 손실 — 동결 TMM 디코더로 예측 두께를 관측 스펙트럼에 되비춘다.

    L = MAE(d_hat, d_true) + beta(step) * L1(R_dec(d_hat), R_obs)

조용히 깨질 수 있는 세 가지를 여기에 적는다 (계약은 CLAUDE.md).

1. **동결**: `PhysicalStack.theta`는 `nn.Parameter`라 학습 모델의 서브모듈이 되면
   `model.parameters()`를 옵티마이저에 넘기는 순간 풀린다. `FrozenDecoder`는
   `requires_grad_(False)`에 더해 **파라미터를 하나도 보유하지 않고**(상수 분광량만 버퍼)
   학습 모델의 형제로 둔다 — 체크포인트 state_dict에도 섞이지 않는다.
2. **dtype**: theta가 동결이면 (lam, n_layers, n_s)가 상수이므로 1회 계산해 complex64로
   캐시한다. 소비자 GPU의 float64 처리율이 1/32라 이 캐스팅이 물리 항의 실용성을 가른다.
3. **beta=0**: 물리 항을 그래프에 넣지 않고 진단으로만 기록한다 — 대조군의 학습 경로가
   물리 항 도입 전과 같아야 ablation 차이를 물리 항에 귀속할 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from src.calibrate import load_physical_stack
from src.data.dataset import REPO_ROOT
from src.physics.tmm import tmm_reflectance

__all__ = [
    "DEFAULT_DECODER",
    "FrozenDecoder",
    "PhysicsLoss",
    "PhysicsParts",
    "beta_at",
    "build_physics_loss",
]

N0_AIR = 1.0
# Stage A 채택 디코더 (reports/stage_a.md — 자유도 7 + Si 표 Schinke 2015).
DEFAULT_DECODER = "runs/stage_a/joint-lam3-sin2-si2-schinke/model.pt"
_PHYSICS_KEYS = frozenset({"decoder", "beta", "warmup_steps"})


class FrozenDecoder(nn.Module):
    """Stage A 체크포인트를 동결 forward 모델로 감싼다 — d (B, L) [nm] → R (B, W).

    Args:
        checkpoint: Stage A run의 `model.pt` 경로. 상대경로는 저장소 루트 기준.
        dtype: TMM 누적에 쓸 complex dtype. 학습은 complex64, 검증은 complex128.

    Attributes:
        physical_values: 물성 파라미터 현재값 (보고용).
        provenance: metrics.json에 남길 디코더 출처·게이트 수치 스냅샷.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_DECODER,
        *,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if not dtype.is_complex:
            raise ValueError(f"dtype은 complex여야 한다 (받은 값: {dtype})")
        path = Path(checkpoint)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"디코더 체크포인트가 없다: {path}")

        stack, ckpt = load_physical_stack(path)
        stack.theta.requires_grad_(False)  # 명시 동결 (계약 1겹)
        with torch.no_grad():
            lam, n_layers, ns = stack.spectra()

        real_dtype = torch.zeros((), dtype=dtype).real.dtype
        # 버퍼만 등록한다 — 이 모듈은 파라미터를 보유하지 않는다 (계약 2겹)
        self.register_buffer("lam", lam.to(real_dtype))
        self.register_buffer("n_layers", n_layers.to(dtype))
        self.register_buffer("ns", ns.to(dtype))

        self.complex_dtype = dtype
        self.n_channels = int(lam.shape[0])
        self.n_stack_layers = int(n_layers.shape[0])
        self.physical_values: dict[str, float] = dict(stack.physical_values())
        result: dict[str, Any] = ckpt.get("result", {})
        diag: dict[str, Any] = result.get("diag", {})
        self.provenance: dict[str, Any] = {
            "decoder": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
            "free": list(result.get("free", [])),
            "si_source": stack.si_source,
            "lam_range_nm": [float(lam.min()), float(lam.max())],
            "stage_a_rmse": diag.get("rmse"),
            "stage_a_violation_rate": diag.get("violation_rate"),
            "physical_values": dict(self.physical_values),
        }

    def forward(self, d: Tensor) -> Tensor:
        """두께 d: (B, L) [nm] → 재구성 반사율 R: (B, W). dtype은 버퍼의 실수 dtype."""
        return tmm_reflectance(d, self.n_layers, N0_AIR, self.ns, self.lam)

    @torch.no_grad()
    def reconstruct(self, d: Tensor, *, batch_size: int = 4096) -> Tensor:
        """평가용 배치 재구성. d: (N, L) → R: (N, W).

        중간 텐서가 (B, L, W)라 holdout 81,000행을 한 번에 넣으면 수백 MB가 되므로
        배치로 끊는다.
        """
        if len(d) == 0:
            return d.new_zeros((0, self.n_channels))
        return torch.cat([self(d[i : i + batch_size]) for i in range(0, len(d), batch_size)])

    @torch.no_grad()
    def residual_l1(self, d: Tensor, r_obs: Tensor, *, batch_size: int = 4096) -> Tensor:
        """행별 재구성 L1. d: (N, L), r_obs: (N, W) → (N,).

        라벨을 쓰지 않으므로 학습 중 진단이자 계측 신뢰도 지표다 (README §3.4).
        """
        if len(d) != len(r_obs):
            raise ValueError(f"d {len(d)}행과 r_obs {len(r_obs)}행이 다르다")
        recon = self.reconstruct(d, batch_size=batch_size)
        return (recon - r_obs.to(recon.dtype)).abs().mean(dim=1)


def beta_at(step: int, beta: float, warmup_steps: int) -> float:
    """워밍업 선형 상승 후 일정한 beta. step은 0부터 세는 전역 스텝.

    물리 항을 처음부터 켜면 무작위 초기 예측의 재구성 잔차가 지도 항을 압도한다.
    """
    if step < 0:
        raise ValueError(f"step은 0 이상이어야 한다 (받은 값: {step})")
    if warmup_steps <= 0:
        return beta
    return beta * min(1.0, step / warmup_steps)


class PhysicsParts(NamedTuple):
    """손실 분해. total만 backward 대상이고 나머지는 로깅·진단용.

    Attributes:
        total: 최적화 대상 = sup + beta * phys.
        sup: 지도 항 MAE(d_hat, d_true) [nm].
        phys: 재구성 L1 — **beta를 곱하기 전** 값이라 run 사이 직접 비교할 수 있다.
        beta: 이 스텝에 실제 적용된 beta (워밍업 반영).
    """

    total: Tensor
    sup: Tensor
    phys: Tensor
    beta: float


class PhysicsLoss:
    """`L = MAE(d_hat, d) + beta(step) * L1(R_dec(d_hat), R_obs)`.

    Args:
        decoder: 동결 디코더.
        beta: 물리 항 가중. 0이면 대조군 (진단만 기록, gradient에 미포함).
        warmup_steps: beta 선형 워밍업 스텝 수. 0이면 워밍업 없음.
    """

    def __init__(self, decoder: FrozenDecoder, *, beta: float, warmup_steps: int = 0) -> None:
        if beta < 0.0:
            raise ValueError(f"beta는 0 이상이어야 한다 (받은 값: {beta})")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps는 0 이상이어야 한다 (받은 값: {warmup_steps})")
        self.decoder = decoder
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)

    @property
    def config(self) -> dict[str, Any]:
        """metrics.json에 남길 스냅샷 (디코더 출처 포함)."""
        return {
            "beta": self.beta,
            "warmup_steps": self.warmup_steps,
            **self.decoder.provenance,
        }

    def __call__(self, d_hat: Tensor, d_true: Tensor, r_obs: Tensor, step: int) -> PhysicsParts:
        """d_hat/d_true: (B, L) [nm], r_obs: (B, W) 관측 반사율."""
        sup = l1_loss(d_hat, d_true)
        beta = beta_at(step, self.beta, self.warmup_steps)
        if beta == 0.0:
            # 대조군: 그래프에 넣지 않는다 — 학습 경로가 물리 항 도입 전과 같아야 한다.
            with torch.no_grad():
                phys = self._reconstruction_l1(d_hat.detach(), r_obs)
            return PhysicsParts(total=sup, sup=sup, phys=phys, beta=0.0)
        phys = self._reconstruction_l1(d_hat, r_obs)
        return PhysicsParts(total=sup + beta * phys, sup=sup, phys=phys, beta=beta)

    def _reconstruction_l1(self, d: Tensor, r_obs: Tensor) -> Tensor:
        recon = self.decoder(d)
        return l1_loss(recon, r_obs.to(recon.dtype))


def build_physics_loss(
    train_cfg: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
) -> PhysicsLoss | None:
    """train config의 `physics` 블록으로 손실을 만든다. 블록이 없으면 None (지도 항만).

    ```yaml
    train:
      physics:
        decoder: runs/stage_a/joint-lam3-sin2-si2-schinke/model.pt   # 생략 시 채택 디코더
        beta: 100.0            # 필수 — 대조군은 0.0을 명시한다
        warmup_steps: 3000
    ```

    Raises:
        ValueError: 모르는 키가 있거나 `beta`가 없는 경우 — 조용한 오타로 물리 항이
            빠진 run이 생기지 않게 한다.
    """
    spec = train_cfg.get("physics")
    if spec is None:
        return None
    unknown = set(spec) - _PHYSICS_KEYS
    if unknown:
        raise ValueError(f"모르는 physics 키: {sorted(unknown)} (가능: {sorted(_PHYSICS_KEYS)})")
    if "beta" not in spec:
        raise ValueError('physics 블록에 "beta"가 필요하다 — 대조군도 beta: 0.0을 명시한다')
    decoder = FrozenDecoder(spec.get("decoder", DEFAULT_DECODER), dtype=dtype)
    if device is not None:
        decoder = decoder.to(device)
    return PhysicsLoss(
        decoder,
        beta=float(spec["beta"]),
        warmup_steps=int(spec.get("warmup_steps", 0)),
    )
