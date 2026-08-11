"""Stage B 물리 디코더 — Stage A 캘리브레이션 결과를 동결한 d → R 재구성 모듈.

Stage A(src/calibrate.py)가 피팅한 CalibratedStack에서 TMM 입력 물리량
(λ 그리드, 층 굴절률, 기판 굴절률)을 한 번 뽑아 **buffer로 동결**한다.
학습 파라미터가 없으므로 옵티마이저에 잡히지 않고, gradient는 두께 d로만 흐른다 —
Stage B 물리 손실 L1(R_dec(d_hat), R_obs)의 디코더 역할.

dtype 계약 (CLAUDE.md): 검증·캘리브레이션은 complex128, **학습은 complex64**.
기본값 complex64는 Stage A의 float64 물리량을 캐스팅해 담는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.physics.tmm import tmm_reflectance

_REAL_OF_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


class TMMDecoder(nn.Module):
    """동결된 캘리브레이션 물리량으로 d → R을 계산하는 미분가능 디코더.

    Args:
        lam: (W,) real — 학습된 λ 그리드 [nm] (채널 순서).
        n_layers: (4, W) complex — 층별 굴절률 (SiN/SiO₂/SiN/SiO₂).
        ns: (W,) complex — Si 기판 굴절률 (n − i·k).
        dtype: complex64(학습 기본) | complex128(검증용).

    Shapes:
        forward: d (B, 4) real [nm] → R (B, W) real (dtype에 대응하는 실수형).
    """

    def __init__(
        self,
        lam: Tensor,
        n_layers: Tensor,
        ns: Tensor,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if dtype not in _REAL_OF_COMPLEX:
            raise TypeError(f"dtype은 complex64 | complex128 이어야 한다 (받은 값: {dtype})")
        if lam.ndim != 1 or n_layers.shape != (4, lam.shape[0]) or ns.shape != lam.shape:
            raise ValueError(
                f"shape 규약 위반: lam {tuple(lam.shape)}, n_layers {tuple(n_layers.shape)},"
                f" ns {tuple(ns.shape)} — (W,), (4, W), (W,) 이어야 한다"
            )
        rdtype = _REAL_OF_COMPLEX[dtype]
        self.register_buffer("lam", lam.detach().to(rdtype))
        self.register_buffer("n_layers", n_layers.detach().to(dtype))
        self.register_buffer("ns", ns.detach().to(dtype))

    def forward(self, d: Tensor) -> Tensor:
        """두께 d: (B, 4) [nm] → 재구성 반사율 R: (B, W)."""
        return tmm_reflectance(d, self.n_layers, 1.0, self.ns, self.lam)


def load_tmm_decoder(
    run_dir: Path | str, dtype: torch.dtype = torch.complex64
) -> tuple[TMMDecoder, dict[str, Any]]:
    """Stage A run 디렉토리의 model.pt에서 동결 디코더를 만든다.

    Args:
        run_dir: 캘리브레이션 산출물 디렉토리 (예: runs/stage_a/sio2-freeze-refine).
        dtype: TMMDecoder의 complex dtype.

    Returns:
        (decoder, meta) — meta는 출처 기록용
        {"run_dir", "step", "val_rmse"} (metrics.json/로그에 남긴다).
    """
    # 지연 import: calibrate가 train_gpu를, train_gpu가 이 모듈을 import하므로
    # 모듈 수준에서 import하면 순환이 된다.
    from src.calibrate import load_calibrated_stack

    run_dir = Path(run_dir)
    stack, ckpt = load_calibrated_stack(run_dir / "model.pt")
    with torch.no_grad():
        lam, n_layers, ns = stack.spectra()
    meta = {
        "run_dir": str(run_dir),
        "step": int(ckpt["step"]),
        "val_rmse": float(ckpt["val_rmse"]),
    }
    return TMMDecoder(lam, n_layers, ns, dtype=dtype), meta
