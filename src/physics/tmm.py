"""미분가능 Transfer Matrix Method (TMM) — FringeNet의 물리 코어.

수직입사, 등방성, 평행 평면층, 표면 거칠기 없음을 가정한 다층막 반사율 계산.
Abelès 특성행렬을 Macleod 관례로 구현하며, 입사측 층부터 순서대로 곱한다.

    delta_j = 2*pi * n_j * d_j / lambda

    M_j = [[cos(delta_j),        i*sin(delta_j)/n_j],
           [i*n_j*sin(delta_j),  cos(delta_j)      ]]

    (B, C)^T = (M_1 @ M_2 @ ... @ M_L) @ (1, n_s)^T
    r = (n0*B - C) / (n0*B + C)
    R = |r|^2
    T = 4 * n0 * Re(n_s) / |n0*B + C|^2

굴절률 부호 관례
----------------
복소 굴절률은 ``n = n' - i*k`` (k >= 0 이 흡수)로 쓴다. 층을 지나는 파가
``exp(-i*delta)`` 위상을 얻는 이 관례에서 k > 0 이면 진폭이 ``exp(-2*pi*k*d/lam)``
로 감쇠한다. 부호를 뒤집으면 이득 매질이 된다.

파장 독립성
-----------
서로 다른 파장 채널끼리는 간섭하지 않으므로 R(lambda)는 채널별로 완전히 독립이다.
따라서 W축은 전부 브로드캐스트로 처리하고, 파이썬 루프는 층 수 L에 대해서만 돈다
(L=4 수준으로 작고, 행렬곱은 순서가 있어 순차 처리가 불가피하다).

Shape 규약
----------
    d:        (B, L) real   — 두께. autograd가 흐르도록 실수로 유지한다.
    n_layers: (L, W) complex — 층별·채널별 굴절률.
    n0:       scalar 또는 (W,) — 입사 매질(공기). 실수 가정.
    ns:       scalar 또는 (W,) — 기판. 복소 허용.
    lam:      (W,) real     — 파장. d와 같은 길이 단위.
    반환 R:   (B, W) real
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

__all__ = [
    "stack_amplitudes",
    "tmm_reflectance",
    "tmm_reflectance_jacobian",
    "tmm_rt",
    "tmm_transmittance",
]

_REAL_OF_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


class StackAmplitudes(NamedTuple):
    """스택 특성행렬에서 나온 진폭들. 모두 (B, W) complex.

    Attributes:
        b: 특성행렬 B 성분 — (M_1...M_L) @ (1, ns)^T 의 첫 성분.
        c: 특성행렬 C 성분 — 같은 벡터의 둘째 성분.
        n0: 브로드캐스트된 입사 매질 굴절률.
        ns: 브로드캐스트된 기판 굴절률.
    """

    b: torch.Tensor
    c: torch.Tensor
    n0: torch.Tensor
    ns: torch.Tensor


def _real_dtype_of(complex_dtype: torch.dtype) -> torch.dtype:
    """complex dtype에 대응하는 real dtype을 준다."""
    try:
        return _REAL_OF_COMPLEX[complex_dtype]
    except KeyError:
        raise TypeError(
            f"n_layers는 complex64 또는 complex128이어야 한다 (받은 dtype: {complex_dtype})"
        ) from None


def _as_complex(
    value: float | complex | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
    w: int,
) -> torch.Tensor:
    """스칼라 또는 (W,) 입력을 (B, W)에 브로드캐스트되는 complex 텐서로 정규화한다."""
    tensor = (
        value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=dtype, device=device)
    )
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1 and tensor.shape[0] in (1, w):
        return tensor
    raise ValueError(
        f"{name}는 스칼라 또는 shape ({w},) 여야 한다 (받은 shape: {tuple(tensor.shape)})"
    )


class _StackTerms(NamedTuple):
    """층 특성행렬 성분과 도함수에 필요한 상수. m11~m22는 (B, L, W) complex.

    Attributes:
        m11, m12, m21, m22: 층별 특성행렬 M_j의 네 성분.
        n_layers: (1, L, W) complex — 브로드캐스트용 층 굴절률.
        wave: (W,) real — 2*pi/lambda. delta = n_j * d_j * wave 이므로 d에 대한
            위상 도함수가 k_j = n_j * wave 다 (해석적 야코비안이 쓴다).
        n0, ns: 브로드캐스트된 입사·기판 굴절률.
        rdtype: complex dtype에 대응하는 실수 dtype.
    """

    m11: torch.Tensor
    m12: torch.Tensor
    m21: torch.Tensor
    m22: torch.Tensor
    n_layers: torch.Tensor
    wave: torch.Tensor
    n0: torch.Tensor
    ns: torch.Tensor
    rdtype: torch.dtype


def _stack_terms(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> _StackTerms:
    """shape 규약을 검증하고 층별 특성행렬 성분을 만든다 — 전 진입점의 단일 출발점.

    cos/sin이 이 계산의 지배 비용이므로, R과 야코비안이 같은 항을 공유하도록 여기서
    한 번만 계산한다.

    Raises:
        ValueError: shape 규약이 어긋난 경우.
        TypeError: n_layers가 complex dtype이 아닌 경우.
    """
    if n_layers.ndim != 2:
        raise ValueError(f"n_layers는 (L, W) 여야 한다 (받은 shape: {tuple(n_layers.shape)})")
    if d.ndim != 2:
        raise ValueError(f"d는 (B, L) 여야 한다 (받은 shape: {tuple(d.shape)})")
    if lam.ndim != 1:
        raise ValueError(f"lam은 (W,) 여야 한다 (받은 shape: {tuple(lam.shape)})")

    n_l, n_w = n_layers.shape
    if d.shape[1] != n_l:
        raise ValueError(f"d의 층 수 {d.shape[1]}가 n_layers의 층 수 {n_l}와 다르다")
    if lam.shape[0] != n_w:
        raise ValueError(f"lam의 채널 수 {lam.shape[0]}가 n_layers의 채널 수 {n_w}와 다르다")
    if n_l == 0:
        raise ValueError("층이 최소 1개는 있어야 한다 (무층 극한은 d=0으로 표현한다)")

    cdtype = n_layers.dtype
    rdtype = _real_dtype_of(cdtype)
    device = n_layers.device

    d = d.to(device=device, dtype=rdtype)
    lam = lam.to(device=device, dtype=rdtype)
    n0_c = _as_complex(n0, dtype=cdtype, device=device, name="n0", w=n_w)
    ns_c = _as_complex(ns, dtype=cdtype, device=device, name="ns", w=n_w)

    wave = (2.0 * math.pi) / lam
    # delta: (B, L, W). d는 실수인 채로 complex와 곱해져 autograd가 d까지 흐른다.
    delta = n_layers.unsqueeze(0) * d.unsqueeze(-1) * wave

    cos_d = torch.cos(delta)
    i_sin = 1j * torch.sin(delta)
    n_b = n_layers.unsqueeze(0)  # (1, L, W)

    # M_j 의 네 성분. 각각 (B, L, W).
    return _StackTerms(
        m11=cos_d,
        m12=i_sin / n_b,
        m21=i_sin * n_b,
        m22=cos_d,
        n_layers=n_b,
        wave=wave,
        n0=n0_c,
        ns=ns_c,
        rdtype=rdtype,
    )


def stack_amplitudes(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> StackAmplitudes:
    """스택 특성행렬을 기판 벡터에 적용해 (B, C) 진폭을 구한다.

    R, T가 모두 이 결과에서 파생되므로 계산의 단일 진입점으로 둔다.

    Args:
        d: (B, L) real — 층 두께 (lam과 같은 단위).
        n_layers: (L, W) complex — 층별 굴절률.
        n0: 입사 매질 굴절률 (스칼라 또는 (W,)).
        ns: 기판 굴절률 (스칼라 또는 (W,), 복소 허용).
        lam: (W,) real — 파장.

    Returns:
        StackAmplitudes — b, c는 (B, W) complex.

    Raises:
        ValueError: shape 규약이 어긋난 경우.
        TypeError: n_layers가 complex dtype이 아닌 경우.
    """
    t = _stack_terms(d, n_layers, n0, ns, lam)

    # 입사측 층부터 순서대로 누적: M = M_1 @ M_2 @ ... @ M_L.
    a11, a12, a21, a22 = t.m11[:, 0], t.m12[:, 0], t.m21[:, 0], t.m22[:, 0]
    for j in range(1, t.m11.shape[1]):
        b11, b12, b21, b22 = t.m11[:, j], t.m12[:, j], t.m21[:, j], t.m22[:, j]
        a11, a12, a21, a22 = (
            a11 * b11 + a12 * b21,
            a11 * b12 + a12 * b22,
            a21 * b11 + a22 * b21,
            a21 * b12 + a22 * b22,
        )

    # (B, C)^T = M @ (1, ns)^T
    return StackAmplitudes(b=a11 + a12 * t.ns, c=a21 + a22 * t.ns, n0=t.n0, ns=t.ns)


def _abs2(z: torch.Tensor) -> torch.Tensor:
    """|z|^2. z.abs()**2 대신 써서 0 근방에서 sqrt의 미분 특이점을 피한다."""
    return z.real**2 + z.imag**2


def tmm_rt(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """반사율 R과 투과율 T를 한 번의 행렬 누적으로 함께 계산한다.

    T는 기판으로 들어가는 전력이며 n0가 실수(무흡수 입사 매질)임을 가정한다.
    층이 무흡수이면 R + T = 1이 성립하고, 층에 흡수가 있으면 R + T < 1 이며
    차이가 층 내 흡수량이 된다.

    Args:
        d: (B, L) real — 층 두께 (lam과 같은 단위).
        n_layers: (L, W) complex — 층별 굴절률.
        n0: 입사 매질 굴절률 (스칼라 또는 (W,)).
        ns: 기판 굴절률 (스칼라 또는 (W,), 복소 허용).
        lam: (W,) real — 파장.

    Returns:
        (R, T): 각각 (B, W) real.
    """
    amp = stack_amplitudes(d, n_layers, n0, ns, lam)
    rdtype = _real_dtype_of(n_layers.dtype)

    num = amp.n0 * amp.b - amp.c
    denom_sq = _abs2(amp.n0 * amp.b + amp.c)

    r_out = (_abs2(num) / denom_sq).to(rdtype)
    t_out = (4.0 * amp.n0.real * amp.ns.real / denom_sq).to(rdtype)
    return r_out, t_out


def tmm_reflectance(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """다층 스택의 반사율 R(lambda)를 계산한다 — 학습 루프가 쓰는 주 진입점.

    Args:
        d: (B, L) real — 층 두께 (lam과 같은 단위).
        n_layers: (L, W) complex — 층별 굴절률.
        n0: 입사 매질 굴절률 (스칼라 또는 (W,)).
        ns: 기판 굴절률 (스칼라 또는 (W,), 복소 허용).
        lam: (W,) real — 파장.

    Returns:
        R: (B, W) real — dtype은 n_layers에 대응하는 실수 dtype.
    """
    amp = stack_amplitudes(d, n_layers, n0, ns, lam)
    r = (amp.n0 * amp.b - amp.c) / (amp.n0 * amp.b + amp.c)
    return _abs2(r).to(_real_dtype_of(n_layers.dtype))


def tmm_reflectance_jacobian(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R과 두께 야코비안 dR/dd를 한 번의 누적으로 함께 계산한다 (해석적).

    M_j = cos(delta_j)*I + i*sin(delta_j)*K_j 이고 K_j = [[0, 1/n_j], [n_j, 0]] 는
    K_j^2 = I 를 만족하므로 M_j = exp(i*delta_j*K_j) 이고

        dM_j/dd_j = i * k_j * K_j * M_j,      k_j = 2*pi*n_j/lambda

    이다. 스택 곱을 M_1..M_{j-1} (접두) 과 M_j..M_L @ (1, ns)^T (접미 벡터) 로 나눠 각각
    한 번씩 모으면, **cos/sin을 다시 계산하지 않고** L개 도함수를 전부 얻는다. 중앙차분이
    forward 2L회를 쓰는 자리를 약 2회분으로 줄이므로 LM 역해의 반복당 비용이 내려간다.

    Args:
        d: (B, L) real — 층 두께 (lam과 같은 단위).
        n_layers: (L, W) complex — 층별 굴절률.
        n0: 입사 매질 굴절률 (스칼라 또는 (W,)).
        ns: 기판 굴절률 (스칼라 또는 (W,), 복소 허용).
        lam: (W,) real — 파장.

    Returns:
        (R, J): R (B, W) real — `tmm_reflectance`와 같은 값.
        J (B, W, L) real — J[..., j] = dR/dd_j (단위는 1/[d의 단위]).
    """
    t = _stack_terms(d, n_layers, n0, ns, lam)
    n_l = t.m11.shape[1]

    # 접미 벡터 u_j = M_j @ ... @ M_L @ (1, ns)^T 를 뒤에서부터 모은다.
    # suffix[k] = u_{L+1-k} 이므로 suffix[0] = (1, ns), suffix[n_l] = (B, C).
    one = torch.ones_like(t.m11[:, 0])
    suffix = [(one, t.ns * one)]
    for j in range(n_l - 1, -1, -1):
        p, q = suffix[-1]
        suffix.append((t.m11[:, j] * p + t.m12[:, j] * q, t.m21[:, j] * p + t.m22[:, j] * q))

    # 접두 곱 P = M_1 @ ... @ M_{j-1} 을 앞에서부터 누적하며 층별 도함수를 받는다.
    # d(B, C)/dd_j = P @ (i*k_j*K_j) @ u_j 이고, K_j @ (p, q) = (q/n_j, n_j*p) 이므로
    # k_j/n_j = wave, k_j*n_j = n_j^2 * wave 로 나눗셈이 사라진다.
    zero = torch.zeros_like(one)
    p11, p12, p21, p22 = one, zero, zero, one
    i_wave = 1j * t.wave
    d_b: list[torch.Tensor] = []
    d_c: list[torch.Tensor] = []
    for j in range(n_l):
        p, q = suffix[n_l - j]
        n_j = t.n_layers[:, j]
        w_p = i_wave * q
        w_q = i_wave * (n_j * n_j) * p
        d_b.append(p11 * w_p + p12 * w_q)
        d_c.append(p21 * w_p + p22 * w_q)
        m11, m12, m21, m22 = t.m11[:, j], t.m12[:, j], t.m21[:, j], t.m22[:, j]
        p11, p12, p21, p22 = (
            p11 * m11 + p12 * m21,
            p11 * m12 + p12 * m22,
            p21 * m11 + p22 * m21,
            p21 * m12 + p22 * m22,
        )

    # R은 `tmm_reflectance`와 **비트 동일**해야 한다 — LM 역해가 두 경로의 R을 같은 잔차
    # 장부에 섞어 쓴다. 그래서 (B, C)를 접미 벡터(suffix[n_l])가 아니라 **접두 곱**에서
    # 만든다: 위 루프가 끝난 p11~p22 = M_1 @ ... @ M_L 은 `stack_amplitudes`와 누적 순서가
    # 같아 반올림까지 일치한다 (접미 재귀는 곱 순서가 반대라 어긋난다).
    b, c = p11 + p12 * t.ns, p21 + p22 * t.ns
    num = t.n0 * b - c
    den = t.n0 * b + c
    den_sq = _abs2(den)
    r_out = _abs2(num / den).to(t.rdtype)

    # R = N/D 이므로 dR = (dN - R*dD)/D. 복소 z에 대해 d|z|^2 = 2*Re(conj(z)*dz).
    cols = []
    for j in range(n_l):
        d_num = t.n0 * d_b[j] - d_c[j]
        d_den = t.n0 * d_b[j] + d_c[j]
        d_n = 2.0 * (num.real * d_num.real + num.imag * d_num.imag)
        d_d = 2.0 * (den.real * d_den.real + den.imag * d_den.imag)
        cols.append((d_n - r_out * d_d) / den_sq)
    return r_out, torch.stack(cols, dim=-1).to(t.rdtype)


def tmm_transmittance(
    d: torch.Tensor,
    n_layers: torch.Tensor,
    n0: float | complex | torch.Tensor,
    ns: float | complex | torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """기판으로 들어가는 투과율 T = 4*n0*Re(ns) / |n0*B + C|^2 를 계산한다.

    Returns:
        T: (B, W) real.
    """
    return tmm_rt(d, n_layers, n0, ns, lam)[1]
