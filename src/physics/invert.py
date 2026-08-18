"""동결 forward 모델의 두께 역해 — 배치 Levenberg–Marquardt.

두 곳이 같은 최적화를 쓰되 **출발점이 다르고, 그 차이가 측정의 전부**다.

- Stage A 게이트 (d): `d_true`에서 출발 → 재는 것은 전역 탐색 난이도가 아니라 디코더의
  **내재 편향**이다 (forward 모델 오차가 두께 추정을 얼마나 밀어내는가). 라벨을 쓰므로
  경쟁 성능 수치로 쓸 수 없다.
- 역산 refinement: `d_hat`(신경망 예측)에서 출발 → **추론 후 보정**. 라벨을 쓰지 않으므로
  test·실계측에도 그대로 적용된다.

야코비안은 두 방식을 고를 수 있고 **비용이 크게 다르다**. 중앙차분(`"fd"`)은 반복마다
forward 2L회를 쓴다. 해석적(`"analytic"`)은 디코더의 `forward_jacobian`을 불러 cos/sin을
공유하므로 약 forward 2회에 L개 열을 전부 얻는다 — 반복당 9회가 3회로 줄어든다.
autograd는 (M, W, L) 야코비안에 출력 채널마다 backward가 필요해 여기서는 셋 중 가장 비싸다.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "DEFAULT_BOX_NM",
    "MAD_TO_SIGMA",
    "PHYSICAL_RANGE_NM",
    "flag_unreliable",
    "inversion_stats",
    "lm_invert",
    "refine_with_fallback",
    "residual_l1_rows",
]

# MAD를 정규분포 표준편차로 환산하는 계수 (1/Phi^-1(0.75)).
MAD_TO_SIGMA = 1.4826

# 두께 격자의 물리 범위 [nm]. 상자 제약(아래)은 의도적으로 이보다 넓다.
PHYSICAL_RANGE_NM = (10.0, 300.0)
# 해의 상자 제약 [nm] — 라벨의 사전지식을 역해에 넣지 않는 보수적 선택이다.
# 좁히면 수치가 좋아지는 방향이므로, 대신 범위 밖 해의 비율을 함께 보고한다.
DEFAULT_BOX_NM = (1.0, 400.0)


def _device_of(model: nn.Module) -> torch.device:
    """model이 올라가 있는 장치. 파라미터가 없는 모듈(FrozenDecoder)도 버퍼로 잡힌다."""
    for tensor in (*model.parameters(), *model.buffers()):
        return tensor.device
    return torch.device("cpu")


@torch.no_grad()
def lm_invert(
    model: nn.Module,
    x: np.ndarray,
    d_init: np.ndarray,
    *,
    iters: int = 30,
    step_nm: float = 1e-3,
    box: tuple[float, float] = DEFAULT_BOX_NM,
    damping: Literal["batch", "row"] = "batch",
    chunk: int | None = None,
    jacobian: Literal["fd", "analytic"] = "analytic",
    tol_nm: float | None = None,
    patience: int = 4,
) -> np.ndarray:
    """관측 R에 맞도록 두께를 역해한다 — d_init에서 출발하는 배치 LM.

    **장치·dtype은 model을 따라간다** — `model.to("cuda")`로 올리면 여기도 GPU에서 돈다.
    작업 dtype도 model의 출력 dtype이다 (complex64 디코더면 float32). 별도 인자를 두지 않는
    이유는 model과 어긋난 값을 받을 자리를 아예 없애기 위해서다.

    Args:
        model: 동결 forward 모델. d (M, L) → R (M, W).
        x: (M, W) 관측 반사율.
        d_init: (M, L) 출발 두께 [nm]. **이 인자가 무엇이냐가 실험의 정의다** (모듈 설명).
        iters: LM 반복 수. step_nm: 중앙차분 보폭 [nm]. box: 해의 상자 제약 [nm].
            float32에서는 step_nm이 너무 작으면 야코비안이 자릿수를 잃는다 (두께 100 nm에
            보폭 1e-3이면 상대 섭동 1e-5, float32 유효자리는 약 1e-7).
        damping: 감쇠항 스케일을 배치 평균으로 잡을지(`"batch"`) 행별로 잡을지(`"row"`).
            `"batch"`는 한 행의 갱신이 같은 배치의 다른 행에 의존하므로 **청크 구성에 결과가
            달라진다** — Stage A 게이트 (d)가 쓰는 기존 동작이라 기본값으로 남긴다.
            `"row"`는 행이 독립이라 청크 불변이고, 출발점이 행마다 제각각인 refinement에 맞다.
        chunk: 행을 이 크기로 끊어 처리한다 (중간 텐서가 (M, L, W)라 8만 행은 10 GB가 넘는다).
        jacobian: `"analytic"`은 model의 `forward_jacobian`(반복당 forward 약 2회분),
            `"fd"`는 중앙차분(2L회). 정확도 근거는 tests/test_tmm.py §8 — float32에서는
            해석적이 오히려 더 정확하다(중앙차분은 보폭이 유효자리를 먹는다). 두 값은
            반올림 수준에서 다르므로 커밋된 리포트를 재생성할 때는 그 리포트가 쓴 값을
            그대로 써야 한다 (모듈 설명).
        tol_nm: 조기 종료 문턱 [nm]. None이면 끄고 전 행이 `iters`회를 다 돈다. 값을 주면
            갱신 폭이 이 값 미만인 반복이 `patience`회 연속인 행을 **작업 집합에서 뺀다** —
            남은 행의 결과는 바뀌지 않는다(행 독립). `damping="row"`에서만 쓸 수 있다.
        patience: 위 연속 횟수. 기각된 스텝은 갱신 폭 0이라 함께 센다 — 감쇠가 올라가며
            스텝이 되살아나는 LM 특성 때문에 1~2는 위험하다.

    Returns:
        d_hat: (M, L) float64 [nm] — 작업 dtype과 무관하게 항상 CPU float64로 돌려준다.

    Raises:
        ValueError: 행 수가 어긋나거나, 청크·조기 종료를 배치 감쇠와 함께 쓴 경우.
        TypeError: `jacobian="analytic"`인데 model에 `forward_jacobian`이 없는 경우.
    """
    if len(x) != len(d_init):
        raise ValueError(f"x {len(x)}행과 d_init {len(d_init)}행이 다르다")
    if jacobian not in ("fd", "analytic"):
        raise ValueError(f"jacobian은 'fd' | 'analytic' 이어야 한다 (받은 값: {jacobian!r})")
    if jacobian == "analytic" and not hasattr(model, "forward_jacobian"):
        raise TypeError(
            f"jacobian='analytic'은 model에 forward_jacobian이 필요하다"
            f" ({type(model).__name__}에 없다) — FrozenDecoder를 쓰거나 'fd'로 둘 것"
        )
    if chunk is not None and damping == "batch" and chunk < len(x):
        raise ValueError(
            "damping='batch'는 청크 구성에 결과가 의존한다 — 청크를 쓰려면 damping='row'"
        )
    if tol_nm is not None and damping == "batch":
        raise ValueError(
            "damping='batch'는 감쇠 스케일이 배치 구성에 의존한다 — 조기 종료로 행이 빠지면"
            " 남은 행의 답이 조용히 달라진다. tol_nm을 쓰려면 damping='row'"
        )
    if patience < 1:
        raise ValueError(f"patience는 1 이상이어야 한다 (받은 값: {patience})")
    if chunk is not None and chunk < len(x):
        parts = [
            lm_invert(
                model,
                x[s : s + chunk],
                d_init[s : s + chunk],
                iters=iters,
                step_nm=step_nm,
                box=box,
                damping=damping,
                chunk=None,
                jacobian=jacobian,
                tol_nm=tol_nm,
                patience=patience,
            )
            for s in range(0, len(x), chunk)
        ]
        return np.concatenate(parts)

    device = _device_of(model)
    d = torch.from_numpy(np.asarray(d_init, dtype=np.float64)).to(device)
    work = model(d[:1]).dtype  # 작업 dtype = model의 출력 dtype (complex64 디코더면 float32)
    d = d.to(work)
    obs = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(device=device, dtype=work)
    n_layers = d.shape[1]
    eye_l = torch.eye(n_layers, dtype=work, device=device)
    lam_damp = torch.full((len(d), 1, 1), 1e-3, dtype=work, device=device)

    # 조기 종료로 빠진 행을 받아 두는 버퍼와, 작업 집합 → 버퍼 위치 대응.
    out = d.clone()
    act = torch.arange(len(d), device=device)
    stall = torch.zeros(len(d), dtype=torch.long, device=device)

    resid = model(d) - obs  # (M, W)
    cost = (resid**2).sum(dim=1)
    for _ in range(iters):
        if len(d) == 0:
            break
        if jacobian == "analytic":
            jac = model.forward_jacobian(d)[1]  # (M, W, L)
        else:
            jac = torch.stack(
                [
                    (model(d + step_nm * eye_l[j]) - model(d - step_nm * eye_l[j]))
                    / (2.0 * step_nm)
                    for j in range(n_layers)
                ],
                dim=-1,
            )  # (M, W, L)
        jtj = jac.transpose(1, 2) @ jac  # (M, L, L)
        jtr = (jac.transpose(1, 2) @ resid.unsqueeze(-1)).squeeze(-1)  # (M, L)
        diag = jtj.diagonal(dim1=1, dim2=2)  # (M, L)
        scale = diag.mean() if damping == "batch" else diag.mean(dim=1).reshape(-1, 1, 1)
        damped = jtj + lam_damp * eye_l.expand_as(jtj) * scale
        delta = torch.linalg.solve(damped, -jtr.unsqueeze(-1)).squeeze(-1)
        cand = (d + delta).clamp(*box)
        resid_c = model(cand) - obs
        cost_c = (resid_c**2).sum(dim=1)
        # 비용이 내려간 행만 갱신하고, 그 행의 감쇠를 낮춘다 (실패한 행은 높인다).
        better = cost_c < cost
        # 정체 판정은 **제안된** 이동 폭으로 한다 (수용 여부와 무관). 기각을 이동 0으로
        # 세면 감쇠가 오르며 스텝이 되살아나는 행을 잘라내 어려운 행에서 정확도를 잃는다.
        moved = (cand - d).abs().amax(dim=1)
        d = torch.where(better.unsqueeze(-1), cand, d)
        resid = torch.where(better.unsqueeze(-1), resid_c, resid)
        cost = torch.where(better, cost_c, cost)
        lam_damp = torch.where(
            better.reshape(-1, 1, 1), (lam_damp * 0.3).clamp(min=1e-9), lam_damp * 3.0
        )
        if tol_nm is None:
            continue
        stall = torch.where(moved < tol_nm, stall + 1, torch.zeros_like(stall))
        done = stall >= patience
        if bool(done.any()):
            out[act[done]] = d[done]
            keep = ~done
            act, d, obs, resid, cost, lam_damp, stall = (
                act[keep],
                d[keep],
                obs[keep],
                resid[keep],
                cost[keep],
                lam_damp[keep],
                stall[keep],
            )
    out[act] = d
    return out.double().cpu().numpy()


def inversion_stats(
    d_hat: np.ndarray, d_true: np.ndarray, *, box: tuple[float, float] = DEFAULT_BOX_NM
) -> dict[str, Any]:
    """역해 결과의 nm 단위 오차 통계와 [0, 1] 비율들.

    Args:
        d_hat: (M, L) 역해 두께 [nm]. d_true: (M, L) 참 두께 [nm]. box: `lm_invert`에 쓴 상자.
    """
    err = np.asarray(d_hat, dtype=np.float64) - np.asarray(d_true, dtype=np.float64)
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "mae_per_layer": [float(v) for v in abs_err.mean(axis=0)],
        "bias_per_layer": [float(v) for v in err.mean(axis=0)],
        "rmse_nm": float(np.sqrt((err**2).mean())),
        # 평균은 중앙값과 꼬리가 섞인 혼합값이라 규제자 신뢰도를 두 성분으로 나눠 말할 수 없다.
        "abs_err_median": float(np.median(abs_err)),
        "abs_err_p99": float(np.percentile(abs_err, 99)),
        "abs_err_max": float(abs_err.max()),
        "out_of_physical": float(
            ((d_hat < PHYSICAL_RANGE_NM[0]) | (d_hat > PHYSICAL_RANGE_NM[1])).mean()
        ),
        "at_box_boundary": float((np.isclose(d_hat, box[0]) | np.isclose(d_hat, box[1])).mean()),
    }


def flag_unreliable(residual: np.ndarray, *, k_sigma: float = 5.0) -> tuple[np.ndarray, float]:
    """재구성 잔차만으로 "이 답은 관측을 설명하지 못한다"는 행을 지목한다 — **라벨 미사용**.

    정답 골짜기에 수렴한 행의 잔차는 측정 노이즈 바닥에 붙는다(E|ε| = 0.0075). 잘못된 분지의
    바닥은 그보다 높으므로 잔차 자체가 오답의 증거다 — Stage A 게이트 (b)와 같은 논리이고,
    라벨이 아니라 **관측과 우리 답**만 쓰므로 test·실계측에 그대로 적용된다.

    문턱은 잔차 분포에서 만든다: 행의 대다수가 정상이므로 중앙값과 MAD가 곧 정상 행의 분포이고
    소수의 이상치는 robust 통계를 흔들지 못한다. **holdout 성능을 보며 k_sigma를 고르면 평가셋
    선택이 되므로 사전등록한다** (`reports/inversion_refine.md`의 이동 상한 τ가 그 함정에
    빠졌다 — τ는 선택자가 이동 거리였고 문턱도 holdout에서 골랐다).

    Args:
        residual: (N,) 행별 재구성 L1 (`residual_l1_rows` 출력).
        k_sigma: 문턱 = 중앙값 + k_sigma × robust σ. 사전등록 값 5.0.

    Returns:
        (mask, threshold) — mask (N,) bool은 지목된 행, threshold는 쓰인 문턱.
        산포가 0이면(전 행 잔차 동일) 이상치를 정의할 수 없으므로 아무 행도 지목하지 않는다.

    Raises:
        ValueError: k_sigma가 음수인 경우.
    """
    if k_sigma < 0:
        raise ValueError(f"k_sigma는 0 이상이어야 한다 (받은 값: {k_sigma})")
    res = np.asarray(residual, dtype=np.float64)
    median = float(np.median(res))
    sigma = float(np.median(np.abs(res - median))) * MAD_TO_SIGMA
    if sigma <= 0.0:
        return np.zeros(len(res), dtype=bool), float("inf")
    threshold = median + k_sigma * sigma
    return res > threshold, threshold


def residual_l1_rows(
    model: nn.Module, d: np.ndarray, x: np.ndarray, *, chunk: int = 4096
) -> np.ndarray:
    """행별 재구성 L1 |R_model(d) − R_obs| — 라벨을 쓰지 않는 신뢰도 지표 (README §3.4).

    **장치는 model을 따라간다** — 관측도 같은 장치로 올린다 (GPU 디코더에 CPU 관측을 빼면
    장치 불일치로 죽는다).

    Args:
        model: 동결 forward 모델. d: (M, L) [nm]. x: (M, W) 관측. chunk: 배치 크기.

    Returns:
        (M,) float64.
    """
    device = _device_of(model)
    d_t = torch.from_numpy(np.asarray(d, dtype=np.float64)).to(device)
    obs_np = np.asarray(x, dtype=np.float64)
    out = np.empty(len(d_t), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(d_t), chunk):
            pred: Tensor = model(d_t[s : s + chunk])
            obs = torch.from_numpy(obs_np[s : s + chunk]).to(device=device, dtype=pred.dtype)
            out[s : s + chunk] = (pred - obs).abs().mean(dim=1).cpu().numpy()
    return out


def refine_with_fallback(
    model: nn.Module,
    x: np.ndarray,
    d_init: np.ndarray,
    *,
    iters: int = 30,
    tol_nm: float | None = 1e-4,
    k_sigma: float = 5.0,
    chunk: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    """배포 경로 전체: LM 역해 → 잔차로 실패 지목 → 지목된 행은 `d_init`으로 되돌림.

    **라벨이 한 군데도 들어가지 않는다** — 쓰는 것은 관측 `x`와 우리 답뿐이므로 test·실계측에
    그대로 적용된다. 되돌리는 이유는 지목된 행에서 신경망 예측이 LM 결과보다 정확하기
    때문이다 — LM이 잘못된 분지 바닥까지 성실하게 내려간다 (측정 정본은
    reports/cnn_recipe_judge.md «규칙의 근거» 표, scripts/judge_recipe.py 산출).

    문턱은 잔차 분포에서 만들므로 **transductive다**: 들어온 행 집합이 문턱을 정한다. 배치
    계측에서는 제약이 아니지만, 한 행씩 처리하는 배포에는 미리 계산한 문턱이 필요하다.

    Args:
        model: 동결 forward 디코더. x: (M, W) 관측. d_init: (M, L) 신경망 예측 [nm].
        iters / tol_nm / chunk: `lm_invert`에 그대로 넘긴다 (해석적 야코비안 고정).
        k_sigma: 되돌림 문턱 배수. **사전등록 값이며 성능을 보고 고르지 않는다**
            (`flag_unreliable`).

    Returns:
        (d_final, info) — info에 `flagged`(지목 행 수)·`threshold`·`residual`(M,)·
        `d_lm`(되돌림 전 해)이 들어간다. 진단·리포트가 그 둘을 함께 봐야 한다.
    """
    d_lm = lm_invert(
        model,
        x,
        d_init,
        iters=iters,
        damping="row",
        chunk=chunk,
        jacobian="analytic",
        tol_nm=tol_nm,
    )
    residual = residual_l1_rows(model, d_lm, x, chunk=chunk)
    flagged, threshold = flag_unreliable(residual, k_sigma=k_sigma)
    d_final = np.where(flagged[:, None], np.asarray(d_init, dtype=np.float64), d_lm)
    return d_final, {
        "flagged": int(flagged.sum()),
        "flagged_frac": float(flagged.mean()),
        "threshold": threshold,
        "k_sigma": k_sigma,
        "residual": residual,
        "d_lm": d_lm,
        "mask": flagged,
    }
