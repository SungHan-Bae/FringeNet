"""Adachi MDF의 Si 초기 계수 프리핏 — src/physics/adachi_si_init.json 산출.

Adachi 함수형(src/physics/dispersion.py의 adachi_si_eps)을 같은 파일의
Aspnes & Studna 1983 근사 독취 테이블(_SI_LAM_NM/_SI_N/_SI_K)에 최소제곱으로
맞춰 캘리브레이션 초기 계수를 만든다. Adachi 원논문의 계수 테이블을 쓰지 않는
이유: 원문(유료) 접근 불가 + 프로젝트 규약상 수치는 스크립트 산출물이어야 한다.
어차피 초기값 용도라 Stage A 캘리브레이션이 학습으로 갱신한다.

결정론: 시드 고정, 데이터가 테이블 상수뿐이라 재실행 시 동일 결과.

사용:
    python scripts/fit_adachi_init.py            # JSON 갱신 + 비교표 출력
    python scripts/fit_adachi_init.py --dry-run  # JSON 쓰지 않고 비교표만
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softplus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.dispersion import (  # noqa: E402
    ADACHI_SI_PARAM_NAMES,
    adachi_si_nk,
    si_nk,
    softplus_inverse,
)

JSON_PATH = Path(__file__).resolve().parents[1] / "src" / "physics" / "adachi_si_init.json"

# 프리핏 시작값 — 교과서 수준의 Si 임계점 에너지(E1 ≈ 3.38, E2 ≈ 4.27, 간접갭 1.12 eV)와
# 자릿수 규모의 진폭 추정. 최종 계수는 아래 최소제곱이 결정하므로 정밀도는 중요하지 않다.
START = {
    "eps_inf": 1.0,
    "D": 1.0,
    "Eg": 1.12,
    "B1": 5.0,
    "B1x": 1.5,
    "Gamma1": 0.10,
    "E1": 3.38,
    "C0": 1.0,
    "gamma0": 0.30,
    "E0p": 3.35,
    "C": 3.0,
    "gamma": 0.10,
    "E2": 4.27,
}

# k가 5 → 5e-4로 4자릿수를 가로지르므로 k는 로그 공간에서 맞춘다 (간접갭 꼬리가
# 진폭 큰 임계점 영역에 눌리지 않게). floor는 log 발산 방지.
# k 점별 가중 k/(k+K_SENS): R은 기판 k가 큰 곳(자외선 쪽)에서만 민감하고
# k ≲ 0.02의 장파장 꼬리에는 사실상 둔감하다(ΔR ~ 1e-3 미만). 균등 가중이면 꼬리
# 점 수가 압도해 정작 R에 중요한 390–500 nm(k 0.07–0.5)를 포기하는 해로 간다.
K_FLOOR = 1e-4
K_SENS = 0.05
K_WEIGHT = 0.5
STEPS = 40_000
LR = 5e-3
SEED = 42

# 피팅 그리드: 19점 원표가 아니라 si_nk() 선형 보간의 조밀 그리드 — 점 사이에서
# broadening이 좁아지며 생기는 스파이크를 봉쇄한다 (1차 실행에서 Γ1이 0.012 eV까지
# 좁아지는 것을 확인). 캘리브레이션 대역(284–793 nm) 밖은 가중 0.3.
LAM_DENSE = np.arange(272.0, 1001.0, 3.0)
BAND = (284.0, 793.0)
OUT_OF_BAND_WEIGHT = 0.3

# 간접갭 Eg는 프리핏에서 고정 — Si의 가장 확실한 물성 상수(1.12 eV)이고, 자유로 두면
# 장파장 꼬리를 포기하고 1.8 eV로 표류한다(1차 실행 확인). 캘리브레이션에서는 학습.
EG_FROZEN_IDX = ADACHI_SI_PARAM_NAMES.index("Eg")


def fit() -> tuple[np.ndarray, dict[str, float]]:
    """프리핏 실행. 반환 (계수 value 공간 (13,), 진단 dict)."""
    torch.manual_seed(SEED)
    lam = torch.from_numpy(LAM_DENSE).to(torch.float64)
    n_np, k_np = si_nk(LAM_DENSE)
    n_tab = torch.from_numpy(n_np).to(torch.float64)
    k_tab = torch.from_numpy(k_np).to(torch.float64)
    w = torch.where(
        torch.from_numpy((BAND[0] <= LAM_DENSE) & (BAND[1] >= LAM_DENSE)),
        1.0,
        OUT_OF_BAND_WEIGHT,
    ).to(torch.float64)
    w = w / w.sum()
    w_k = w * k_tab / (k_tab + K_SENS)
    w_k = w_k / w_k.sum()

    start = torch.tensor([START[name] for name in ADACHI_SI_PARAM_NAMES], dtype=torch.float64)
    raw = torch.nn.Parameter(softplus_inverse(start).clone())
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    best_loss, best_raw = float("inf"), raw.detach().clone()
    for _ in range(STEPS):
        n_pred, k_pred = adachi_si_nk(lam, softplus(raw))
        loss_n = (w * (n_pred - n_tab) ** 2).sum()
        loss_k = (w_k * (torch.log(k_pred + K_FLOOR) - torch.log(k_tab + K_FLOOR)) ** 2).sum()
        loss = loss_n + K_WEIGHT * loss_k
        opt.zero_grad()
        loss.backward()
        raw.grad[EG_FROZEN_IDX] = 0.0  # Eg 고정
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss, best_raw = loss.item(), raw.detach().clone()

    params = softplus(best_raw)
    with torch.no_grad():
        n_pred, k_pred = adachi_si_nk(lam, params)
    in_band = (lam >= BAND[0]) & (lam <= BAND[1])
    n_rel = ((n_pred - n_tab).abs() / n_tab)[in_band]
    k_logdev = (torch.log10(k_pred + K_FLOOR) - torch.log10(k_tab + K_FLOOR)).abs()[in_band]
    diag = {
        "loss": best_loss,
        "band_nm": list(BAND),
        "n_reldev_median": float(n_rel.median()),
        "n_reldev_max": float(n_rel.max()),
        "k_log10dev_median": float(k_logdev.median()),
        "k_log10dev_max": float(k_logdev.max()),
    }

    print(f"{'λ[nm]':>7} {'n_tab':>7} {'n_fit':>7} {'k_tab':>9} {'k_fit':>9}")
    step = 5  # 조밀 그리드는 15 nm 간격으로만 출력
    for i in range(0, len(lam), step):
        print(f"{lam[i]:7.0f} {n_tab[i]:7.3f} {n_pred[i]:7.3f} {k_tab[i]:9.4f} {k_pred[i]:9.4f}")
    print("\n계수 (value 공간):")
    for name, v in zip(ADACHI_SI_PARAM_NAMES, params.tolist(), strict=True):
        print(f"  {name:8s} = {v:.6g}")
    print(f"\n진단: {json.dumps(diag, indent=2)}")
    return params.numpy(), diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="JSON을 쓰지 않고 비교표만 출력")
    args = parser.parse_args()
    params, diag = fit()
    if args.dry_run:
        return
    JSON_PATH.write_text(
        json.dumps(
            {
                "comment": "scripts/fit_adachi_init.py 산출 — Adachi MDF를 Aspnes & Studna "
                "테이블에 프리핏한 Si 초기 계수. 손으로 수정하지 말 것.",
                "names": list(ADACHI_SI_PARAM_NAMES),
                "values": [float(v) for v in params],
                "fit_diagnostics": diag,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"\n저장: {JSON_PATH}")


if __name__ == "__main__":
    main()
