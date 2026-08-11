"""Stage A 캘리브레이션 — train (d_true, R_obs) 서브셋으로 TMM forward 미지수를 피팅한다.

무엇을 학습하나 (CLAUDE.md Level 2 Stage A 스펙):
  - λ 그리드: lam = lam_min + cumsum(softplus(·)) — 채널 순서대로 단조. 파장축이
    비식별화라 오름/내림 방향도 모른다 → ``descending`` 플래그로 두 방향을 모두
    후보 탐색에 넣는다 (EDA의 "오른쪽 끝 신호 3배"는 짧은 λ가 오른쪽일 가능성을
    시사하지만 단정하지 않는다).
  - SiN(layer 1·3): Cauchy n(λ) = A + B/λ² + C/λ⁴ (λ[μm], k=0 가정) — 학습.
  - SiO₂(layer 2·4): 같은 Cauchy — **문헌값(Malitson 1965 fit)에 freeze (게이지 고정)**.
    delta = 2πnd/λ 가 (n, λ) 공통 스케일에 불변이라 SiO₂를 고정해야 λ가 식별된다.
  - Si 기판: n(λ), k(λ) 곡선 — 채널축 knot 조각별 선형 보간, k ≥ 0 (softplus).

산출물: runs/stage_a/<run_name>/{model.pt, train.log, metrics.json}
(+ 진행 중 resume.pt — 완료 시 삭제). 판정 게이트 (a) RMSE는 여기서 즉시 보고하고,
(c) 잔차 백색성 진단·플롯은 scripts/diagnose_calibration.py 가 model.pt를 읽어 수행한다.

세션 유실 대비 계약 (CLAUDE.md — train_gpu.py와 동일): best 갱신 즉시 model.pt 저장,
eval 블록마다 resume.pt(+RNG) 저장·미러, 재실행 시 완료 run 스킵 + 진행 run 재개
(무중단 실행과 동일 결과 — 테스트로 검증).

사용:
    python -m src.calibrate --config configs/stage_a/sio2-freeze.yaml
    python -m src.calibrate --config ... --fit-rows 2000 --steps 40 \
        --lam-init 400,800 --run-name smoke   # 스모크 (후보 탐색 생략)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn.functional import mse_loss, softplus

from src.data.dataset import REPO_ROOT, prepare_train_arrays
from src.physics.dispersion import (
    cauchy_n,
    fit_cauchy,
    linear_interp_matrix,
    si3n4_n,
    si_nk,
    sio2_n,
    softplus_inverse,
)
from src.physics.tmm import tmm_reflectance
from src.train import build_lr_scheduler, log_line
from src.train_gpu import RESUME_NAME, _atomic_save, _mirror_copy, resolve_device

RUNS_DIR = REPO_ROOT / "runs"

# 판정 게이트 (a): 재구성 RMSE < 1.2σ ≈ 0.0105 (σ ≈ 0.0087 노이즈 바닥 — CLAUDE.md).
GATE_A_RMSE = 0.0105

# "value = init + scale × raw" 재파라미터화의 scale — λ(수백 nm)와 Cauchy C(~1e-4)처럼
# 스케일이 극단적으로 다른 양을 Adam이 단일 lr·균일 보폭(raw 공간)으로 움직이게 한다.
_LAM_MIN_SCALE = 100.0  # raw 1 ≈ λ_min 100 nm 이동
_DLAM_SCALE = 0.3  # 채널 간격(softplus 전 raw)의 보폭
_SI_N_SCALE = 0.5  # Si n knot 보폭
_SI_K_SCALE = 1.0  # Si k knot(softplus 전 raw) 보폭 — k가 작을 땐 곱셈적 변화


class CalibratedStack(nn.Module):
    """캘리브레이션 대상 forward 모델의 미지수 전부를 담는 모듈.

    구조: 공기(1.0) / SiN / SiO₂ / SiN / SiO₂ / Si 기판, 수직입사 (CLAUDE.md 도메인 계약).
    dtype은 캘리브레이션 계약대로 float64/complex128 고정.

    Args:
        n_channels: 스펙트럼 채널 수 W.
        n_si_knots: Si n·k 곡선의 knot 수 (채널축 균등 배치, 조각별 선형 보간).
        lam_init: 초기 가정 λ 범위 (min, max) [nm] — 실제 그리드는 학습된다.
        descending: True면 채널 0이 λ_max (λ가 채널 순서로 감소).
    """

    def __init__(
        self,
        n_channels: int = 226,
        n_si_knots: int = 16,
        lam_init: tuple[float, float] = (400.0, 800.0),
        descending: bool = False,
    ) -> None:
        super().__init__()
        lam_lo, lam_hi = float(lam_init[0]), float(lam_init[1])
        if not 0.0 < lam_lo < lam_hi:
            raise ValueError(f"lam_init은 0 < min < max 여야 한다 (받은 값: {lam_init})")
        if n_channels < 2:
            raise ValueError(f"n_channels는 2 이상이어야 한다 (받은 값: {n_channels})")
        self.n_channels = int(n_channels)
        self.n_si_knots = int(n_si_knots)
        self.lam_init = (lam_lo, lam_hi)
        self.descending = bool(descending)
        f64 = torch.float64

        # 채널 순서의 초기 λ (descending이면 채널 0이 λ_max).
        lam0 = np.linspace(lam_lo, lam_hi, n_channels)
        step0 = (lam_hi - lam_lo) / (n_channels - 1)

        # λ 그리드 — 단조 보장: lam_min + cumsum(softplus(·)).
        self.register_buffer("lam_min_init", softplus_inverse(torch.tensor(lam_lo, dtype=f64)))
        self.raw_lam_min = nn.Parameter(torch.zeros((), dtype=f64))
        self.register_buffer(
            "dlam_init", softplus_inverse(torch.full((n_channels - 1,), step0, dtype=f64))
        )
        self.raw_dlam = nn.Parameter(torch.zeros(n_channels - 1, dtype=f64))

        # SiN Cauchy — 학습 (초기값: Luke 2015 Sellmeier를 초기 그리드 위에서 Cauchy 근사).
        sin_init = torch.from_numpy(fit_cauchy(lam0, si3n4_n(lam0)))
        self.register_buffer("sin_init", sin_init)
        self.register_buffer("sin_scale", torch.clamp(0.5 * sin_init.abs(), min=1e-5))
        self.raw_sin = nn.Parameter(torch.zeros(3, dtype=f64))

        # SiO₂ Cauchy — freeze (게이지 고정: n과 λ는 동시 식별 불가 — CLAUDE.md).
        self.register_buffer("sio2_cauchy", torch.from_numpy(fit_cauchy(lam0, sio2_n(lam0))))

        # Si 기판 n·k — 채널축 knot 보간. knot 0은 채널 0이므로 descending이면 λ_max 쪽.
        self.register_buffer("interp", linear_interp_matrix(n_channels, n_si_knots))
        knot_lam = np.linspace(lam_lo, lam_hi, n_si_knots)
        if descending:
            knot_lam = knot_lam[::-1].copy()
        n_si, k_si = si_nk(knot_lam)
        self.register_buffer("si_n_init", torch.from_numpy(n_si))
        self.raw_si_n = nn.Parameter(torch.zeros(n_si_knots, dtype=f64))
        self.register_buffer(
            "si_k_init", softplus_inverse(torch.from_numpy(np.clip(k_si, 1e-4, None)))
        )
        self.raw_si_k = nn.Parameter(torch.zeros(n_si_knots, dtype=f64))

    @property
    def model_cfg(self) -> dict[str, Any]:
        """체크포인트에서 같은 구조를 복원하기 위한 생성 인자 스냅샷."""
        return {
            "n_channels": self.n_channels,
            "n_si_knots": self.n_si_knots,
            "lam_init": list(self.lam_init),
            "descending": self.descending,
        }

    def lam(self) -> Tensor:
        """학습된 λ 그리드. 반환 (W,) float64 — 채널 순서 (descending이면 감소)."""
        lam_min = softplus(self.lam_min_init + _LAM_MIN_SCALE * self.raw_lam_min)
        steps = softplus(self.dlam_init + _DLAM_SCALE * self.raw_dlam)
        grid = torch.cat([lam_min.reshape(1), lam_min + torch.cumsum(steps, dim=0)])
        return torch.flip(grid, dims=[0]) if self.descending else grid

    def spectra(self) -> tuple[Tensor, Tensor, Tensor]:
        """TMM 입력 물리량을 전부 계산한다.

        Returns:
            (lam (W,) float64, n_layers (4, W) complex128, ns (W,) complex128).
        """
        lam = self.lam()
        n_sin = cauchy_n(lam, self.sin_init + self.sin_scale * self.raw_sin)
        n_sio2 = cauchy_n(lam, self.sio2_cauchy)
        stack_r = torch.stack([n_sin, n_sio2, n_sin, n_sio2])
        n_layers = torch.complex(stack_r, torch.zeros_like(stack_r))  # 층은 k=0 가정
        n_si = self.interp @ (self.si_n_init + _SI_N_SCALE * self.raw_si_n)
        k_si = softplus(self.interp @ (self.si_k_init + _SI_K_SCALE * self.raw_si_k))
        ns = torch.complex(n_si, -k_si)  # n − i·k (tmm.py 부호 관례, k ≥ 0)
        return lam, n_layers, ns

    def forward(self, d: Tensor) -> Tensor:
        """두께 d: (B, 4) [nm] → 재구성 반사율 R: (B, W) float64."""
        lam, n_layers, ns = self.spectra()
        return tmm_reflectance(d, n_layers, 1.0, ns, lam)


def load_calibrated_stack(path: Path | str) -> tuple[CalibratedStack, dict[str, Any]]:
    """model.pt에서 캘리브레이션 모델을 복원한다. 반환 (model, 체크포인트 dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CalibratedStack(
        n_channels=int(ckpt["model_cfg"]["n_channels"]),
        n_si_knots=int(ckpt["model_cfg"]["n_si_knots"]),
        lam_init=tuple(ckpt["model_cfg"]["lam_init"]),
        descending=bool(ckpt["model_cfg"]["descending"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def _fingerprint(cfg: dict[str, Any], model_cfg: dict[str, Any], n_fit: int) -> str:
    """resume 호환성 판별용 설정 지문 — 다른 설정의 resume.pt를 이어받지 않도록."""
    return json.dumps(
        {"model": model_cfg, "fit": cfg["fit"], "seed": cfg["seed"], "n_fit": n_fit},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


@torch.no_grad()
def _eval_rmse(model: CalibratedStack, d: Tensor, r_obs: Tensor, batch_size: int) -> float:
    """R 재구성 RMSE (전 행·전 채널) — 판정 게이트 (a)와 같은 정의."""
    sq_sum = 0.0
    for start in range(0, len(d), batch_size):
        pred = model(d[start : start + batch_size])
        sq_sum += float(((pred - r_obs[start : start + batch_size]) ** 2).sum())
    return float(np.sqrt(sq_sum / r_obs.numel()))


def search_lam_init(
    x_search: np.ndarray,
    d_search: np.ndarray,
    cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """초기 λ 범위 후보(오름/내림 × 범위)를 짧은 full-batch 피팅으로 훑는다.

    fringe 위상 정렬 때문에 손실 지형에 λ 스케일 방향의 국소최소가 많다 — 잘못된
    초기 범위에서 출발하면 본 피팅이 엉뚱한 fringe 차수에 안착할 수 있어, 후보별
    짧은 피팅의 최종 RMSE로 출발점을 고른다. 고정 행·full-batch·고정 스텝이라
    결정론적이고, 재실행 시 같은 결과가 나온다 (RNG 미사용).

    Args:
        x_search: (M, W) float32 — R_obs 표본.
        d_search: (M, 4) — 두께 [nm].
        cfg: 전체 config (search/model 섹션 사용).
        run_dir: 후보별 결과를 train.log에 기록.
        device: 계산 디바이스.

    Returns:
        (선택된 {"lam_init": [lo, hi], "descending": bool, "rmse": float}, 전 후보 결과).
    """
    search_cfg = cfg["search"]
    steps = int(search_cfg["steps"])
    lr = float(search_cfg["lr"])
    x_t = torch.from_numpy(x_search).to(device=device, dtype=torch.float64)
    d_t = torch.from_numpy(d_search).to(device=device, dtype=torch.float64)

    results: list[dict[str, Any]] = []
    for lam_lo, lam_hi in search_cfg["candidates"]:
        for descending in (False, True):
            model = CalibratedStack(
                n_channels=x_search.shape[1],
                n_si_knots=int(cfg["model"]["n_si_knots"]),
                lam_init=(float(lam_lo), float(lam_hi)),
                descending=descending,
            ).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(steps):
                loss = mse_loss(model(d_t), x_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            rmse = _eval_rmse(model, d_t, x_t, batch_size=len(d_t))
            row = {
                "lam_init": [float(lam_lo), float(lam_hi)],
                "descending": descending,
                "rmse": rmse,
            }
            results.append(row)
            arrow = "↓" if descending else "↑"
            log_line(run_dir, f"[search] λ [{lam_lo:g}, {lam_hi:g}] {arrow}: rmse {rmse:.5f}")

    best = min(results, key=lambda r: r["rmse"])
    log_line(
        run_dir,
        f"[search] 선택: λ {best['lam_init']} {'내림' if best['descending'] else '오름'}차순"
        f" (rmse {best['rmse']:.5f})",
    )
    return best, results


def fit_calibration(
    x_fit: np.ndarray,
    d_fit: np.ndarray,
    x_val: np.ndarray,
    d_val: np.ndarray,
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    lam_init: tuple[float, float],
    descending: bool,
    device: torch.device | None = None,
    mirror_dir: Path | None = None,
    resume: bool = True,
    _abort_after_eval: int | None = None,
) -> dict[str, Any]:
    """본 피팅 — 미니배치 Adam으로 CalibratedStack을 (d_true, R_obs)에 맞춘다.

    세션 유실 대비: best(val RMSE) 갱신 즉시 model.pt 저장, eval 블록마다
    resume.pt(+RNG·배치 generator) 저장·미러. 재개 시 무중단 실행과 동일 결과.

    Args:
        x_fit: (N, W) float32 — 피팅용 R_obs. d_fit: (N, 4) 두께 [nm].
        x_val / d_val: best 선택·게이트 (a) RMSE용 분리 표본.
        cfg: 전체 config (fit/model/seed 사용).
        run_dir: 산출물 디렉토리 (train.log / model.pt / resume.pt).
        lam_init: 초기 λ 범위 (후보 탐색 결과 또는 CLI 지정).
        descending: λ 채널 방향.
        mirror_dir: 지정 시 산출물을 매 eval 블록 복사 (Colab Drive 백업).
        resume: False면 resume.pt를 무시하고 처음부터.
        _abort_after_eval: 테스트 전용 — N번째 eval 블록 직후 RuntimeError로 중단.

    Returns:
        {"ckpt_path", "best_step", "best_val_rmse", "steps", "wall_sec",
         "gate_a": {"rmse", "threshold", "pass"}}
    """
    fit_cfg = cfg["fit"]
    steps_total = int(fit_cfg["steps"])
    batch_size = int(fit_cfg["batch_size"])
    eval_every = int(fit_cfg["eval_every"])
    eval_batch = int(fit_cfg.get("eval_batch", 8192))
    seed = int(cfg["seed"])
    if steps_total < 1 or eval_every < 1:
        raise ValueError(f"steps({steps_total})·eval_every({eval_every})는 1 이상이어야 한다")
    device = device or torch.device("cpu")

    model = CalibratedStack(
        n_channels=x_fit.shape[1],
        n_si_knots=int(cfg["model"]["n_si_knots"]),
        lam_init=lam_init,
        descending=descending,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(fit_cfg["lr"]))
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=str(fit_cfg.get("lr_schedule", "cosine")),
        warmup_steps=int(fit_cfg.get("warmup_steps", 0)),
        total_steps=steps_total,
    )
    # 배치 표집 전용 generator — 전역 RNG와 분리해 resume 복원을 단순·정확하게 한다.
    gen = torch.Generator().manual_seed(seed)

    x_fit_t = torch.from_numpy(x_fit).to(device=device, dtype=torch.float64)
    d_fit_t = torch.from_numpy(d_fit).to(device=device, dtype=torch.float64)
    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float64)
    d_val_t = torch.from_numpy(d_val).to(device=device, dtype=torch.float64)
    n = len(x_fit_t)

    fingerprint = _fingerprint(cfg, model.model_cfg, n)
    resume_path = run_dir / RESUME_NAME
    best_rmse = float("inf")
    best_step = -1
    done_steps = 0
    wall_prev = 0.0

    def save_best(step: int, rmse: float) -> None:
        """best 갱신 즉시 저장 — 세션이 언제 죽어도 최신 best가 남는다."""
        _atomic_save(
            {
                "model_cfg": model.model_cfg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": step,
                "val_rmse": rmse,
                "fingerprint": fingerprint,
            },
            run_dir / "model.pt",
        )
        _mirror_copy(run_dir, mirror_dir, ("model.pt",))

    if resume:
        if not resume_path.exists() and mirror_dir is not None:
            _mirror_copy(mirror_dir, run_dir, (RESUME_NAME,))
            if resume_path.exists():
                log_line(run_dir, "[calib] 미러에서 resume.pt 복원")
        if resume_path.exists():
            # resume.pt는 이 모듈이 만든 자기 산출물 — RNG 상태 등 비텐서 객체 포함
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise ValueError(
                    "resume.pt의 설정이 현재 config와 다르다 — "
                    "run_name을 바꾸거나 resume.pt를 지우고 다시 실행할 것"
                )
            model.load_state_dict(state["model"])
            model.to(device)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            gen.set_state(state["generator"])
            torch.set_rng_state(state["torch_rng"].cpu())
            np.random.set_state(state["numpy_rng"])  # noqa: NPY002 (seed.py와 동일 이유)
            random.setstate(state["py_rng"])
            best_rmse = state["best_val_rmse"]
            best_step = state["best_step"]
            done_steps = state["step"]
            wall_prev = state["wall_sec"]
            log_line(
                run_dir,
                f"[calib] resume: step {done_steps}까지 완료 상태에서 재개"
                f" (best {best_rmse:.5f} @ step {best_step})",
            )

    t_start = time.perf_counter()
    loss_sum = 0.0
    loss_cnt = 0
    eval_block = 0
    t_block = time.perf_counter()
    for step in range(done_steps + 1, steps_total + 1):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        loss = mse_loss(model(d_fit_t[idx]), x_fit_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item()
        loss_cnt += 1

        if step % eval_every == 0 or step == steps_total:
            val_rmse = _eval_rmse(model, d_val_t, x_val_t, eval_batch)
            marker = ""
            if val_rmse < best_rmse:
                best_rmse, best_step = val_rmse, step
                save_best(step, val_rmse)
                marker = " *"
            log_line(
                run_dir,
                f"[calib] step {step:5d}/{steps_total}  train_rmse"
                f" {np.sqrt(loss_sum / max(loss_cnt, 1)):.5f}  val_rmse {val_rmse:.5f}"
                f"  lr {optimizer.param_groups[0]['lr']:.2e}"
                f"  {time.perf_counter() - t_block:.1f}s{marker}",
            )
            loss_sum, loss_cnt = 0.0, 0
            t_block = time.perf_counter()
            _atomic_save(
                {
                    "fingerprint": fingerprint,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),  # noqa: NPY002
                    "py_rng": random.getstate(),
                    "best_val_rmse": best_rmse,
                    "best_step": best_step,
                    "wall_sec": wall_prev + time.perf_counter() - t_start,
                },
                resume_path,
            )
            _mirror_copy(run_dir, mirror_dir, ("train.log", RESUME_NAME))
            eval_block += 1
            if _abort_after_eval is not None and eval_block >= _abort_after_eval:
                raise RuntimeError("세션 중단(테스트 전용)")

    resume_path.unlink(missing_ok=True)
    wall_sec = wall_prev + time.perf_counter() - t_start
    ckpt_path = run_dir / "model.pt"
    if ckpt_path.is_relative_to(REPO_ROOT):
        ckpt_path = ckpt_path.relative_to(REPO_ROOT)
    gate_a = {"rmse": best_rmse, "threshold": GATE_A_RMSE, "pass": best_rmse < GATE_A_RMSE}
    log_line(
        run_dir,
        f"\n[calib] best val_rmse {best_rmse:.5f} @ step {best_step}"
        f" — 게이트 (a) RMSE < {GATE_A_RMSE}: {'통과' if gate_a['pass'] else '실패'}"
        f" (잔차 백색성 (c)는 scripts/diagnose_calibration.py로 별도 판정)",
    )
    return {
        "ckpt_path": str(ckpt_path),
        "best_step": best_step,
        "best_val_rmse": best_rmse,
        "steps": steps_total,
        "wall_sec": wall_sec,
        "gate_a": gate_a,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A 캘리브레이션")
    parser.add_argument("--config", required=True, help="configs/stage_a/*.yaml 경로")
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--fit-rows", type=int, default=None, help="피팅 행 수 (config 덮어씀)")
    parser.add_argument("--diag-rows", type=int, default=None, help="진단 행 수 (config 덮어씀)")
    parser.add_argument("--steps", type=int, default=None, help="피팅 스텝 수 (config 덮어씀)")
    parser.add_argument(
        "--lam-init", default=None, help='"400,800" — 초기 λ 범위 직접 지정 (후보 탐색 생략)'
    )
    parser.add_argument(
        "--descending", action="store_true", help="--lam-init와 함께: λ를 채널 내림차순으로"
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (기본: cuda 있으면 cuda)")
    parser.add_argument("--mirror-dir", default=None, help="산출물 미러 디렉토리 (Drive 백업)")
    parser.add_argument("--no-resume", action="store_true", help="resume.pt 무시, 처음부터")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.fit_rows is not None:
        cfg["data"]["fit_rows"] = args.fit_rows
    if args.diag_rows is not None:
        cfg["data"]["diag_rows"] = args.diag_rows
    if args.steps is not None:
        cfg["fit"]["steps"] = args.steps
    resume = not args.no_resume

    experiment = cfg.get("experiment")
    if not experiment:
        raise ValueError('config에 "experiment" 키가 필요하다 — runs/<experiment>/<run_name> 구조')
    run_dir = RUNS_DIR / str(experiment) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    prev: dict[str, Any] | None = None
    if metrics_path.exists():
        prev = json.loads(metrics_path.read_text())
    if resume and prev is not None and "result" in prev:
        log_line(run_dir, f"[calib] {experiment}/{cfg['run_name']}: 완료된 run — 스킵")
        return

    seed = int(cfg["seed"])
    device = resolve_device(args.device)
    data_cfg = cfg["data"]
    n_fit = int(data_cfg["fit_rows"])
    n_diag = int(data_cfg["diag_rows"])
    # 프로젝트 공통 holdout(학습 평가용)을 뺀 train 쪽에서만 표집한다 — Stage B 평가와
    # 물리 파라미터 피팅 사이의 정보 누설을 원천 차단 (파라미터 수백 개라 미미하지만).
    x, y, train_idx, _ = prepare_train_arrays(
        val_frac=float(data_cfg.get("val_frac", 0.1)), seed=seed
    )
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=n_fit + n_diag, replace=False)
    x_fit, d_fit = x[pick[:n_fit]], y[pick[:n_fit]]
    x_diag, d_diag = x[pick[n_fit:]], y[pick[n_fit:]]
    del x, y
    log_line(
        run_dir,
        f"run {experiment}/{cfg['run_name']}: fit {n_fit:,} + diag {n_diag:,} 행"
        f" (holdout 제외 train에서 표집) / device={device.type}",
    )

    # 초기 λ 선택 — CLI 지정 > 이전 실행의 선택(재개) > 후보 탐색.
    search_results: list[dict[str, Any]] | None = None
    if args.lam_init is not None:
        lo, hi = (float(v) for v in args.lam_init.split(","))
        selected = {"lam_init": [lo, hi], "descending": bool(args.descending), "rmse": None}
        log_line(run_dir, f"[search] CLI 지정 λ {selected['lam_init']} — 후보 탐색 생략")
    elif resume and prev is not None and "lam_init_selected" in prev:
        selected = prev["lam_init_selected"]
        search_results = prev.get("search")
        log_line(run_dir, f"[search] 이전 실행의 선택 재사용: {selected}")
    else:
        rows = int(cfg["search"]["rows"])
        selected, search_results = search_lam_init(x_fit[:rows], d_fit[:rows], cfg, run_dir, device)

    metrics: dict[str, Any] = {
        "experiment": experiment,
        "run_name": cfg["run_name"],
        "seed": seed,
        "rows": {"fit": n_fit, "diag": n_diag},
        "config": cfg,
        "lam_init_selected": selected,
        "search": search_results,
    }
    # 시작 시점 설정·선택 스냅샷 (중단돼도 남고, 재개 시 lam_init_selected를 재사용한다).
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    result = fit_calibration(
        x_fit,
        d_fit,
        x_diag,
        d_diag,
        cfg,
        run_dir,
        lam_init=tuple(selected["lam_init"]),
        descending=bool(selected["descending"]),
        device=device,
        mirror_dir=None if args.mirror_dir is None else Path(args.mirror_dir),
        resume=resume,
    )
    metrics["result"] = result
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    _mirror_copy(
        run_dir,
        None if args.mirror_dir is None else Path(args.mirror_dir),
        ("metrics.json", "train.log", "model.pt"),
    )
    log_line(run_dir, f"\n산출물: {run_dir}/ (metrics.json, train.log, model.pt)")


if __name__ == "__main__":
    main()
