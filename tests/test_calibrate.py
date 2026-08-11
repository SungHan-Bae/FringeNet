"""Stage A 캘리브레이션(src/calibrate.py·src/physics/dispersion.py) 단위 테스트.

대회 데이터·GPU 없이 전부 돈다 — 합성 데이터는 CalibratedStack 자신(참 파라미터로
섭동한 사본)으로 생성한다. 채널 수를 줄인 작은 스택으로 빠르게 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.calibrate import (
    CalibratedStack,
    fit_calibration,
    identify_initial_grid,
    load_calibrated_stack,
)
from src.physics.dispersion import (
    cauchy_n,
    fit_cauchy,
    linear_interp_matrix,
    si3n4_n,
    si_nk,
    sio2_n,
    softplus_inverse,
)

# ---------------------------------------------------------------------------
# dispersion — 문헌값·유틸
# ---------------------------------------------------------------------------


def test_sio2_literature_value() -> None:
    # fused silica의 n_d (587.6 nm) = 1.4585 — Malitson 1965의 대표값.
    assert sio2_n(np.array([587.6]))[0] == pytest.approx(1.4585, abs=2e-3)


def test_si3n4_plausible_range() -> None:
    lam = np.linspace(400.0, 800.0, 50)
    n = si3n4_n(lam)
    assert np.all((n > 1.9) & (n < 2.2))
    assert np.all(np.diff(n) < 0)  # 정상 분산: λ↑ → n↓


def test_si_table_plausible() -> None:
    lam = np.linspace(400.0, 900.0, 60)
    n, k = si_nk(lam)
    assert np.all(np.diff(n) <= 0)  # 가시광에서 단조 감소
    assert np.all(k >= 0)
    assert 3.8 < si_nk(np.array([600.0]))[0][0] < 4.1


def test_fit_cauchy_reproduces_sellmeier() -> None:
    lam = np.linspace(400.0, 800.0, 226)
    coeffs = fit_cauchy(lam, sio2_n(lam))
    fitted = cauchy_n(torch.from_numpy(lam), torch.from_numpy(coeffs)).numpy()
    assert np.max(np.abs(fitted - sio2_n(lam))) < 2e-4


def test_linear_interp_matrix() -> None:
    p = linear_interp_matrix(5, 3)
    assert p.shape == (5, 3)
    assert torch.allclose(p.sum(dim=1), torch.ones(5, dtype=torch.float64))
    assert torch.all(p >= 0)
    knots = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64)
    curve = p @ knots
    # 끝점·knot 위치는 knot값 그대로, 중간은 선형 보간.
    assert curve[0] == pytest.approx(1.0)
    assert curve[2] == pytest.approx(3.0)
    assert curve[4] == pytest.approx(2.0)
    assert curve[1] == pytest.approx(2.0)  # (1+3)/2


def test_softplus_inverse_roundtrip() -> None:
    y = torch.tensor([1e-4, 0.5, 1.77, 400.0, 1000.0], dtype=torch.float64)
    assert torch.allclose(torch.nn.functional.softplus(softplus_inverse(y)), y, rtol=1e-12)


# ---------------------------------------------------------------------------
# CalibratedStack — 파라미터화 계약
# ---------------------------------------------------------------------------


def _small_stack(**kwargs: object) -> CalibratedStack:
    defaults: dict = {"n_channels": 32, "n_si_knots": 5, "lam_init": (400.0, 800.0)}
    defaults.update(kwargs)
    return CalibratedStack(**defaults)


def _perturb(model: CalibratedStack, scale: float = 1.0, seed: int = 0) -> None:
    """raw 파라미터를 무작위로 흔든다 (제약이 섭동 후에도 유지되는지 검증용)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def test_lam_grid_monotone_after_perturbation() -> None:
    for descending in (False, True):
        model = _small_stack(descending=descending)
        _perturb(model, scale=2.0)
        lam = model.lam()
        assert lam.shape == (32,)
        assert torch.all(lam > 0)
        diffs = lam.diff()
        assert torch.all(diffs < 0) if descending else torch.all(diffs > 0)


def test_gauge_sio2_frozen() -> None:
    model = _small_stack()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"raw_lam_min", "raw_dlam", "raw_sin", "raw_si_n", "raw_si_k"}

    # 최적화 스텝을 밟아도 SiO₂ Cauchy(buffer)는 변하지 않고 SiN은 움직인다.
    sio2_before = model.sio2_cauchy.clone()
    _, n_layers_before, _ = model.spectra()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    d = torch.rand(8, 4, dtype=torch.float64) * 290 + 10
    for _ in range(3):
        loss = model(d).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(model.sio2_cauchy, sio2_before)
    _, n_layers_after, _ = model.spectra()
    assert not torch.allclose(n_layers_after[0], n_layers_before[0])  # SiN(layer 1) 갱신
    # λ가 움직이므로 SiO₂ 채널값은 변할 수 있으나, 함수 자체(계수)는 고정이 계약이다.


def test_si_k_nonnegative_and_sign_convention() -> None:
    model = _small_stack()
    _perturb(model, scale=3.0, seed=7)
    _, n_layers, ns = model.spectra()
    assert torch.all(ns.imag <= 0)  # n − i·k, k ≥ 0 (tmm.py 부호 관례)
    assert torch.all(n_layers.imag == 0)  # 층은 k=0 가정
    assert torch.all(n_layers.real > 1.0)


def test_stack_accepts_explicit_lam_grid() -> None:
    grid = np.linspace(700.0, 300.0, 32)  # 채널 내림차순
    model = CalibratedStack(n_channels=32, n_si_knots=5, lam_grid=grid)
    assert model.descending
    assert np.allclose(model.lam().detach().numpy(), grid)
    with pytest.raises(ValueError, match="단조"):
        CalibratedStack(n_channels=32, n_si_knots=5, lam_grid=np.ones(32))


def test_forward_shapes_and_physical_range() -> None:
    model = _small_stack()
    d = torch.rand(16, 4, dtype=torch.float64) * 290 + 10
    r = model(d)
    assert r.shape == (16, 32)
    assert r.dtype == torch.float64
    assert torch.all((r >= 0) & (r <= 1))  # 무흡수층 + 흡수 기판 → 0 ≤ R ≤ 1
    grad = torch.autograd.grad(r.sum(), model.raw_lam_min)[0]
    assert torch.isfinite(grad)


# ---------------------------------------------------------------------------
# fit_calibration — 합성 복원·체크포인트·resume 계약
# ---------------------------------------------------------------------------


def _synthetic_problem(
    n_rows: int = 256, seed: int = 0
) -> tuple[CalibratedStack, np.ndarray, np.ndarray]:
    """참 스택(문헌 초기값에서 섭동)과 그로부터 생성한 (R_obs, d) 표본."""
    truth = _small_stack()
    with torch.no_grad():
        truth.raw_lam_min.add_(0.15)  # λ 전체 +15 nm 이동
        truth.raw_sin.add_(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64))
        truth.raw_si_n.add_(0.5)
        truth.raw_si_k.add_(0.4)
    gen = torch.Generator().manual_seed(seed)
    d = (torch.rand(n_rows, 4, generator=gen, dtype=torch.float64) * 290 + 10).round()
    with torch.no_grad():
        r = truth(d)
    return truth, r.numpy().astype(np.float32), d.numpy().astype(np.float32)


def _fit_cfg(steps: int = 60, eval_every: int = 20) -> dict:
    return {
        "seed": 0,
        "model": {"n_si_knots": 5},
        "fit": {
            "steps": steps,
            "batch_size": 128,
            "lr": 1.0e-2,
            "lr_schedule": "cosine",
            "warmup_steps": 5,
            "eval_every": eval_every,
            "eval_batch": 256,
        },
    }


def test_identify_initial_grid_recovers_synthetic(tmp_path: Path) -> None:
    """주파수 식별이 합성 데이터의 λ 그리드·n_SiN을 닫힌형으로 복원하는지 검증.

    조건부 평균이 정확한 주변화가 되도록 실데이터처럼 **전수 격자**로 생성한다
    (15값 20 nm 격자 × 4층 = 50,625행 — 무작위 표본은 bin 노이즈로 추정이 흔들린다).
    """
    truth_grid = np.linspace(750.0, 310.0, 32)  # 내림차순
    truth = CalibratedStack(n_channels=32, n_si_knots=5, lam_grid=truth_grid)
    vals = np.arange(20.0, 320.0, 20.0)  # 15개 값, Nyquist 1/(2·20) > f_max 0.022 유지
    mesh = np.stack(np.meshgrid(vals, vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 4)
    d = torch.from_numpy(mesh).to(torch.float64)
    r = np.empty((len(d), 32), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(d), 8192):
            r[s : s + 8192] = truth(d[s : s + 8192]).numpy()
        n_sin_truth = truth.spectra()[1][0].real.numpy()

    ident = identify_initial_grid(r, mesh.astype(np.float32), tmp_path)
    lam_est = ident["lam_grid"]
    assert ident["diagnostics"]["descending"]
    assert np.all(np.diff(lam_est) < 0)
    # λ 복원: 주파수 그리드 해상도(~0.5–3 nm) + 조화 근사 오차 허용
    assert float(np.median(np.abs(lam_est - truth_grid))) < 3.0
    assert float(np.abs(lam_est - truth_grid).max()) < 10.0
    # n_SiN 복원 (f₁·λ/2)
    assert float(np.abs(ident["n_sin_samples"][1] - n_sin_truth).max()) < 0.05


def test_fit_recovers_synthetic_truth(tmp_path: Path) -> None:
    truth, r_obs, d = _synthetic_problem()
    init_rmse_model = _small_stack()
    with torch.no_grad():
        init_pred = init_rmse_model(torch.from_numpy(d).to(torch.float64))
    init_rmse = float(((init_pred.numpy() - r_obs.astype(np.float64)) ** 2).mean() ** 0.5)

    result = fit_calibration(
        r_obs[:192],
        d[:192],
        r_obs[192:],
        d[192:],
        _fit_cfg(steps=800, eval_every=200),
        tmp_path,
        lam_init=(400.0, 800.0),
        descending=False,
    )
    # 노이즈 없는 합성 문제 — 초기 대비 큰 폭으로 내려가야 학습이 실제로 작동하는 것
    # (실측: 800스텝에 ~240× 감소. 여유를 두고 20×를 요구한다).
    assert result["best_val_rmse"] < 0.05 * init_rmse

    model, ckpt = load_calibrated_stack(tmp_path / "model.pt")
    assert ckpt["val_rmse"] == pytest.approx(result["best_val_rmse"])
    with torch.no_grad():
        pred = model(torch.from_numpy(d[192:]).to(torch.float64))
    rmse = float(((pred.numpy() - r_obs[192:].astype(np.float64)) ** 2).mean() ** 0.5)
    assert rmse == pytest.approx(result["best_val_rmse"], rel=1e-6)  # 체크포인트 왕복 일치


def test_resume_after_interrupt_matches_uninterrupted(tmp_path: Path) -> None:
    _, r_obs, d = _synthetic_problem(seed=3)
    cfg = _fit_cfg(steps=60, eval_every=20)
    kwargs: dict = {"lam_init": (400.0, 800.0), "descending": False}

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = fit_calibration(r_obs[:192], d[:192], r_obs[192:], d[192:], cfg, full_dir, **kwargs)

    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        fit_calibration(
            r_obs[:192],
            d[:192],
            r_obs[192:],
            d[192:],
            cfg,
            part_dir,
            _abort_after_eval=1,
            **kwargs,
        )
    assert (part_dir / "resume.pt").exists()
    assert (part_dir / "model.pt").exists()  # best는 갱신 즉시 저장 — 중단 시점에도 존재
    resumed = fit_calibration(r_obs[:192], d[:192], r_obs[192:], d[192:], cfg, part_dir, **kwargs)

    assert resumed["best_step"] == full["best_step"]
    assert resumed["best_val_rmse"] == pytest.approx(full["best_val_rmse"], abs=1e-12)
    full_model, _ = load_calibrated_stack(full_dir / "model.pt")
    part_model, _ = load_calibrated_stack(part_dir / "model.pt")
    for (name, a), (_, b) in zip(
        full_model.state_dict().items(), part_model.state_dict().items(), strict=True
    ):
        assert torch.allclose(a, b, atol=1e-12), name
    assert not (part_dir / "resume.pt").exists()  # 완료 시 재개 상태는 정리된다
    assert "resume" in (part_dir / "train.log").read_text()


def test_resume_rejects_mismatched_config(tmp_path: Path) -> None:
    _, r_obs, d = _synthetic_problem(seed=5)
    cfg = _fit_cfg(steps=40, eval_every=20)
    kwargs: dict = {"lam_init": (400.0, 800.0), "descending": False}
    with pytest.raises(RuntimeError, match="중단"):
        fit_calibration(
            r_obs[:192],
            d[:192],
            r_obs[192:],
            d[192:],
            cfg,
            tmp_path,
            _abort_after_eval=1,
            **kwargs,
        )
    other = _fit_cfg(steps=40, eval_every=20)
    other["fit"]["lr"] = 9.9e-3
    with pytest.raises(ValueError, match="resume"):
        fit_calibration(r_obs[:192], d[:192], r_obs[192:], d[192:], other, tmp_path, **kwargs)
