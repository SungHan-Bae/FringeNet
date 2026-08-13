"""Stage A 캘리브레이션의 단위 테스트 — 파라미터화 계약과 식별 복원.

대회 데이터·GPU 없이 전부 돈다. 합성 데이터는 `PhysicalStack` 자신(참 파라미터로
설정)이 만들고, 그것을 되찾을 수 있는지 확인한다.

여기서 지키려는 계약:
  1. **게이지 고정** — SiO₂ n(λ)는 어떤 파라미터를 흔들어도 문헌값에서 움직이지 않는다.
     이게 깨지면 λ와 n이 동시에 자유로워져 해가 하나로 정해지지 않는다.
  2. **자유도가 선언한 것만 움직인다** — `free`에 없는 파라미터는 초기값에 고정.
  3. **물리 제약** — λ 강단조·양수, k_Si > 0, 0 ≤ R < 1.
  4. **주파수 식별의 닫힌형 복원** — 전수 격자 합성 데이터에서 참 λ를 되찾는다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.calibrate import (
    GATE_A_RMSE,
    NOISE_BOUND,
    NOISE_SIGMA,
    PARAM_NAMES,
    PhysicalStack,
    fit_lam_coefficients,
    residual_stats,
)
from src.physics.freq_id import identify_wavelength_grid

DTYPE = torch.float64
# 테스트용 λ 계수 — 실데이터 식별 결과와 같은 형태 (1/λ가 채널에 거의 선형).
LAM_COEFFS = (0.00126322, 1.799421, -0.008633)


def _stack(free: tuple[str, ...] = (), **kwargs: object) -> PhysicalStack:
    defaults: dict = {"n_channels": 32, "lam_coeffs": LAM_COEFFS, "free": free}
    defaults.update(kwargs)
    return PhysicalStack(**defaults)  # type: ignore[arg-type]


def _perturb(model: PhysicalStack, scale: float = 1.0, seed: int = 0) -> None:
    """자유 파라미터를 무작위로 흔든다 (계약이 섭동 후에도 유지되는지 보기 위해)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.theta.add_(scale * torch.randn(model.theta.shape, generator=gen, dtype=DTYPE))


# ---------------------------------------------------------------------------
# 게이트 상수 — 노이즈 측정값에서 유도된 값이라 고정한다
# ---------------------------------------------------------------------------


def test_gate_constants_are_derived_from_measured_noise() -> None:
    """σ는 측정값(scripts/measure_noise.py), 임계는 1.2σ, 상한은 유계 노이즈 관측값."""
    assert pytest.approx(0.008658) == NOISE_SIGMA
    assert pytest.approx(1.2 * NOISE_SIGMA) == GATE_A_RMSE
    # 균등분포 가정의 폭 σ√3 ≈ 0.015 와 관측 최소값 −0.015117 사이에 있어야 한다.
    assert NOISE_SIGMA * np.sqrt(3.0) <= NOISE_BOUND <= 0.016


# ---------------------------------------------------------------------------
# λ 3계수 적합
# ---------------------------------------------------------------------------


def test_fit_lam_coefficients_recovers_smooth_curve() -> None:
    """1/λ이 채널의 2차 다항식인 그리드에서 계수를 정확히 되찾는다."""
    u = np.linspace(0.0, 1.0, 226)
    nu0, r1, r2 = LAM_COEFFS
    lam = 1.0 / (nu0 * (1.0 + r1 * u + r2 * u**2))
    got = fit_lam_coefficients(lam)
    assert got == pytest.approx(LAM_COEFFS, rel=1e-9)


def test_fit_lam_coefficients_is_robust_to_outliers() -> None:
    """식별 실패 채널이 섞여도 강건 적합이 계수를 지켜야 한다."""
    u = np.linspace(0.0, 1.0, 226)
    nu0, r1, r2 = LAM_COEFFS
    lam = 1.0 / (nu0 * (1.0 + r1 * u + r2 * u**2))
    corrupted = lam.copy()
    corrupted[[10, 50, 51, 120, 200]] *= 1.35  # 35% 튀는 채널 5개
    got = fit_lam_coefficients(corrupted)
    assert got == pytest.approx(LAM_COEFFS, rel=2e-3)


# ---------------------------------------------------------------------------
# PhysicalStack — 파라미터화 계약
# ---------------------------------------------------------------------------


def test_gauge_sio2_is_frozen_under_any_perturbation() -> None:
    """SiO₂ n(λ)는 λ가 움직여도 **문헌 곡선 위에** 있어야 한다 (게이지 고정).

    λ가 바뀌면 평가 지점이 바뀌니 값 자체는 변한다. 검증할 것은 "그 λ에서의
    Malitson 값"과 일치하는지다 — 즉 SiO₂에 자유도가 없다는 것.
    """
    from src.physics.dispersion import sio2_n

    model = _stack(free=("lam_nu0", "lam_r1", "sin_b1"))
    _perturb(model, scale=2.0)
    with torch.no_grad():
        lam, n_layers, _ = model.spectra()
    assert np.allclose(n_layers[1].real.numpy(), sio2_n(lam.numpy()), atol=1e-12)
    assert np.allclose(n_layers[3].real.numpy(), sio2_n(lam.numpy()), atol=1e-12)


def test_sio2_scale_is_the_only_way_to_release_the_gauge() -> None:
    """`sio2_scale`은 기본 동결이고, 자유로 둘 때만 SiO₂가 문헌에서 벗어난다.

    게이지 검정(`configs/stage_a/gauge-sio2-scale.yaml`)이 성립하려면 이 손잡이가
    **정확히 배율로만** 작동해야 한다 — 그래야 적합값 1.0이 "λ 스케일이 맞다"로 읽힌다.
    """
    from src.physics.dispersion import sio2_n

    frozen = _stack(free=("lam_nu0", "sin_b1"))
    assert frozen.physical_values()["sio2_scale"] == pytest.approx(1.0)

    released = _stack(free=("sio2_scale",))
    _perturb(released, scale=4.0)
    scale = released.physical_values()["sio2_scale"]
    assert scale != pytest.approx(1.0)
    with torch.no_grad():
        lam, n_layers, _ = released.spectra()
    # λ는 동결이므로 평가 지점이 같다 → 문헌값의 정확한 배수여야 한다.
    expected = sio2_n(lam.numpy()) * scale
    assert np.allclose(n_layers[1].real.numpy(), expected, atol=1e-12)
    assert np.allclose(n_layers[3].real.numpy(), expected, atol=1e-12)


def test_holdout_channel_range_excludes_a_contiguous_block(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`holdout_channel_range`는 연속 블록을 피팅에서 빼고 그 채널을 별도로 평가한다.

    균등 간격판(`holdout_channels`)은 이웃 상관 때문에 사실상 보간이라 요구가 약하다 —
    연속 블록이 진짜 외삽 검정이다 (08-13 리뷰).
    """
    from src.calibrate import fit_physical

    n_ch = 32
    rng = np.random.default_rng(0)
    model = _stack(n_channels=n_ch)
    d = rng.uniform(20.0, 280.0, size=(24, 4))
    with torch.no_grad():
        x = model(torch.from_numpy(d).to(DTYPE)).numpy()
    data = {"x_fit": x, "d_fit": d, "x_diag": x, "d_diag": d}
    cfg = {"model": {"free": [], "holdout_channel_range": [24, 31]}}

    out = fit_physical(data, cfg, tmp_path, lam_coeffs=LAM_COEFFS)

    assert out["holdout"]["channels"] == list(range(24, 32))
    assert out["n_fit_channels"] == n_ch - 8

    with pytest.raises(ValueError, match="함께 줄 수 없다"):
        fit_physical(
            data,
            {"model": {"free": [], "holdout_channels": 4, "holdout_channel_range": [0, 3]}},
            tmp_path,
            lam_coeffs=LAM_COEFFS,
        )
    with pytest.raises(ValueError, match="범위 밖"):
        fit_physical(
            data,
            {"model": {"free": [], "holdout_channel_range": [30, n_ch]}},
            tmp_path,
            lam_coeffs=LAM_COEFFS,
        )


def test_load_physical_stack_accepts_checkpoints_missing_later_params(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """파라미터를 추가해도 기존 체크포인트가 로드돼야 한다 (하위 호환).

    `init_*` 버퍼는 PARAM_NAMES가 늘면 함께 늘어난다. 이 장치가 없으면 손잡이 하나를
    추가하는 순간 커밋된 run 전부가 로드 불가가 된다.
    """
    from src.calibrate import load_physical_stack

    model = _stack(free=("sin_b1",))
    state = {k: v for k, v in model.state_dict().items() if k != "init_sio2_scale"}
    torch.save({"model_cfg": model.model_cfg, "state_dict": state}, tmp_path / "model.pt")

    restored, _ = load_physical_stack(tmp_path / "model.pt")
    assert restored.physical_values()["sio2_scale"] == pytest.approx(1.0)
    with torch.no_grad():
        assert torch.allclose(restored.lam(), model.lam())

    bogus = {**state, "init_sin_b1": state["init_sin_b1"]}
    bogus.pop("theta")
    torch.save({"model_cfg": model.model_cfg, "state_dict": bogus}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="init_. 버퍼가 아니다"):
        load_physical_stack(tmp_path / "bad.pt")


def test_frozen_parameters_do_not_move() -> None:
    """`free`에 없는 파라미터는 섭동 후에도 초기값이어야 한다."""
    model = _stack(free=("sin_b1",))
    before = model.physical_values()
    _perturb(model, scale=3.0)
    after = model.physical_values()
    assert after["sin_b1"] != pytest.approx(before["sin_b1"])
    for name in PARAM_NAMES:
        if name != "sin_b1":
            assert after[name] == pytest.approx(before[name])


def test_lam_grid_is_positive_and_strictly_monotone() -> None:
    """λ는 양수·강단조여야 한다 (분광기 격자 분산)."""
    model = _stack(free=("lam_nu0", "lam_r1", "lam_r2"))
    for seed in range(5):
        _perturb(model, scale=1.0, seed=seed)
        with torch.no_grad():
            lam = model.lam().numpy()
        assert (lam > 0).all()
        assert np.all(np.diff(lam) < 0) or np.all(np.diff(lam) > 0)


def test_si_k_positive_and_sign_convention() -> None:
    """k_Si > 0 이고 기판 굴절률은 n − i·k 관례를 따라야 한다."""
    model = _stack(free=("si_klog", "si_de"))
    _perturb(model, scale=2.0)
    with torch.no_grad():
        _, _, ns = model.spectra()
    assert (ns.imag < 0).all()  # n − i·k, k > 0
    assert ((-ns.imag) > 0).all()


def test_forward_shape_and_physical_range() -> None:
    """R: (B, W) 이고 0 ≤ R < 1 이어야 한다."""
    model = _stack(free=("sin_b1",))
    d = torch.tensor([[10.0, 300.0, 150.0, 20.0], [100.0, 100.0, 100.0, 100.0]], dtype=DTYPE)
    with torch.no_grad():
        r = model(d)
    assert r.shape == (2, 32)
    assert r.dtype == DTYPE
    assert (r >= 0).all() and (r < 1).all()


def test_unknown_free_parameter_raises() -> None:
    """오타를 조용히 무시하지 않는다 (자유도를 잘못 세면 판정이 무의미해진다)."""
    with pytest.raises(ValueError, match="모르는 자유 파라미터"):
        _stack(free=("sin_b9",))


def test_model_cfg_round_trips() -> None:
    """model_cfg만으로 같은 구조를 복원할 수 있어야 한다 (체크포인트 계약)."""
    model = _stack(free=("lam_nu0", "sin_b1"), si_source="Si_nk_Green-2008.yml")
    clone = PhysicalStack(**model.model_cfg)
    with torch.no_grad():
        clone.theta.copy_(model.theta)
        assert torch.allclose(clone.lam(), model.lam())
    assert clone.free == model.free
    assert clone.si_source == model.si_source


def test_coarse_si_source_differs_from_literature_table() -> None:
    """대조군(`coarse`)이 실제로 다른 Si 곡선을 써야 한다 — ablation의 전제."""
    lit = _stack(si_source="Si_nk_Aspnes.yml")
    coarse = _stack(si_source="coarse")
    with torch.no_grad():
        n_lit = lit.spectra()[2].real.numpy()
        n_coarse = coarse.spectra()[2].real.numpy()
    assert np.abs(n_lit - n_coarse).max() > 0.1


# ---------------------------------------------------------------------------
# 잔차 통계 — 유계 노이즈 위반 판정
# ---------------------------------------------------------------------------


def test_residual_stats_counts_bounded_noise_violations() -> None:
    """상한을 넘는 관측을 정확히 센다 (게이트 (b)의 핵심 계산)."""
    model = _stack()
    d = torch.tensor([[100.0, 100.0, 100.0, 100.0]], dtype=DTYPE).repeat(4, 1)
    with torch.no_grad():
        clean = model(d).numpy()
    obs = clean.copy()
    obs[0, 0] += NOISE_BOUND * 2.0  # 확실한 위반 1건
    obs[1, 1] += NOISE_BOUND * 0.5  # 상한 이내
    stats = residual_stats(model, obs.astype(np.float32), d.numpy())
    assert stats["n_obs"] == obs.size
    assert stats["violation_rate"] == pytest.approx(1.0 / obs.size)
    assert stats["max_abs_residual"] > NOISE_BOUND

    channels = np.array([1, 2, 3])  # 위반 채널(0)을 제외하면 0%
    masked = residual_stats(model, obs.astype(np.float32), d.numpy(), channels=channels)
    assert masked["violation_rate"] == 0.0


# ---------------------------------------------------------------------------
# 두께축 주파수 식별 — 닫힌형 복원
# ---------------------------------------------------------------------------


def test_identify_wavelength_grid_recovers_synthetic_lambda() -> None:
    """전수 격자 합성 데이터에서 참 λ 그리드를 닫힌형으로 되찾는지 검증.

    조건부 평균이 정확한 주변화가 되도록 실데이터처럼 **전수 격자**로 생성한다
    (15값 20 nm 격자 × 4층 = 50,625행 — 무작위 표본은 bin 노이즈로 추정이 흔들린다).
    Nyquist: 1/(2·20 nm) = 0.025 > f_max 0.022 를 지킨다.
    """
    n_ch = 32
    u = np.linspace(0.0, 1.0, n_ch)
    nu0, r1, r2 = LAM_COEFFS
    truth_grid = 1.0 / (nu0 * (1.0 + r1 * u + r2 * u**2))  # 내림차순 792→284 nm
    truth = PhysicalStack(n_channels=n_ch, lam_coeffs=LAM_COEFFS)

    vals = np.arange(20.0, 320.0, 20.0)
    mesh = np.stack(np.meshgrid(*([vals] * 4), indexing="ij"), axis=-1).reshape(-1, 4)
    d = torch.from_numpy(mesh).to(DTYPE)
    r = np.empty((len(d), n_ch), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(d), 8192):
            r[s : s + 8192] = truth(d[s : s + 8192]).numpy()
        n_sin_truth = truth.spectra()[1][0].real.numpy()

    ident = identify_wavelength_grid(r, mesh)
    lam_est = ident["lam_grid"]
    diag = ident["diagnostics"]

    assert diag["descending"]
    assert np.all(np.diff(lam_est) < 0)
    # λ 복원: 주파수 격자 해상도(~0.5–3 nm) + 조화 근사 오차 허용
    assert float(np.median(np.abs(lam_est - truth_grid))) < 3.0
    assert float(np.abs(lam_est - truth_grid).max()) < 10.0
    # n_SiN 복원 (f₁·λ/2)
    assert float(np.abs(ident["n_sin_samples"][1] - n_sin_truth).max()) < 0.05
    # 자체 검증 수치: 같은 물리량의 층별 독립 추정이 일치해야 한다
    assert diag["lam24_dev_median"] < 3.0
    assert diag["n_sin13_dev_median"] < 0.02


def test_identify_wavelength_grid_rejects_incomplete_grid() -> None:
    """격자를 덮지 못한 표본은 조용히 통과시키지 않는다 (조건부 평균이 무의미해진다).

    layer_1이 100 nm만 갖는데 다른 층이 200 nm를 가지면, 조건 E[R | d₁ = 200]의
    표본이 비어 주변화가 성립하지 않는다 — 그걸 잡아야 한다.
    """
    d = np.full((16, 4), 100.0)
    d[:, 1] = 200.0  # 격자값 {100, 200} 중 layer_1은 100을 못 가진다
    x = np.zeros((16, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="행이 없다"):
        identify_wavelength_grid(x, d)


def test_identify_wavelength_grid_reports_total_failure() -> None:
    """전 채널 식별이 실패하면 numpy 오류가 아니라 원인을 말하는 예외를 던진다."""
    vals = np.arange(20.0, 320.0, 20.0)
    mesh = np.stack(np.meshgrid(*([vals] * 4), indexing="ij"), axis=-1).reshape(-1, 4)
    flat = np.zeros((len(mesh), 8), dtype=np.float32)  # 무늬가 전혀 없는 스펙트럼
    with pytest.raises(ValueError, match="모든 채널의 주파수 식별이 실패"):
        identify_wavelength_grid(flat, mesh)
