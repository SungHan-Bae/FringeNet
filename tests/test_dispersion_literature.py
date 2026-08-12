"""문헌 분산 데이터 로더·스플라인의 단위 테스트 (Stage A).

여기서 지키려는 것은 **출처의 무결성**이다. 문헌 표를 손으로 옮겨 적으면 뾰족한
임계점 구조가 조용히 깎인다 (거친 19점 대조군이 E1 봉우리를 4.3% 깎고 단파장 n을
최대 0.43 틀리는 것이 그 예다). 그래서 refractiveindex.info 원본 파일
(`src/physics/literature/*.yml`, CC0)을 그대로 읽는데, 그 로더가 (1) 파일의 수치를
정확히 재현하고 (2) 코드에 하드코딩된 Sellmeier 계수와 일치하는지를 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.physics.dispersion import (
    HC_EV_NM,
    SI3N4_LUKE_SELLMEIER,
    SIO2_MALITSON_SELLMEIER,
    CoarseTableNK,
    TabulatedNK,
    load_tabulated_nk,
    sellmeier_n_t,
    si3n4_n,
    si_nk_coarse_table,
    sio2_n,
)

DTYPE = torch.float64


def test_literature_files_match_hardcoded_sellmeier() -> None:
    """문헌 원본 파일의 formula-1 계수가 모듈 상수와 일치해야 한다 (전사 오류 방지)."""
    from src.physics.dispersion import (
        _SI3N4_SELLMEIER_B,
        _SI3N4_SELLMEIER_C_UM2,
        _SIO2_SELLMEIER_B,
        _SIO2_SELLMEIER_C_UM2,
    )

    assert SIO2_MALITSON_SELLMEIER[0] == pytest.approx(_SIO2_SELLMEIER_B)
    assert SIO2_MALITSON_SELLMEIER[1] == pytest.approx(_SIO2_SELLMEIER_C_UM2)
    assert SI3N4_LUKE_SELLMEIER[0] == pytest.approx(_SI3N4_SELLMEIER_B)
    assert SI3N4_LUKE_SELLMEIER[1] == pytest.approx(_SI3N4_SELLMEIER_C_UM2)


@pytest.mark.parametrize(
    ("coeffs", "reference"),
    [(SIO2_MALITSON_SELLMEIER, sio2_n), (SI3N4_LUKE_SELLMEIER, si3n4_n)],
)
def test_sellmeier_torch_matches_numpy(coeffs: tuple, reference) -> None:  # noqa: ANN001
    """미분가능 Sellmeier가 기존 numpy 구현과 부동소수점 수준으로 같아야 한다."""
    lam = torch.linspace(284.0, 793.0, 226, dtype=DTYPE)
    got = sellmeier_n_t(
        lam, torch.tensor(coeffs[0], dtype=DTYPE), torch.tensor(coeffs[1], dtype=DTYPE)
    )
    assert torch.allclose(got, torch.from_numpy(reference(lam.numpy())), atol=1e-14)


def test_sellmeier_is_differentiable_in_wavelength() -> None:
    """λ로 미분가능하고, 정상 분산이면 dn/dλ < 0 이어야 한다 (λ 학습의 전제)."""
    lam = torch.linspace(300.0, 780.0, 64, dtype=DTYPE, requires_grad=True)
    n = sellmeier_n_t(
        lam,
        torch.tensor(SIO2_MALITSON_SELLMEIER[0], dtype=DTYPE),
        torch.tensor(SIO2_MALITSON_SELLMEIER[1], dtype=DTYPE),
    )
    n.sum().backward()
    assert lam.grad is not None
    assert torch.isfinite(lam.grad).all()
    assert (lam.grad < 0).all()


@pytest.mark.parametrize("filename", ["Si_nk_Aspnes.yml", "Si_nk_Green-2008.yml"])
def test_tabulated_nk_interpolates_through_its_knots(filename: str) -> None:
    """스플라인은 표의 절점을 정확히 지나야 한다 (보간이지 근사가 아니다)."""
    lam, n, k = load_tabulated_nk(filename)
    table = TabulatedNK(filename)
    with torch.no_grad():
        n_hat, k_hat = table(torch.from_numpy(lam))
    assert np.abs(n_hat.numpy() - n).max() < 1e-12
    positive = k > 0
    assert np.abs(k_hat.numpy()[positive] / k[positive] - 1.0).max() < 1e-12


def test_tabulated_nk_is_positive_and_differentiable() -> None:
    """k > 0 (log 공간 스플라인)이고 λ로 미분가능해야 한다."""
    table = TabulatedNK("Si_nk_Aspnes.yml")
    lam = torch.linspace(284.0, 793.0, 128, dtype=DTYPE, requires_grad=True)
    n, k = table(lam)
    assert (k > 0).all()
    (n.sum() + k.sum()).backward()
    assert lam.grad is not None and torch.isfinite(lam.grad).all()


def test_energy_axis_spline_recovers_e1_peak_that_coarse_table_clips() -> None:
    """거친 표 + λ축 선형 보간이 c-Si E1 봉우리를 깎는다는 사실을 고정한다.

    E1 임계점(≈3.4 eV)에서 Aspnes 표는 격자점을 갖지만, 거친 19점 표는 λ축 선형
    보간이라 봉우리가 낮아진다. 이 차이가 캘리브레이션 개선분 중 최대 항목이다.
    """
    lam = torch.linspace(284.0, 793.0, 226, dtype=DTYPE)
    n_new, _ = TabulatedNK("Si_nk_Aspnes.yml")(lam)
    n_old, _ = CoarseTableNK()(lam)
    assert float(n_new.max()) > float(n_old.max()) + 0.2
    # 봉우리 위치는 E1 근방이어야 한다 (문헌 3.40 eV).
    peak_energy = HC_EV_NM / float(lam[int(n_new.argmax())])
    assert 3.3 < peak_energy < 3.55


def test_coarse_table_nk_matches_numpy_interpolation() -> None:
    """대조군은 `si_nk`(np.interp)와 **정확히** 같아야 한다 (분리 측정의 전제)."""
    lam_nm, n_tab, k_tab = si_nk_coarse_table()
    lam = torch.linspace(284.0, 793.0, 226, dtype=DTYPE)
    with torch.no_grad():
        n_got, k_got = CoarseTableNK()(lam)
    assert np.allclose(n_got.numpy(), np.interp(lam.numpy(), lam_nm, n_tab), atol=1e-12)
    assert np.allclose(k_got.numpy(), np.interp(lam.numpy(), lam_nm, k_tab), rtol=1e-9)
