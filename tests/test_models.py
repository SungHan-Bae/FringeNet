"""모델 단위 테스트 — 데이터 파일 없이 전부 돈다.

계약(src/models/__init__.py): 모든 모델은 (B, 226) 스펙트럼 -> (B, 4) 두께[nm].
baseline MLP에 대해 shape 계약, 출력 bound, 미분 가능성, 시드 재현성, 팩토리를 본다.
"""

from __future__ import annotations

import pytest
import torch

from src.models import MLP, ThicknessBound, build_model
from src.utils.seed import set_seed

B = 8


def _batch(n_channels: int = 226) -> torch.Tensor:
    """반사율 스케일([0, 1))의 재현 가능한 입력 배치."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(B, n_channels, generator=generator)


# ---------------------------------------------------------------------------
# shape 계약
# ---------------------------------------------------------------------------
def test_mlp_maps_spectrum_batch_to_four_thicknesses() -> None:
    model = MLP()
    out = model(_batch())
    assert out.shape == (B, 4)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_forward_rejects_wrong_shapes() -> None:
    model = MLP(hidden_dims=(8,))
    with pytest.raises(ValueError):
        model(torch.zeros(4, 225))  # 채널 수 불일치
    with pytest.raises(ValueError):
        model(torch.zeros(4, 1, 226))  # CNN용 3-D 입력은 받지 않는다


# ---------------------------------------------------------------------------
# 출력 bound
# ---------------------------------------------------------------------------
def test_thickness_bound_midpoint_and_range() -> None:
    bound = ThicknessBound(10.0, 300.0)
    # 로짓 0 -> sigmoid 0.5 -> 범위 정중앙
    assert torch.allclose(bound(torch.zeros(2, 4)), torch.full((2, 4), 155.0))
    out = bound(torch.linspace(-50.0, 50.0, 101).reshape(1, -1))
    assert (out >= 10.0).all()
    assert (out <= 300.0).all()


def test_output_bound_flag_controls_physical_range() -> None:
    bounded = MLP(hidden_dims=(16,), output_bound=True)
    unbounded = MLP(hidden_dims=(16,), output_bound=False)
    # 마지막 Linear의 bias를 크게 밀어, bound가 실제로 출력을 가두는지 본다.
    with torch.no_grad():
        bounded.net[-2].bias.fill_(1e4)  # [-1]은 ThicknessBound, [-2]가 Linear
        unbounded.net[-1].bias.fill_(1e4)
    x = _batch()
    out_bounded = bounded(x)
    assert (out_bounded >= 10.0).all()
    assert (out_bounded <= 300.0).all()
    assert (unbounded(x) > 300.0).any()  # 플래그 off면 정말 아무 제약이 없어야 한다


def test_unbounded_head_initializes_at_range_center() -> None:
    # hidden 없는 순수 선형 모델 + 영입력 -> 출력 = bias = (10+300)/2 정확히
    model = MLP(hidden_dims=(), output_bound=False)
    out = model(torch.zeros(3, 226))
    assert torch.allclose(out, torch.full((3, 4), 155.0))


# ---------------------------------------------------------------------------
# 미분 가능성 · 재현성
# ---------------------------------------------------------------------------
def test_gradients_flow_to_every_parameter() -> None:
    set_seed(0)
    model = MLP(hidden_dims=(32, 16))
    loss = torch.nn.functional.l1_loss(model(_batch()), torch.full((B, 4), 155.0))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    total = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert total > 0.0


def test_seeded_construction_is_reproducible() -> None:
    set_seed(123)
    first = MLP(hidden_dims=(32,))
    set_seed(123)
    second = MLP(hidden_dims=(32,))
    for p1, p2 in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# 하이퍼파라미터 검증
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"activation": "swish"},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"hidden_dims": (64, 0)},
        {"d_min": 300.0, "d_max": 10.0},
    ],
)
def test_invalid_hyperparameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MLP(**kwargs)  # type: ignore[arg-type]


def test_dropout_layers_present_only_when_positive() -> None:
    with_dropout = MLP(hidden_dims=(8,), dropout=0.3)
    without = MLP(hidden_dims=(8,), dropout=0.0)
    assert any(isinstance(m, torch.nn.Dropout) for m in with_dropout.modules())
    assert not any(isinstance(m, torch.nn.Dropout) for m in without.modules())


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------
def test_build_model_from_config_dict() -> None:
    config = {"name": "mlp", "hidden_dims": [32, 16], "output_bound": False}
    model = build_model(config)
    assert isinstance(model, MLP)
    assert model(_batch()).shape == (B, 4)
    assert config["name"] == "mlp"  # 원본 dict를 변경하지 않는다


@pytest.mark.parametrize("config", [{}, {"name": "transformer"}])
def test_build_model_rejects_bad_config(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_model(config)
