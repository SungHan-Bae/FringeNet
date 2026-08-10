"""모델 단위 테스트 — 데이터 파일 없이 전부 돈다.

계약(src/models/__init__.py): 모든 모델은 (B, 226) 스펙트럼 -> (B, 4) 두께[nm].
baseline MLP에 대해 shape 계약, 출력 bound, 미분 가능성, 시드 재현성, 팩토리를 본다.
"""

from __future__ import annotations

import pytest
import torch

from src.models import CNN1D, MLP, ThicknessBound, WinnerSkipMLP, build_model
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
# 블록 구성 — Linear -> Norm -> Activation -> Dropout 순서, norm/residual 옵션
# ---------------------------------------------------------------------------
def test_block_order_is_linear_norm_activation_dropout() -> None:
    model = MLP(hidden_dims=(8,), norm="batchnorm", activation="gelu", dropout=0.1)
    kinds = [type(m) for m in model.net[0].body]
    assert kinds == [torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.GELU, torch.nn.Dropout]


def test_norm_option_selects_layer_type() -> None:
    bn = MLP(hidden_dims=(8,), norm="batchnorm")
    ln = MLP(hidden_dims=(8,), norm="layernorm")
    none = MLP(hidden_dims=(8,), norm="none")
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in bn.modules())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in bn.modules())
    assert any(isinstance(m, torch.nn.LayerNorm) for m in ln.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in ln.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm1d | torch.nn.LayerNorm) for m in none.modules())


def test_residual_uses_projection_only_when_widths_differ() -> None:
    model = MLP(hidden_dims=(32, 32), residual=True)
    assert isinstance(model.net[0].skip, torch.nn.Linear)  # 226 -> 32: projection
    assert isinstance(model.net[1].skip, torch.nn.Identity)  # 32 -> 32: identity
    assert model(_batch()).shape == (B, 4)
    no_residual = MLP(hidden_dims=(32, 32), residual=False)
    assert no_residual.net[0].skip is None


# ---------------------------------------------------------------------------
# 미분 가능성 · 재현성
# ---------------------------------------------------------------------------
def test_gradients_flow_to_every_parameter() -> None:
    set_seed(0)
    model = MLP(hidden_dims=(32, 16), residual=True)  # projection skip 파라미터까지 확인
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
        {"activation": "tanh"},
        {"norm": "instancenorm"},
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
# CNN1D — shape 계약, 다중 스케일, 채널 셔플 대조군, 파라미터 매칭
# ---------------------------------------------------------------------------
_SMALL_CNN = {"channels": (8, 16), "strides": (1, 2)}


def test_cnn_maps_spectrum_batch_to_four_thicknesses() -> None:
    model = CNN1D(**_SMALL_CNN)
    out = model(_batch())
    assert out.shape == (B, 4)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_cnn_rejects_wrong_shapes() -> None:
    model = CNN1D(**_SMALL_CNN)
    with pytest.raises(ValueError):
        model(torch.zeros(4, 225))  # 채널 수 불일치
    with pytest.raises(ValueError):
        model(torch.zeros(4, 1, 226))  # 3-D 입력은 forward 안에서 스스로 만든다 (계약)


def test_cnn_output_bound_flag_controls_physical_range() -> None:
    bounded = CNN1D(**_SMALL_CNN, output_bound=True)
    unbounded = CNN1D(**_SMALL_CNN, output_bound=False)
    with torch.no_grad():
        bounded.head.bias.fill_(1e4)
        unbounded.head.bias.fill_(1e4)
    x = _batch()
    out_bounded = bounded(x)
    assert (out_bounded >= 10.0).all()
    assert (out_bounded <= 300.0).all()
    assert (unbounded(x) > 300.0).any()


def test_cnn_unbounded_head_bias_initializes_at_range_center() -> None:
    # MLP와 같은 규약 — bound 없이도 학습 시작점이 "범위 중앙 예측"이어야 공정 (mlp.py 참조)
    model = CNN1D(**_SMALL_CNN, output_bound=False)
    assert torch.allclose(model.head.bias, torch.full((4,), 155.0))


def test_cnn_multiscale_splits_channels_across_kernel_branches() -> None:
    model = CNN1D(channels=(12, 24), strides=(1, 2), kernel_sizes=(3, 7, 15))
    first = model.blocks[0]
    assert len(first.branches) == 3
    assert all(conv.out_channels == 4 for conv in first.branches)
    assert {conv.kernel_size[0] for conv in first.branches} == {3, 7, 15}
    assert model(_batch()).shape == (B, 4)


def test_cnn_multiscale_rejects_indivisible_channels() -> None:
    with pytest.raises(ValueError):
        CNN1D(channels=(10,), strides=(1,), kernel_sizes=(3, 7, 15))  # 10 % 3 != 0


def test_cnn_channel_shuffle_is_fixed_permutation_and_reroutes_input() -> None:
    set_seed(0)
    model = CNN1D(**_SMALL_CNN, channel_shuffle_seed=7)
    model.eval()
    perm = model.channel_perm
    assert perm is not None
    # 유효한 순열이고, 같은 시드면 인스턴스가 달라도 같은 순열 (재현성)
    assert torch.equal(torch.sort(perm).values, torch.arange(226))
    assert torch.equal(perm, CNN1D(**_SMALL_CNN, channel_shuffle_seed=7).channel_perm)
    assert not torch.equal(perm, torch.arange(226))  # identity면 대조군이 아니다
    # 셔플 모델(x) == 같은 가중치로 순열을 끄고 미리 섞은 입력을 준 것
    x = _batch()
    y_shuffled = model(x)
    x_pre_permuted = x[:, perm]
    model.channel_perm.copy_(torch.arange(226))
    assert torch.allclose(y_shuffled, model(x_pre_permuted))


def test_cnn_no_shuffle_by_default() -> None:
    assert CNN1D(**_SMALL_CNN).channel_perm is None


def test_cnn_flatten_head_preserves_position_dimension() -> None:
    # gap: 헤드 입력 = C_last. flatten: C_last * W_last (226 -> stride 2 한 번 -> 113)
    gap = CNN1D(channels=(8, 16), strides=(1, 2), head="gap")
    flat = CNN1D(channels=(8, 16), strides=(1, 2), head="flatten")
    assert gap.head.in_features == 16
    assert flat.head.in_features == 16 * 113
    assert flat(_batch()).shape == (B, 4)


def test_cnn_flatten_head_bias_also_initializes_at_range_center() -> None:
    model = CNN1D(**_SMALL_CNN, head="flatten", output_bound=False)
    assert torch.allclose(model.head.bias, torch.full((4,), 155.0))


def test_cnn_dilation_widens_receptive_field_without_changing_params_or_length() -> None:
    base = CNN1D(channels=(8, 16), strides=(1, 1), head="flatten")
    dilated = CNN1D(channels=(8, 16), strides=(1, 1), dilations=(1, 4), head="flatten")
    # dilation은 파라미터 수를 바꾸지 않는다 (수용영역만) — 통제 변인 유지의 근거
    n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n_params(base) == n_params(dilated)
    # 출력 길이도 보존 (padding = d * (k // 2)) -> flatten 헤드 크기 동일
    assert dilated.head.in_features == base.head.in_features == 16 * 226
    out = dilated(_batch())
    assert out.shape == (B, 4)
    # dilation이 실제로 적용됐는지
    assert dilated.blocks[1].branches[0].dilation == (4,)


def test_cnn_dilated_gradients_flow() -> None:
    set_seed(0)
    model = CNN1D(channels=(8, 16), strides=(1, 2), dilations=(1, 2), kernel_sizes=(3, 7))
    loss = torch.nn.functional.l1_loss(model(_batch()), torch.full((B, 4), 155.0))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cnn_flatten_default_still_matches_mlp_parameter_count() -> None:
    # flatten 헤드(662,020)도 baseline MLP(646,660) 대비 ±10% 안이어야 비교가 성립한다
    mlp_params = sum(p.numel() for p in MLP(hidden_dims=(512, 512, 512), dropout=0.0).parameters())
    cnn_params = sum(p.numel() for p in CNN1D(head="flatten").parameters())
    assert abs(cnn_params - mlp_params) / mlp_params < 0.10, (cnn_params, mlp_params)


def test_cnn_gradients_flow_to_every_parameter() -> None:
    set_seed(0)
    model = CNN1D(channels=(8, 16), strides=(1, 2), kernel_sizes=(3, 7))
    loss = torch.nn.functional.l1_loss(model(_batch()), torch.full((B, 4), 155.0))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    total = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert total > 0.0


def test_cnn_default_matches_mlp_baseline_parameter_count() -> None:
    # 변인 통제의 핵심: 기본 CNN은 baseline MLP 512x3과 파라미터 수가 ±10% 안이어야
    # "성능 차 = 구조 bias 기여"라고 말할 수 있다 (Task 5 비교의 전제).
    mlp_params = sum(p.numel() for p in MLP(hidden_dims=(512, 512, 512), dropout=0.0).parameters())
    cnn_params = sum(p.numel() for p in CNN1D().parameters())
    assert abs(cnn_params - mlp_params) / mlp_params < 0.10, (cnn_params, mlp_params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_sizes": (4,)},  # 짝수 커널 — 분기 길이 정렬 불가
        {"kernel_sizes": ()},
        {"channels": (8,), "strides": (1, 2)},  # 길이 불일치
        {"channels": ()},
        {"channels": (8, -1), "strides": (1, 1)},
        {"strides": (0, 1), "channels": (8, 8)},
        {"channels": (8, 8), "strides": (1, 1), "dilations": (1,)},  # 길이 불일치
        {"channels": (8, 8), "strides": (1, 1), "dilations": (1, 0)},
        {"head": "maxpool"},
        {"activation": "tanh"},
        {"norm": "instancenorm"},
        {"dropout": 1.0},
        {"d_min": 300.0, "d_max": 10.0},
    ],
)
def test_cnn_invalid_hyperparameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CNN1D(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------
def test_build_model_from_config_dict() -> None:
    config = {"name": "mlp", "hidden_dims": [32, 16], "output_bound": False}
    model = build_model(config)
    assert isinstance(model, MLP)
    assert model(_batch()).shape == (B, 4)
    assert config["name"] == "mlp"  # 원본 dict를 변경하지 않는다


def test_build_cnn_from_config_dict() -> None:
    config = {"name": "cnn", "channels": [8, 16], "strides": [1, 2], "output_bound": False}
    model = build_model(config)
    assert isinstance(model, CNN1D)
    assert model(_batch()).shape == (B, 4)


@pytest.mark.parametrize("config", [{}, {"name": "transformer"}])
def test_build_model_rejects_bad_config(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_model(config)


# ---------------------------------------------------------------------------
# WinnerSkipMLP — 1등 솔루션 충실 재현 (strong_baseline)
# ---------------------------------------------------------------------------
_TINY_WINNER = {"up_dims": (16, 32, 48), "head_dim": 8}


def test_winner_skip_mlp_maps_spectrum_batch_to_four_thicknesses() -> None:
    model = WinnerSkipMLP(**_TINY_WINNER)
    out = model(_batch())
    assert out.shape == (B, 4)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_winner_skip_mlp_rejects_wrong_shapes() -> None:
    model = WinnerSkipMLP(**_TINY_WINNER)
    with pytest.raises(ValueError):
        model(torch.zeros(4, 225))
    with pytest.raises(ValueError):
        model(torch.zeros(4, 1, 226))


def test_winner_skip_mlp_param_count_matches_original() -> None:
    # 기본 인자 = 원본 SkipConnectionModel(226->2000->4000->7000->10000->...->300->4).
    # 이 수가 어긋나면 폭·블록 구성 어딘가가 원본과 다른 것이다 (재현 주장의 근거 고정).
    model = WinnerSkipMLP()
    assert sum(p.numel() for p in model.parameters()) == 213_208_104


def test_winner_skip_mlp_block_structure_matches_original() -> None:
    # 블록 = Linear -> GELU(tanh 근사) -> BatchNorm1d, down 입구 LayerNorm(eps 1e-5),
    # dropout 없음(원본이 정의만 하고 미호출), head는 bare Linear (bound 없음)
    model = WinnerSkipMLP(**_TINY_WINNER)
    for block in [*model.ups, *model.downs]:
        kinds = [type(m) for m in block]
        assert kinds == [torch.nn.Linear, torch.nn.GELU, torch.nn.BatchNorm1d]
        assert block[1].approximate == "tanh"
    assert all(isinstance(n, torch.nn.LayerNorm) and n.eps == 1e-5 for n in model.norms)
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert not any(isinstance(m, ThicknessBound) for m in model.modules())


def test_winner_skip_mlp_forward_matches_original_expression() -> None:
    # 원본 forward 전개식과 대조 — skip 위치(마지막 down 제외)와 LayerNorm 위치(down 입구,
    # skip이 더해진 뒤의 값에 적용)를 고정한다. 원본: down_i(ln(skip_{i-1})) + up_{k-i}
    set_seed(0)
    model = WinnerSkipMLP(**_TINY_WINNER).eval()
    x = _batch()
    up1 = model.ups[0](x)
    up2 = model.ups[1](up1)
    up3 = model.ups[2](up2)
    skip1 = model.downs[0](model.norms[0](up3)) + up2
    skip2 = model.downs[1](model.norms[1](skip1)) + up1
    down3 = model.downs[2](model.norms[2](skip2))  # 마지막 down은 skip 없음
    expected = model.head(down3)
    assert torch.allclose(model(x), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"up_dims": (16,)},  # skip이 성립하려면 up 중간 출력이 필요 — 2개 이상
        {"up_dims": (16, 0)},
        {"up_dims": (16, 32), "head_dim": 0},
    ],
)
def test_winner_skip_mlp_invalid_dims_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        WinnerSkipMLP(**kwargs)  # type: ignore[arg-type]


def test_build_winner_skip_mlp_from_config_dict() -> None:
    config = {"name": "winner_skip_mlp", "up_dims": [16, 32], "head_dim": 8}
    model = build_model(config)
    assert isinstance(model, WinnerSkipMLP)
    assert model(_batch()).shape == (B, 4)
