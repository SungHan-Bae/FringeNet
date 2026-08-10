"""모델 레지스트리·팩토리.

인터페이스 규약 — 모든 모델은 x (B, 226) float 스펙트럼을 받아 (B, 4) 두께[nm]를 낸다.
채널축이 필요한 모델(1D CNN 등)은 forward 안에서 스스로 (B, 1, 226)으로 바꾼다.
train.py가 모델 종류를 모른 채 config만으로 실험을 바꿀 수 있게 하기 위한 규약이다.

configs/*.yaml의 model 섹션이 그대로 ``build_model``의 입력이 된다::

    model:
      name: mlp
      hidden_dims: [512, 512, 256]
      output_bound: true
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from src.models.heads import ThicknessBound
from src.models.mlp import MLP

_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLP,
}

__all__ = ["MLP", "ThicknessBound", "build_model"]


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """model config dict로 모델을 만든다.

    Args:
        config: ``{"name": <레지스트리 키>, **모델 생성자 kwargs}``. 원본은 변경하지 않는다.

    Raises:
        ValueError: "name" 키가 없거나 레지스트리에 없는 이름인 경우.
    """
    kwargs = dict(config)
    name = kwargs.pop("name", None)
    if name is None:
        raise ValueError('model config에 "name" 키가 필요하다')
    model_cls = _REGISTRY.get(name)
    if model_cls is None:
        raise ValueError(f"모르는 모델 이름 {name!r} — 사용 가능: {sorted(_REGISTRY)}")
    return model_cls(**kwargs)
