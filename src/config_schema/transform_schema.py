# src/config_schema/transform_schema.py
from typing import Any

from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class TransformConfig:
    _target_: str = MISSING


@dataclass
class MNISTTransformSchema(TransformConfig):
    _target_: str = "torchvision.transforms.Compose"
    transforms: list[Any] = MISSING
