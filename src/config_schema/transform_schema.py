# src/config_schema/transform_schema.py
from pydantic.dataclasses import dataclass
from typing import List, Any
from omegaconf import MISSING


@dataclass
class TransformConfig:
    _target_: str = MISSING


@dataclass
class MNISTTransformSchema(TransformConfig):
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = MISSING
