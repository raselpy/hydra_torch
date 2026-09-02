from omegaconf import MISSING

from config_schema.data_module.transform_schema import (
    MNISTTransformSchema,
    TransformConfig,
)


def test_transform_config_target_defaults_to_missing():
    cfg = TransformConfig()
    assert cfg._target_ == MISSING


def test_mnist_transform_schema_has_expected_target():
    cfg = MNISTTransformSchema(transforms=[])
    assert cfg._target_ == "torchvision.transforms.Compose"


def test_mnist_transform_schema_is_a_transform_config():
    cfg = MNISTTransformSchema(transforms=[])
    assert isinstance(cfg, TransformConfig)


def test_mnist_transform_schema_stores_given_transforms():
    dummy_transforms = ["ToTensor", "Normalize"]
    cfg = MNISTTransformSchema(transforms=dummy_transforms)
    assert cfg.transforms == dummy_transforms