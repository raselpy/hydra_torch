# tests/test_serve.py
"""These tests deliberately avoid triggering the app's lifespan (which loads
a real checkpoint via _load_model()) — no trained checkpoint exists in CI,
since checkpoints/ is gitignored. Full end-to-end /predict testing against a
real checkpoint is a known gap, left for manual verification (see README) —
noted here explicitly rather than silently claiming more coverage than
actually exists.
"""

import os

from hydra_torch.serve import _CONFIG_PATH, CIFAR10_CLASSES


def test_config_path_resolves():
    resolved = os.path.abspath(_CONFIG_PATH)
    assert os.path.isdir(resolved)
    assert os.path.isfile(os.path.join(resolved, "config.yaml"))


def test_cifar10_classes_match_canonical_order():
    # torchvision.datasets.CIFAR10's label indices, in order — must match
    # exactly, or predictions will report the wrong class name even when
    # the model itself predicted the correct index.
    assert CIFAR10_CLASSES == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    assert len(CIFAR10_CLASSES) == 10
