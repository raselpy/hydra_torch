from functools import partial

import torch
from torch import nn, optim

from hydra_torch.tasks import (
    CIFAR10ClassificationTrainingTask,
    MNISTClassificationTrainingTask,
    _ClassificationTask,
)


def _make_task(cls=_ClassificationTask, num_features=8, num_classes=10):
    model = nn.Linear(num_features, num_classes)
    optimizer = partial(optim.SGD, lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    task = cls(model=model, optimizer=optimizer, loss_function=loss_function)
    task.log = lambda *args, **kwargs: None  # avoid requiring a Trainer
    return task


def _make_batch(batch_size=4, num_features=8, num_classes=10):
    images = torch.randn(batch_size, num_features)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


def test_configure_optimizers_returns_bound_optimizer():
    task = _make_task()
    result = task.configure_optimizers()
    assert isinstance(result, optim.SGD)
    assert result.defaults["lr"] == 0.01


def test_forward_delegates_to_model():
    task = _make_task()
    images, _ = _make_batch()
    output = task(images)
    expected = task.model(images)
    assert torch.equal(output, expected)


def test_training_step_returns_scalar_loss():
    task = _make_task()
    batch = _make_batch()
    loss = task.training_step(batch, 0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_training_step_logs_expected_keys():
    task = _make_task()
    logged = {}
    task.log = lambda key, value, **kwargs: logged.setdefault(key, value)
    task.training_step(_make_batch(), 0)
    assert "train_loss" in logged
    assert "train_accuracy" in logged


def test_validation_step_logs_accuracy():
    task = _make_task()
    logged = {}
    task.log = lambda key, value, **kwargs: logged.setdefault(key, value)
    task.validation_step(_make_batch(), 0)
    assert "validation_accuracy" in logged


def test_test_step_logs_accuracy():
    task = _make_task()
    logged = {}
    task.log = lambda key, value, **kwargs: logged.setdefault(key, value)
    task.test_step(_make_batch(), 0)
    assert "test_accuracy" in logged


def test_mnist_and_cifar10_subclasses_instantiate():
    mnist_task = _make_task(cls=MNISTClassificationTrainingTask)
    cifar_task = _make_task(cls=CIFAR10ClassificationTrainingTask)
    assert isinstance(mnist_task, MNISTClassificationTrainingTask)
    assert isinstance(cifar_task, CIFAR10ClassificationTrainingTask)