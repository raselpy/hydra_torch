# src/hydra_torch/transforms.py
import torch


def repeat_channels(img: torch.Tensor) -> torch.Tensor:
    """Repeats a single-channel image tensor to 3 channels for ResNet compatibility."""
    if img.ndim == 3 and img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img
