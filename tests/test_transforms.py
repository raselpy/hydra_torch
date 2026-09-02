import torch

from hydra_torch.data.transforms import repeat_channels


def test_repeat_channels_expands_single_channel_to_three():
    img = torch.randn(1, 28, 28)
    result = repeat_channels(img)
    assert result.shape == (3, 28, 28)
    # each of the 3 channels should be identical to the original
    assert torch.equal(result[0], img[0])
    assert torch.equal(result[1], img[0])
    assert torch.equal(result[2], img[0])


def test_repeat_channels_leaves_three_channel_image_unchanged():
    img = torch.randn(3, 28, 28)
    result = repeat_channels(img)
    assert torch.equal(result, img)


def test_repeat_channels_leaves_non_3d_tensor_unchanged():
    img = torch.randn(4, 3, 28, 28)  # batched input, ndim == 4
    result = repeat_channels(img)
    assert torch.equal(result, img)