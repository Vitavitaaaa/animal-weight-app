import torch

from backend.models.segmentation import (
    UNet,
    UNetEncoder
)


def test_unet_output_shape():

    model = UNet()

    x = torch.randn(2, 3, 256, 256)

    out = model(x)

    assert out.shape == (2, 4, 256, 256)


def test_unet_encoder_output_shape():

    unet = UNet()

    encoder = UNetEncoder(unet)

    x = torch.randn(2, 3, 256, 256)

    out = encoder(x)

    assert out.shape[0] == 2
    assert out.shape[1] == 128