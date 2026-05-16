import torch

from backend.models.segmentation import UNet
from backend.models.regression import TwoStreamRegressor


def test_regressor_output_shape():

    seg_model = UNet()

    model = TwoStreamRegressor(seg_model)

    x = torch.randn(2, 3, 224, 224)

    out = model(x)

    assert out.shape == (2,)