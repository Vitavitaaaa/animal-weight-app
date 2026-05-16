from backend.models.classifier import get_classifier
import torch


def test_classifier_output_shape():
    model = get_classifier()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == (2, 1)