from PIL import Image
import torch

from backend.utils.preprocess import (
    preprocess_classifier,
    preprocess_segmentation,
    preprocess_regression
)


def create_image():
    return Image.new("RGB", (500, 500))


def test_preprocess_classifier():

    image = create_image()

    tensor = preprocess_classifier(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_segmentation():

    image = create_image()

    tensor = preprocess_segmentation(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 256, 256)


def test_preprocess_regression():

    image = create_image()

    tensor = preprocess_regression(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)