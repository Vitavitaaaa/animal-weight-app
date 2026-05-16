from PIL import Image

from backend.utils.transforms import (
    classifier_transform,
    segmentation_transform,
    regression_transform
)


def create_image():
    return Image.new("RGB", (100, 100))


def test_classifier_transform():

    image = create_image()

    out = classifier_transform(image)

    assert out.shape == (3, 224, 224)


def test_segmentation_transform():

    image = create_image()

    out = segmentation_transform(image)

    assert out.shape == (3, 256, 256)


def test_regression_transform():

    image = create_image()

    out = regression_transform(image)

    assert out.shape == (3, 224, 224)