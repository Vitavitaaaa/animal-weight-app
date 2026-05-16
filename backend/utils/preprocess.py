from PIL import Image

from backend.utils.transforms import (
    classifier_transform,
    segmentation_transform,
    regression_transform
)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


# ================= CLASSIFIER =================

def preprocess_classifier(image):
    tensor = classifier_transform(image)
    return tensor.unsqueeze(0)


# ================= SEGMENTATION =================

def preprocess_segmentation(image):
    tensor = segmentation_transform(image)
    return tensor.unsqueeze(0)


# ================= REGRESSION =================

def preprocess_regression(image):
    tensor = regression_transform(image)
    return tensor.unsqueeze(0)