import torch

from backend.utils.inference import run_pipeline


class DummyClassifier:
    def __call__(self, x):
        return torch.tensor([[10.0]])


class DummySegmentation:
    def __call__(self, x):
        return torch.randn(1, 4, 256, 256)


class DummyRegression:
    def __call__(self, x):
        return torch.tensor([523.4])


def test_run_pipeline_success(mocker):

    mocker.patch(
        "backend.utils.inference.classifier_model",
        DummyClassifier()
    )

    mocker.patch(
        "backend.utils.inference.segmentation_model",
        DummySegmentation()
    )

    mocker.patch(
        "backend.utils.inference.regression_model",
        DummyRegression()
    )

    mocker.patch(
        "backend.utils.inference.load_image",
        return_value="fake_image"
    )

    mocker.patch(
        "backend.utils.inference.preprocess_classifier",
        return_value=torch.randn(1, 3, 224, 224)
    )

    mocker.patch(
        "backend.utils.inference.preprocess_segmentation",
        return_value=torch.randn(1, 3, 256, 256)
    )

    mocker.patch(
        "backend.utils.inference.preprocess_regression",
        return_value=torch.randn(1, 3, 224, 224)
    )

    result = run_pipeline("test.jpg")

    assert result["is_cow"] is True
    assert result["predicted_weight"] == 523.4