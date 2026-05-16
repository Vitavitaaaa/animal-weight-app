import torch
from backend.utils.inference import run_pipeline


class DummyClassifier:
    def __call__(self, x):
        return torch.tensor([[-10.0]])


def test_run_pipeline_no_cow(mocker):

    mocker.patch(
        "backend.utils.inference.classifier_model",
        DummyClassifier()
    )

    mocker.patch(
        "backend.utils.inference.load_image",
        return_value=None
    )

    mocker.patch(
        "backend.utils.inference.preprocess_classifier",
        return_value=torch.randn(1, 3, 224, 224)
    )

    result = run_pipeline("test.jpg")

    assert result["is_cow"] is False