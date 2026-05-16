import torch
import torchvision.models as models


def get_classifier():

    model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = torch.nn.Linear(in_features, 1)

    return model