import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(pretrained=True, num_classes=2):
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
