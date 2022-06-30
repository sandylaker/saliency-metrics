import numpy as np
import torch
import torch.nn as nn

from saliency_metrics.models import CUSTOM_CLASSIFIERS


@CUSTOM_CLASSIFIERS.register_module()
class TestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Linear(16 * 16 * 5, 10))
        self.set_weights_to_constants()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 16 * 16 * 5)
        x = self.classifier(x)
        return x

    def set_weights_to_constants(self) -> None:
        for i, x in enumerate(self.parameters()):
            prng = np.random.RandomState(i)
            new_paremeters = prng.uniform(0, 0.02, x.shape)
            x.data = torch.tensor(new_paremeters, dtype=torch.float32)
