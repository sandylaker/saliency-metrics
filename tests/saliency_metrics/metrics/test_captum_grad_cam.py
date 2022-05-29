import numpy as np
import pytest
import torch
import torch.nn as nn

from saliency_metrics.metrics import CaptumGradCAM


@pytest.fixture
def conv_classifier():

    model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=4, padding=1, stride=2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)
    )

    yield model


@pytest.mark.parametrize("as_ndarray", [True, False])
@pytest.mark.parametrize("interpolate", [True, False])
def test_captum_grad_cam(conv_classifier, as_ndarray, interpolate):
    gradcam = pytest.importorskip("captum", reason="Captum is not installed.")

    layer = "0"

    captum_grad_cam = CaptumGradCAM(conv_classifier, layer=layer)

    img = torch.randn(2, 3, 6, 6, requires_grad=True)
    target = torch.LongTensor([0, 1])
    smap = captum_grad_cam.attribute(img, target, as_ndarray=as_ndarray, interpolate=interpolate)
    if as_ndarray:
        assert isinstance(smap, np.ndarray)
    else:
        assert isinstance(smap, torch.Tensor)

    if interpolate:
        assert smap.shape == (2, 1, 6, 6)
    else:
        assert smap.shape == (2, 1, 3, 3)
