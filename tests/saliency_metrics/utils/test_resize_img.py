import numpy as np
import torch

from saliency_metrics.utils import resize_img


def test_resize_img_ndarray():
    img = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
    img_out = resize_img(img, (10, 10))
    assert img_out.shape == (10, 10)


def test_resize_img_tensor():
    img = torch.randn(1, 1, 5, 5)
    img_out = resize_img(img, (10, 10))
    assert img_out.shape == (1, 1, 10, 10)
