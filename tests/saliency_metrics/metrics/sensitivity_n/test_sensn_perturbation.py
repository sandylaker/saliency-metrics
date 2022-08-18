import pytest
import torch

from saliency_metrics.metrics.sensitivity_n import SensitivityNPerturbation


# Parametrize img and smap with different shapes, num_masks and n
@pytest.mark.parametrize("shape", [[3, 15, 15], [3, 20, 20], [3, 30, 30]])
@pytest.mark.parametrize("num_masks", [5, 10, 15])
@pytest.mark.parametrize("n", [10, 50, 100])
def test_init(shape, num_masks, n):
    # three chanel image
    img = torch.rand(shape)
    # random saliency map with the same shape
    smap = torch.rand(shape[-2:])

    test = SensitivityNPerturbation(n=n, num_masks=num_masks)
    batched_sample, sum_attributions = test.perturb(img, smap)
    # expected shape (num_masks + 1, channels, height, width)
    assert list(batched_sample.size()) == [num_masks + 1] + shape
    # expected shape (num_masks,)
    assert sum_attributions.shape == (num_masks,)
