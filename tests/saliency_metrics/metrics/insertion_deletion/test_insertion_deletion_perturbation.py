from typing import List

import numpy as np
import pytest
import torch

from saliency_metrics.metrics.insertion_deletion import ProgressivePerturbation


@pytest.fixture
def dummy_img_and_smap():
    # three-channel image
    img = torch.arange(1, 28, dtype=torch.float32).reshape(1, 3, 3, 3)
    # saliency map
    smap = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    yield img, smap


def test_init(dummy_img_and_smap):
    img, smap = dummy_img_and_smap
    num_pixels = torch.numel(smap)
    _, inds = torch.topk(smap.flatten(), num_pixels)
    row_inds, col_inds = (torch.tensor(x) for x in np.unravel_index(inds.numpy(), smap.size()))
    replace_tensor = torch.zeros_like(img)
    id_ptb = ProgressivePerturbation(img, replace_tensor, (row_inds, col_inds))
    perturbed_tensor = img.clone()
    perturbed_tensor[..., 1, 1] = 0
    torch.testing.assert_allclose(id_ptb._current_tensor, img)
    torch.testing.assert_allclose(id_ptb._replace_tensor, replace_tensor)
    assert id_ptb._num_pixels == 9
    id_ptb._perturb_by_inds(torch.tensor(1), torch.tensor(1))
    torch.testing.assert_allclose(id_ptb._current_tensor, perturbed_tensor)


@pytest.mark.parametrize("perturb_step_size", [3, 9])
@pytest.mark.parametrize("forward_batch_size", [2, 3])
def test_perturb(dummy_img_and_smap, forward_batch_size, perturb_step_size):
    img, smap = dummy_img_and_smap
    num_pixels = torch.numel(smap)
    _, inds = torch.topk(smap.flatten(), num_pixels)
    row_inds, col_inds = (torch.tensor(x) for x in np.unravel_index(inds.numpy(), smap.size()))
    replace_tensor = torch.zeros_like(img)
    id_ptb = ProgressivePerturbation(img, replace_tensor, (row_inds, col_inds))
    output_batches: List[torch.tensor] = []
    for batch in id_ptb.perturb(forward_batch_size=forward_batch_size, perturb_step_size=perturb_step_size):
        output_batches.append(batch)
    if perturb_step_size == 9:
        img = torch.zeros((1, 3, 3, 3))
        torch.testing.assert_allclose(output_batches[0], img)
    else:
        img1 = img.clone()
        img2 = img.clone()
        img3 = torch.zeros((1, 3, 3, 3))
        img1[..., 2, :] = 0
        img2[..., [1, 2], :] = 0
        if forward_batch_size == 2:
            torch.testing.assert_allclose(output_batches[0], torch.cat([img1, img2], dim=0))
            torch.testing.assert_allclose(output_batches[1], img3)
        else:
            torch.testing.assert_allclose(output_batches[0], torch.cat([img1, img2, img3], dim=0))
