import numpy as np
import pytest
import torch

from saliency_metrics.metrics import RoarPerturbation, build_perturbation


@pytest.fixture
def dummy_img_and_smap():
    # two single-channel images, the first one has channel mean of 5.0, and the second one has channel mean of 15.0
    img_1 = torch.arange(1, 10, dtype=torch.float32).reshape(1, 3, 3)
    img_2 = torch.arange(11, 20, dtype=torch.float32).reshape(1, 3, 3)
    img = torch.stack([img_1, img_2], dim=0)
    # saliency maps
    smap_1 = np.arange(1, 10, dtype=float).reshape(3, 3)
    smap_2 = np.arange(11, 20, dtype=float).reshape(3, 3)
    smap = np.stack([smap_1, smap_2], axis=0)
    yield img, smap


def build_from_registry():
    cfg = dict(type="RoarPerturbation", top_fraction=0.5)
    ptb = build_perturbation(cfg)
    assert isinstance(ptb, RoarPerturbation)


def test_init(dummy_img_and_smap):
    for invalid_fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="top_fraction should"):
            _ = RoarPerturbation(top_fraction=invalid_fraction)

    top_fraction = 0.1

    ptb_1 = RoarPerturbation(top_fraction, mean=None)
    assert ptb_1.quantile == 0.9
    assert not ptb_1._has_mean
    with pytest.raises(AttributeError):
        ptb_1.get_buffer("_mean")

    ptb_2 = RoarPerturbation(top_fraction, mean=[0.0, 0.0, 0.0])
    assert ptb_2.quantile == 0.9
    assert ptb_2._has_mean
    torch.testing.assert_allclose(ptb_2.get_buffer("_mean"), torch.zeros((3,), dtype=torch.float32))


@pytest.mark.parametrize("top_fraction", [0.0, 0.2, 1.0])
@pytest.mark.parametrize(
    "mean",
    [
        (100,),
        None,
    ],
)
def test_perturb_and_forward(dummy_img_and_smap, top_fraction, mean):
    img, smap = dummy_img_and_smap

    ptb = RoarPerturbation(top_fraction, mean)

    if mean is not None:
        if top_fraction == 1.0:
            expected_ptb_out = torch.full_like(img, 100.0)
        elif top_fraction == 0.0:
            expected_ptb_out = img
        else:
            out = img.clone()
            out[:, :, 2, 1:] = 100
            expected_ptb_out = out
    else:
        if top_fraction == 1.0:
            expected_ptb_out = torch.stack(
                [torch.full((1, 3, 3), 5.0, dtype=torch.float32), torch.full((1, 3, 3), 15.0, dtype=torch.float32)],
                dim=0,
            )
        elif top_fraction == 0.0:
            expected_ptb_out = img
        else:
            out = img.clone()
            out[0, :, 2, 1:] = 5.0
            out[1, :, 2, 1:] = 15.0
            expected_ptb_out = out

    expected_forward_out = expected_ptb_out
    torch.testing.assert_allclose(ptb.perturb(img, smap), expected_ptb_out)
    torch.testing.assert_allclose(ptb.forward(img, smap), expected_forward_out)
