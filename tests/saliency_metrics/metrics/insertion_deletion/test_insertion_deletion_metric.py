import pytest
import torch

from saliency_metrics.metrics.insertion_deletion import InsertionDeletion


@pytest.fixture
def dummy_img_and_smap():
    # three-channel image
    img = torch.linspace(0, 1, 3072, dtype=torch.float32).reshape(3, 32, 32)
    # saliency map
    smap = torch.linspace(0, 1, 1024, dtype=torch.float32).reshape(32, 32)
    yield img, smap


@pytest.mark.parametrize("perturb_step_size", [10, 1025])
def test_evaluate(dummy_img_and_smap, perturb_step_size):
    img, smap = dummy_img_and_smap
    target = 3
    classifier_config = dict(type="torchvision.resnet18", num_classes=10, pretrained=False)
    ins_del = InsertionDeletion(classifier_config, forward_batch_size=32, perturb_step_size=10, summarized=False)
