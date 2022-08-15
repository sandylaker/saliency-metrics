import numpy as np
import pytest
import torch
import torch.nn as nn

from saliency_metrics.metrics.insertion_deletion.insertion_deletion import InsertionDeletion
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
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 16 * 16 * 5)
        x = self.classifier(x)
        return x

    def initialize_weights(self) -> None:
        for i, x in enumerate(self.parameters()):
            prng = np.random.RandomState(i)
            new_parameters = prng.uniform(0, 0.02, x.shape)
            x.data = torch.tensor(new_parameters, dtype=torch.float32)


@pytest.fixture
def dummy_img_and_smap():
    # three-channel image
    img = torch.linspace(0, 1, 3072, dtype=torch.float32).reshape(1, 3, 32, 32)
    # saliency map
    smap = torch.linspace(0, 1, 1024, dtype=torch.float32).reshape(32, 32)
    yield img, smap


@pytest.mark.parametrize("perturb_step_size", [0, 10])
@pytest.mark.parametrize("forward_batch_size", [0, 32])
def test_init(perturb_step_size, forward_batch_size):
    classifier_cfg = dict(type="custom.TestNet")
    if forward_batch_size == 0 or perturb_step_size == 0:
        with pytest.raises(ValueError, match="should be greater than zero"):
            ins_del = InsertionDeletion(
                classifier_cfg, forward_batch_size=forward_batch_size, perturb_step_size=perturb_step_size
            )
    else:
        ins_del = InsertionDeletion(
            classifier_cfg, forward_batch_size=forward_batch_size, perturb_step_size=perturb_step_size
        )
        assert ins_del.forward_batch_size == forward_batch_size
        assert ins_del.perturb_step_size == perturb_step_size


@pytest.mark.parametrize("perturb_step_size", [250, 1025])
@pytest.mark.parametrize("img_dimensions", [3, 4])
def test_evaluate(dummy_img_and_smap, perturb_step_size, img_dimensions):
    img, smap = dummy_img_and_smap
    target = 3
    classifier_cfg = dict(type="custom.TestNet")
    forward_batch_size = 32
    ins_del = InsertionDeletion(
        classifier_cfg, forward_batch_size=forward_batch_size, perturb_step_size=perturb_step_size, summarized=False
    )
    if perturb_step_size == 1025:
        with pytest.raises(ValueError, match="perturb_step_size should be"):
            _ = ins_del.evaluate(img, smap, target, img_path="user/somepath")
    else:
        if img_dimensions == 3:
            with pytest.raises(ValueError, match="img should be"):
                _ = ins_del.evaluate(torch.squeeze(img, 0), smap, target, img_path="user/somepath")
        else:
            single_result = ins_del.evaluate(img, smap, target, img_path="user/somepath")
            expected_result = {
                "del_scores": [
                    0.09772268682718277,
                    0.09906678646802902,
                    0.10010480135679245,
                    0.10033050924539566,
                    0.09997809678316116,
                ],
                "ins_scores": [
                    0.09874380379915237,
                    0.09874299168586731,
                    0.0987432524561882,
                    0.0987226590514183,
                    0.0987049862742424,
                ],
                "img_path": "user/somepath",
                "del_auc": 0.0796705037355423,
                "ins_auc": 0.07898665964603424,
            }
            assert single_result["ins_auc"] == pytest.approx(expected_result["ins_auc"], 1e-4)
            assert single_result["del_auc"] == pytest.approx(expected_result["del_auc"], 1e-4)
            assert len(single_result["del_scores"]) == len(expected_result["del_scores"])
            assert len(single_result["ins_scores"]) == len(expected_result["ins_scores"])
            assert single_result["img_path"] == expected_result["img_path"]
            ins_del.update(single_result)
            test_result = ins_del.get_result
            assert test_result.results[0]["ins_auc"] == pytest.approx(expected_result["ins_auc"], 1e-4)
            assert test_result.results[0]["del_auc"] == pytest.approx(expected_result["del_auc"], 1e-4)
            assert len(test_result.results[0]["del_scores"]) == len(expected_result["del_scores"])
            assert len(test_result.results[0]["ins_scores"]) == len(expected_result["ins_scores"])
            assert test_result.results[0]["img_path"] == expected_result["img_path"]
