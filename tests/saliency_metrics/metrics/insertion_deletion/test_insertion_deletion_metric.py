import pytest
import torch

from saliency_metrics.metrics.insertion_deletion.custom_classifier import TestNet  # noqa:F401
from saliency_metrics.metrics.insertion_deletion.insertion_deletion_metric import InsertionDeletion


@pytest.fixture
def dummy_img_and_smap():
    # three-channel image
    img = torch.linspace(0, 1, 3072, dtype=torch.float32).reshape(3, 32, 32)
    # saliency map
    smap = torch.linspace(0, 1, 1024, dtype=torch.float32).reshape(32, 32)
    yield img, smap


@pytest.mark.parametrize("perturb_step_size", [250, 1025])
@pytest.mark.parametrize("forward_batch_size", [0, 32])
def test_evaluate(dummy_img_and_smap, perturb_step_size, forward_batch_size):
    img, smap = dummy_img_and_smap
    target = 3
    classifier_cfg = dict(type="custom.TestNet")
    ins_del = InsertionDeletion(
        classifier_cfg, forward_batch_size=forward_batch_size, perturb_step_size=perturb_step_size, summarized=False
    )
    if perturb_step_size == 1025:
        with pytest.raises(ValueError, match="perturb_step_size should be"):
            _ = ins_del.evaluate(img, smap, target, "user/somepath")
    else:
        if forward_batch_size == 0:
            with pytest.raises(ValueError, match="forward_batch_size should be"):
                _ = ins_del.evaluate(img, smap, target, "user/somepath")
        else:
            single_result = ins_del.evaluate(img, smap, target, "user/somepath")
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
            assert single_result["ins_auc"] == expected_result["ins_auc"]
            assert single_result["del_auc"] == expected_result["del_auc"]
            assert len(single_result["del_scores"]) == len(expected_result["del_scores"])
            assert len(single_result["ins_scores"]) == len(expected_result["ins_scores"])
            assert single_result["img_path"] == expected_result["img_path"]
            ins_del.update(single_result)
            test_result = ins_del.get_result
            assert test_result.results[0]["ins_auc"] == expected_result["ins_auc"]
            assert test_result.results[0]["del_auc"] == expected_result["del_auc"]
            assert len(test_result.results[0]["del_scores"]) == len(expected_result["del_scores"])
            assert len(test_result.results[0]["ins_scores"]) == len(expected_result["ins_scores"])
            assert test_result.results[0]["img_path"] == expected_result["img_path"]
