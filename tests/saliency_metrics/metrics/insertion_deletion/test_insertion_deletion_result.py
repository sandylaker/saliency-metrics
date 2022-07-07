import mmcv
import numpy as np
import pytest

from saliency_metrics.metrics.insertion_deletion import InsertionDeletionResult


@pytest.mark.parametrize("summarized", [False, True])
def test_result_and_dump(summarized, tmp_path):
    id_result = InsertionDeletionResult(summarized=summarized)
    trial = [
        {
            "del_scores": [0.8, 0.7, 0.6],
            "ins_scores": [0.1, 0.2, 0.3],
            "img_path": "path/to/img0.JPEG",
            "del_auc": 0.1,
            "ins_auc": 0.3,
        },
        {
            "del_scores": [0.7, 0.6, 0.5],
            "ins_scores": [0.2, 0.3, 0.4],
            "img_path": "path/to/img1.JPEG",
            "del_auc": 0.2,
            "ins_auc": 0.4,
        },
    ]
    id_result.add_single_result(trial[0])
    id_result.add_single_result(trial[1])
    file_path = str(tmp_path / "id_result.json")
    id_result.dump(file_path)
    result = mmcv.load(file_path)
    if summarized:
        expected_output = {
            "mean_ins_auc": np.mean([0.3, 0.4]),
            "std_ins_auc": np.std([0.3, 0.4]),
            "mean_del_auc": np.mean([0.1, 0.2]),
            "std_del_auc": np.std([0.1, 0.2]),
            "num_samples": 2,
        }
        assert result == expected_output
    else:
        assert result == trial
