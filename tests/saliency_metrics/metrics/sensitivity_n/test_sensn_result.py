import mmcv
import numpy as np
import pytest

from saliency_metrics.metrics.sensitivity_n import SensitivityNResult


@pytest.mark.parametrize("summarized", [False, True])
def test_result_and_dump(summarized):
    id_result = SensitivityNResult(summarized=summarized)
    trial = [
        {"n": 1, "correlation": 0.1},
        {"n": 1, "correlation": 0.2},
        {"n": 2, "correlation": 0.1},
        {"n": 2, "correlation": 0.2},
    ]
    id_result.add_single_result(trial[0])
    id_result.add_single_result(trial[1])
    id_result.add_single_result(trial[2])
    id_result.add_single_result(trial[3])

    file_path = r"sensitivity_n\results.json"
    id_result.dump(file_path)
    result = mmcv.load(file_path)
    if summarized:
        expected_output = [
            {"n": 1, "mean_correlation": np.mean([0.1, 0.2]), "std_correlation": np.std([0.1, 0.2])},
            {"n": 2, "mean_correlation": np.mean([0.1, 0.2]), "std_correlation": np.std([0.1, 0.2])},
        ]
        assert result == expected_output
    else:
        assert result == [{"n": 1, "correlation": [0.1, 0.2]}, {"n": 2, "correlation": [0.1, 0.2]}]
