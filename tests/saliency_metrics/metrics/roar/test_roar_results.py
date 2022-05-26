import mmcv
import numpy as np

from saliency_metrics.metrics import RoarResults


def test_average_and_dump(tmp_path):
    trials = [{0.1: [0.9, 0.91, 0.92]}, {0.2: [0.8, 0.81, 0.82]}]
    expected_trial_avg = {0.1: [0.91, np.std([0.9, 0.91, 0.92])], 0.2: [0.81, np.std([0.8, 0.81, 0.82])]}

    roar_result = RoarResults()
    roar_result.add_single_result(trials[0])
    roar_result.add_single_result(trials[1])

    file_path = str(tmp_path / "roar_result.json")
    roar_result.average_and_dump(file_path)

    avg_result = mmcv.load(file_path)
    trial_1_avg = avg_result["0.1"]
    trial_2_avg = avg_result["0.2"]
    print(avg_result)
    assert trial_1_avg == expected_trial_avg[0.1]
    assert trial_2_avg == expected_trial_avg[0.2]
