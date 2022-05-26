import os.path as osp
from typing import Dict, List, Tuple

import mmcv
import numpy as np


class RoarResults:
    """Helper class to record the ROAR training results.

    .. code-block:: python

        from saliency_metrics.metrics import RoarResults

        roar_results = RoarResults()
        # Two top_fractions
        # At each top_fraction, there are three trials.
        roar_results.add_single_result({0.1: [0.9, 0.91, 0.92]})
        roar_results.add_single_result({0.2: [0.8, 0.81, 0.82]})

        # Average the accuracies across the trials and dump the result.
        roar_results.average_and_dump("roar_avg_result.json")

        # The dump file should be like this, note that the keys are converted to str.
        # For each top_fraction, mean and std of the accuracies are saved.
        # {"0.1": [0.91, 0.008164965809277268], "0.2": [0.81, 0.008164965809277223]}
    """

    def __init__(self) -> None:
        self._acc_dict: Dict[float, List[float]] = dict()

    def add_single_result(self, single_result: Dict[float, List[float]]) -> None:
        """Add a single result for a specific ``top_fraction``.

        Args:
            single_result: A dictionary, where the key is ``top_fraction`` (``float``), and value is a list of
                accuracies (``float``) of the models, which are repeatedly trained on the perturbed datasets
                parametrized by ``top_fraction``.

        Returns:
            None
        """
        self._acc_dict.update(single_result)

    def average(self) -> Dict[float, Tuple[float, float]]:
        """Average the accuracies.

        Returns:
            A dictionary where each key is the ``top_fraction``, and the corresponding value is a tuple of
            ``(mean_accuracy, std_accuracy)`` representing the mean and standard deviation of the accuracies.
        """
        # explicitly convert the values to python built-in floats for the sake of serialization
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in self._acc_dict.items()}

    def average_and_dump(self, file_path: str) -> None:
        """Average the accuracies and save the averaged result to a file.

        Args:
            file_path: Path to the dumped file. The extension can be ``".json", ".yaml", ".pickle"``.

        .. note::
            The keys in the averaged result are ``float``s. However, when being dumped to a JSON file, the ``float``
            keys will be converted to ``str``.
        """
        mmcv.mkdir_or_exist(osp.dirname(file_path))
        avg_result = self.average()
        mmcv.dump(avg_result, file=file_path)
