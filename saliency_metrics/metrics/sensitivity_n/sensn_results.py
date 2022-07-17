import os.path as osp
from typing import Dict, List

import mmcv
import numpy as np

from saliency_metrics.metrics.serializable_result import SerializableResult


class SensitivityNResult(SerializableResult):
    """Helper class to record the Sensitivity-n training results.

    Args:
        summarized: bool.
        num_masks: int
    """

    def __init__(self, summarized: bool = True, num_masks: int = 100) -> None:
        self.summarized = summarized
        self.num_masks = num_masks
        self.results: List[Dict[str, float]] = []

    def dump(self, file_path: str) -> None:
        """

        Args:
            file_path: Path to the dumped file. The extension can be ``".json", ".yaml", ".pickle"``.

        .. note::
            The keys in the averaged result are ``float``s. However, when being dumped to a JSON file, the ``float``
            keys will be converted to ``str``.
        """
        if self.summarized:
            n_list = []
            correlation_list = []
            for result in self.results:
                correlation_list.append(result["correlation"])
                summarized_result = {
                    "n": result["n"],
                    "mean_correlation": np.mean(correlation_list),
                    "std_correlation": np.std(correlation_list),
                }
            print(summarized_result)
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(summarized_result, file_path)
        else:
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(self.results, file_path)

    def add_single_result(self, single_result: Dict) -> None:
        """Add a single result for a specific ``top_fraction``.

        Args:
            single_result: A dictionary, where the key is ``top_fraction`` (``float``), and value is a list of
                accuracies (``float``) of the models, which are repeatedly trained on the perturbed datasets
                parametrized by ``top_fraction``.
        Returns:
            None
        """
        self.results.append(single_result)
