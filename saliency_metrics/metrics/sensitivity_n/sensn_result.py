import os.path as osp
from collections import defaultdict
from typing import Dict, List

import mmcv
import numpy as np

from ..serializable_result import SerializableResult


class SensitivityNResult(SerializableResult):
    """Helper class to record the Sensitivity-N training results.

    Args:
        summarized: if True, compute the mean and std of Pearson correlation coefficients for each ``n``

    .. code-block:: python

        from saliency_metrics.metrics SensitivityNResult

        sensn_results = SensitivityNResult(summarized = True)
        #Each single_result is a dictionary with "n" and the corresponding "correlation" for a image
        id_result.add_single_result({"n": 1, "correlation": 0.1,})
        id_result.add_single_result({"n": 1, "correlation": 0.2,})

        #dump mean and std for each ``n``
        sensn_results.dump("sensn_avg_result.json")

        #The dump file should look like this:
        #{"n": 1, "mean_correlation":0.15, "std_correlation": 0.05}
    """

    def __init__(self, summarized: bool = True) -> None:
        self.summarized = summarized
        self._corr_dict: defaultdict[str, List[str]] = defaultdict(list)

    def dump(self, file_path: str) -> None:
        """

        Args:
            file_path: Path to the dumped file. The extension can be ``".json", ".yaml", ".pickle"``.

        .. note::
            The keys in the averaged result are ``float``s. However, when being dumped to a JSON file, the ``float``
            keys will be converted to ``str``.
        """
        result: List[Dict] = [{"n": n, "correlation": v} for n, v in self._corr_dict.items()]

        if self.summarized:
            summarized_result: List[Dict] = []
            for res in result:
                summarized_result.append(
                    {
                        "n": res["n"],
                        "mean_correlation": np.mean(res["correlation"]),
                        "std_correlation": np.std(res["correlation"]),
                    }
                )

            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(summarized_result, file_path)

        else:
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(result, file_path)

    def add_single_result(self, single_result: Dict) -> None:
        """Add a single result for a specific ``n``.

        Args:
            single_result: A dictionary, where the key is ``n`` (``int``), and ``correlation`` is a list of
                Pearson correlation coefficients (``float``) of the models.
        Returns:
            None
        """
        self._corr_dict[single_result["n"]].append(single_result["correlation"])
