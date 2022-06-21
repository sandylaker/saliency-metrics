import os.path as osp
from typing import Dict, List

import mmcv
import numpy as np

from saliency_metrics.metrics.serializable_result import SerializableResult


class InsertionDeletionResult(SerializableResult):
    def __init__(
        self,
        summarized: bool = False,
    ):

        self.summarized = summarized
        # TODO check results and read mypy
        self.results: List[Dict[str, float]] = []

    def add_single_result(self, single_result: Dict):
        self.results.append(single_result)

    def dump(self, file_path: str):
        if self.summarized:
            del_auc_array: List[float] = []
            ins_auc_array: List[float] = []
            for result in self.results:
                del_auc_array.append(result["del_auc"])
                ins_auc_array.append(result["ins_auc"])
                summarized_result = {
                    "mean_insertion_auc": np.mean(ins_auc_array),
                    "std_insertion_auc": np.std(ins_auc_array),
                    "mean_deletion_auc": np.mean(del_auc_array),
                    "std_deletion_auc": np.std(del_auc_array),
                    "num_samples": len(del_auc_array),
                }
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(summarized_result, file_path)
        else:
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(self.results, file_path)
