import os.path as osp
from typing import Dict, List

import mmcv
import numpy as np

from saliency_metrics.metrics.serializable_result import SerializableResult


class InsertionDeletionResult(SerializableResult):
    def __init__(
        self,
        summarized: bool = False,
    ) -> None:

        self.summarized = summarized
        self.results: List[Dict[str, float]] = []

    def add_single_result(self, single_result: Dict) -> None:
        self.results.append(single_result)

    def dump(self, file_path: str) -> None:
        if self.summarized:
            del_auc_scores_list: List[float] = []
            ins_auc_scores_list: List[float] = []
            for result in self.results:
                del_auc_scores_list.append(result["del_auc"])
                ins_auc_scores_list.append(result["ins_auc"])
            summarized_result = {
                "mean_insertion_auc": np.mean(ins_auc_scores_list),
                "std_insertion_auc": np.std(ins_auc_scores_list),
                "mean_deletion_auc": np.mean(del_auc_scores_list),
                "std_deletion_auc": np.std(del_auc_scores_list),
                "num_samples": len(del_auc_scores_list),
            }
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(summarized_result, file_path)
        else:
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(self.results, file_path)
