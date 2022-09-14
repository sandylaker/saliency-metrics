import os.path as osp
from typing import List

import mmcv
import numpy as np

from saliency_metrics.metrics.serializable_result import SerializableResult


class SanityCheckResult(SerializableResult):
    def __init__(self, summarized=True):
        self.summarized = summarized
        self.results: List = []

    def add_single_result(self, single_result):
        self.results.append(single_result)
        print("Appended: ", self.results)

    def dump(self, file_path):
        if self.summarized:
            ssim_list: List = []
            for result in self.results:
                ssim_list.append(result["ssim"])
            ssim_array = np.stack(ssim_list)
            summarized_result = {
                "mean_ssim": np.mean(ssim_array, axis=0),
                "std_ssim": np.std(ssim_array, axis=0),
                "num_samples": len(ssim_list),
            }
            print(summarized_result)
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(summarized_result, file_path)
        else:
            mmcv.mkdir_or_exist(osp.dirname(file_path))
            mmcv.dump(self.results, file_path)
