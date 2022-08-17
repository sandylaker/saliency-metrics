from typing import Dict, Union

import numpy as np
import torch

from saliency_metrics.models import build_classifier, freeze_module
from ..build_metric import ReInferenceMetric
from ..serializable_result import SerializableResult
from .sensn_perturbation import SensitivityNPerturbation
from .sensn_result import SensitivityNResult


class SensitivityN(ReInferenceMetric):
    def __init__(
        self,
        classifier_cfg: Dict,
        log_n_max: int,
        log_n_ticks: float,
        summarized: bool = True,
        num_masks: int = 100,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.device = device
        self.classifier = build_classifier(classifier_cfg).to(self.device)
        # freeze the model and turn eval mode on
        freeze_module(self.classifier)

        self._result: SerializableResult = SensitivityNResult(summarized=summarized)
        self._num_masks = num_masks
        n_list = np.logspace(0, log_n_max, int(log_n_max / log_n_ticks), base=10.0, dtype=int)

        # to eliminate the duplicated elements caused by rounding
        self._n_list = np.unique(n_list)
        self._current_n_ind = 0
        self._perturbation = SensitivityNPerturbation(self._n_list[self._current_n_ind], num_masks=self._num_masks)

    @property
    def num_ns(self):
        return len(self._n_list)

    @property
    def current_n(self):
        return self._n_list[self._current_n_ind]

    def increment_n(self) -> None:
        self._current_n_ind += 1
        self._perturbation = SensitivityNPerturbation(self._n_list[self._current_n_ind], self._num_masks)

    def evaluate(self, img: torch.Tensor, smap: torch.Tensor, target: int) -> Dict:
        batched_sample, sum_attributions = self._perturbation.perturb(img, smap)

        with torch.no_grad():
            scores = torch.softmax(self.classifier(batched_sample), -1)[:, target]
        score_diffs = scores[:-1] - scores[-1]
        corrcoef = np.corrcoef(sum_attributions, score_diffs)[1, 0]
        single_result = {"n": self._n_list[self._current_n_ind], "correlation": corrcoef}
        return single_result

    def update(self, single_result: Dict) -> None:
        # update self._result
        self._result.add_single_result(single_result)
