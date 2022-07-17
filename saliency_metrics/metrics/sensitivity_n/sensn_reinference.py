from typing import Dict

import numpy as np
import torch
from sensn_perturbation import SensitivityNPerturbation
from sensn_results import SensitivityNResult

from saliency_metrics.metrics.build_metric import ReInferenceMetric
from saliency_metrics.metrics.serializable_result import SerializableResult
from saliency_metrics.models import freeze_module
from saliency_metrics.models.build_classifier import build_classifier


class SensitivityN(ReInferenceMetric):
    def __init__(
        self, classifier_cfg: Dict, log_n_max: int, log_n_ticks: float, summarized: bool = True, num_masks: int = 100
    ) -> None:
        self.classifier = build_classifier(classifier_cfg)
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

        score_diffs = []
        out = []
        with torch.no_grad():
            for batch in batched_sample:
                scores = torch.softmax(self.classifier(batch.unsqueeze(0)), -1)[:, target]
                out.append(scores.item())

        for i, _ in enumerate(out[0:-1]):
            score_diffs.append((out[:-1][i] - out[-1:][0]))

        corrcoef = np.corrcoef(sum_attributions, score_diffs)[1, 0]

        single_result = {"n": self._n_list[self._current_n_ind], "correlation": corrcoef}

        return single_result

    def update(self, single_result: Dict) -> None:
        # update self._result
        self._result.add_single_result(single_result)
