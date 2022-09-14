from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor

from saliency_metrics.metrics.build_metric import METRICS, ReInferenceMetric
from saliency_metrics.metrics.sanity_check.sanity_check_result import SanityCheckResult
from saliency_metrics.metrics.serializable_result import SerializableResult
from saliency_metrics.models.model_utils import freeze_module


class SanityCheck(ReInferenceMetric):
    def __init__(self, ori_model: torch.nn.Module, perturb_layers: List[str], summarized: bool = True) -> None:
        self._result = SanityCheckResult(summarized)
        self.ori_model = ori_model  # Original Model
        self._ori_state_dict = self.ori_model.state_dict()
        self.model = deepcopy(self.ori_model)  # Model which will be used for performing randomization on
        self.sd = self.model.state_dict()
        self.perturb_layers = perturb_layers
        self.all_layers: List = []
        for i, x in self.ori_model.named_parameters():
            self.all_layers.append(i)
        self._classifier_layers = self._filter_names(self.all_layers)

    def evaluate(self, image: Tensor, smap: np.ndarray, target: int, **kwargs) -> Dict:
        ssim_list: List = []
        for layer in self.perturb_layers:
            self._reload_ckpt()
            consecutive_perturb_layers: List = []
            for j in self._classifier_layers[self._classifier_layers.index(layer) :]:
                consecutive_perturb_layers.append(j)
            ssim_float = self._sanity_check_single(smap, target, consecutive_perturb_layers)
            ssim_list.append(ssim_float)
        return dict(ssim=ssim_list)

    def _sanity_check_single(self, smap: np.ndarray, target: int, consecutive_perturb_layers: List[str]) -> float:
        self._perturb_classifier(self.model, consecutive_perturb_layers)
        smap1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)  # Custom SMAP and SSIM
        return structural_similarity(smap1, smap, data_range=255)

    def _perturb_classifier(self, model: torch.nn.Module, layers: List[str]) -> None:
        for i in layers:
            prng = np.random.RandomState(42)
            new_parameters = prng.uniform(0, 0.2, self.sd[i].shape)  # Random Numbers Sampled from Uniform Distribution
            self.sd[i] = torch.tensor(new_parameters)

    def _reload_ckpt(self, device: Optional[Union[str, torch.device]] = "cpu") -> None:
        self.sd = deepcopy(self._ori_state_dict)
        freeze_module(self.model, eval_mode=True)
        self.device = device
        self.model.to(self.device)

    @staticmethod
    def _filter_names(names: List[str]) -> List[str]:
        res: List = []
        for i in range(len(names) - 1):
            if not names[i] in names[i + 1]:
                res.append(names[i])
        res.append(names[-1])
        return res

    def update(self, single_result: Dict) -> None:
        self._result.add_single_result(single_result)

    def get_result(self) -> SerializableResult:
        return self._result


METRICS.register_module(module=SanityCheck)
