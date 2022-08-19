from typing import Any, Dict, List, Union

import numpy as np
import torch
from scipy.integrate import trapezoid
from torchvision.transforms import GaussianBlur

from saliency_metrics.models.build_classifier import build_classifier
from saliency_metrics.models.model_utils import freeze_module
from ..build_metric import METRICS, ReInferenceMetric
from .insertion_deletion_perturbation import ProgressivePerturbation
from .insertion_deletion_result import InsertionDeletionResult


class InsertionDeletion(ReInferenceMetric):
    def __init__(
        self,
        classifier_cfg: Dict,
        forward_batch_size: int = 128,
        perturb_step_size: int = 10,
        sigma: float = 5.0,
        summarized: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> None:

        self._result = InsertionDeletionResult(summarized)

        self.classifier = build_classifier(classifier_cfg)
        freeze_module(self.classifier, eval_mode=True)
        self.device = device
        self.classifier.to(self.device)

        self.gaussian_blur = GaussianBlur(int(2 * sigma - 1), sigma)
        if forward_batch_size <= 0:
            raise ValueError("forwarded_batch_size should be greater than zero, " f"but got {forward_batch_size}.")
        self.forward_batch_size = forward_batch_size
        if perturb_step_size <= 0:
            raise ValueError(
                "perturb_step_size should be greater than zero and"
                f"less than the number of elements in smap, but got {perturb_step_size}."
            )
        self.perturb_step_size = perturb_step_size

    def evaluate(self, img: torch.Tensor, smap: torch.Tensor, target: int, **kwargs: Any) -> Dict:
        num_pixels = torch.numel(smap)
        if self.perturb_step_size >= num_pixels:
            raise ValueError(
                "perturb_step_size should be less than the number of elements in smap, "
                f"but got {self.perturb_step_size}."
            )

        if img.dim() != 4 or img.size()[0] != 1:
            raise ValueError(
                "img should be a 4-D tensor with the size of the first dimension being 1, "
                f"but got shape: {img.size()}."
            )

        _, inds = torch.topk(smap.flatten(), num_pixels)
        row_inds, col_inds = (torch.tensor(x) for x in np.unravel_index(inds.numpy(), smap.size()))

        # deletion
        del_perturbation = ProgressivePerturbation(img, torch.zeros_like(img), (row_inds, col_inds))
        del_scores: List[float] = []
        with torch.no_grad():
            for batch in del_perturbation.perturb(
                forward_batch_size=self.forward_batch_size, perturb_step_size=self.perturb_step_size
            ):
                scores = torch.softmax(self.classifier(batch), -1)[:, target]
                del_scores.append(scores.cpu().numpy())

        del_scores = np.concatenate(del_scores, 0)
        del_auc = trapezoid(del_scores, dx=1.0 / len(del_scores))

        blurred_img = self.gaussian_blur(img)
        ins_perturbation = ProgressivePerturbation(blurred_img, img, (row_inds, col_inds))
        ins_scores: List[float] = []
        with torch.no_grad():
            for batch in ins_perturbation.perturb(
                forward_batch_size=self.forward_batch_size, perturb_step_size=self.perturb_step_size
            ):
                scores = torch.softmax(self.classifier(batch), -1)[:, target]
                ins_scores.append(scores.cpu().numpy())

        ins_scores = np.concatenate(ins_scores, 0)
        ins_auc = trapezoid(ins_scores, dx=1.0 / len(ins_scores))

        single_result = {
            "del_scores": del_scores,
            "ins_scores": ins_scores,
            "img_path": kwargs["img_path"],
            "del_auc": del_auc,
            "ins_auc": ins_auc,
        }
        return single_result

    def update(self, single_result: Dict) -> None:
        self._result.add_single_result(single_result)

    def get_result(self) -> InsertionDeletionResult:
        return self._result


METRICS.register_module(module=InsertionDeletion)
