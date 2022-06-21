from typing import Dict

import numpy as np
import torch
from insertion_deletion_perturbation import ProgressivePerturbation
from insertion_deletion_result import InsertionDeletionResult
from sklearn import metrics
from torchvision.transforms import GaussianBlur

from saliency_metrics.metrics.build_metric import ReInferenceMetric
from saliency_metrics.models.build_classifier import build_classifier


class InsertionDeletion(ReInferenceMetric):
    def __init__(
        self,
        classifier_cfg: Dict,
        forward_batch_size: int = 128,
        perturb_step_size=10,
        sigma: float = 5.0,
        summarized: bool = False,
    ):

        self._result = InsertionDeletionResult(summarized)

        self.classifier = build_classifier(classifier_cfg)
        self.gaussian_blur = GaussianBlur(int(2 * sigma - 1), sigma)
        self.forward_batch_size = forward_batch_size
        self.perturb_step_size = perturb_step_size

    def evaluate(self, img: torch.Tensor, smap: torch.Tensor, target: int):
        num_pixels = torch.numel(smap)
        _, inds = torch.topk(smap.flatten(), num_pixels)
        row_inds, col_inds = (torch.tensor(x) for x in np.unravel_index(inds.numpy(), smap.size()))

        # deletion
        del_perturbation = ProgressivePerturbation(img, torch.zeros_like(img), (row_inds, col_inds))
        del_scores = []
        with torch.no_grad():
            for batch in del_perturbation.perturb(
                forward_batch_size=self.forward_batch_size, perturb_step_size=self.perturb_step_size
            ):
                scores = torch.softmax(self.classifier(batch), -1)[:, target]
                del_scores.append(scores.cpu().numpy())

        del_scores = np.concatenate(del_scores, 0)
        # TODO Use scipy method
        del_auc = metrics.auc(np.linspace(0, 1, len(del_scores)), del_scores)

        blurred_img = self.gaussian_blur(img)
        ins_perturbation = ProgressivePerturbation(blurred_img, img, (row_inds, col_inds))
        ins_scores = []
        with torch.no_grad():
            for batch in ins_perturbation.perturb(
                forward_batch_size=self.forward_batch_size, perturb_step_size=self.perturb_step_size
            ):
                scores = torch.softmax(self.classifier(batch), -1)[:, target]
                ins_scores.append(scores.cpu().numpy())

        ins_scores = np.concatenate(ins_scores, 0)
        ins_auc = metrics.auc(np.linspace(0, 1, len(ins_scores)), ins_scores)

        single_result = {"del_scores": del_scores, "ins_scores": ins_scores, "del_auc": del_auc, "ins_auc": ins_auc}
        return single_result

    def update(self, single_result: Dict):
        self._result.add_single_result(single_result)

    # TODO - remove wrapper function
    @property
    def get_result(self):
        return self._result
