from copy import deepcopy
from typing import List

import numpy as np
import torch
import torchvision
from PIL import Image
from skimage.metrics import structural_similarity

from saliency_metrics.metrics.build_metric import METRICS, ReInferenceMetric
from saliency_metrics.metrics.sanity_check.sanity_check_result import SanityCheckResult
from saliency_metrics.models.model_utils import freeze_module


class SanityCheck(ReInferenceMetric):
    def __init__(self, ori_model, perturb_layers, summarized=True):  # Removed attr_method, ssim_args = None for now
        self._result = SanityCheckResult(summarized)
        self.ori_model = ori_model  # Original Model
        self._ori_state_dict = self.ori_model.state_dict()
        self.model = deepcopy(self.ori_model)  # Model which will be used for performing randomization on
        self.sd = self.model.state_dict()
        self.perturb_layers = perturb_layers  # These are the layers to be perturbed
        self.all_layers: List = []
        for i, x in self.ori_model.named_parameters():
            self.all_layers.append(i)
        self._classifier_layers = self._filter_names(self.all_layers)

    def evaluate(self, image, smap, target, **kwargs):
        ssim_list: List = []
        for layer in self.perturb_layers:
            self._reload_ckpt()  # After every perturbation setting, reload
            consecutive_perturb_layers: List = []
            for j in self._classifier_layers[self._classifier_layers.index(layer) :]:
                consecutive_perturb_layers.append(j)
            ssim_float = self._sanity_check_single(smap, target, consecutive_perturb_layers)
            ssim_list.append(ssim_float)
        return dict(ssim=ssim_list)

    def _sanity_check_single(self, smap, target, consecutive_perturb_layers):
        self._perturb_classifier(self.model, consecutive_perturb_layers)
        smap1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)  # Custom smap created for now
        return structural_similarity(smap1, smap, data_range=255)

    def _perturb_classifier(self, model, layers):
        for i in layers:
            prng = np.random.RandomState()
            new_parameters = prng.uniform(0, 0.2, self.sd[i].shape)  # Random numbers sampled from Uniform distribution
            self.sd[i] = torch.tensor(new_parameters)

    def _reload_ckpt(self, device="cpu"):
        self.sd = deepcopy(self._ori_state_dict)
        freeze_module(self.model, eval_mode=True)
        self.device = device
        self.model.to(self.device)

    @staticmethod
    def _filter_names(names):
        res: List = []
        for i in range(len(names) - 1):
            if not names[i] in names[i + 1]:
                res.append(names[i])
        res.append(names[-1])
        return res

    def update(self, single_result):
        self._result.add_single_result(single_result)

    def get_result(self):
        return self._result


METRICS.register_module(module=SanityCheck)

classfier = torchvision.models.resnet18(pretrained=False, num_classes=10)
all_layers = []
for i, x in classfier.named_parameters():
    all_layers.append(i)
perturb_layers = all_layers[9:11]
san = SanityCheck(classfier, perturb_layers)
smap = np.random.randint(0, 255, (10, 10), dtype=np.uint8)  # SMAP of Original model
num_images = int(input("How many images?"))
for i in range(num_images):
    img = Image.open(input("Enter the file path"))
    target = int(input("Enter the target value for this image"))
    ssimdict = san.evaluate(img, smap, target)
    print("SSIM Dict: ", ssimdict)
    san.update(ssimdict)

res = san.get_result()
res.dump("workdirs/results.json")
