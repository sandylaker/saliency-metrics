from typing import Optional, Sequence

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


class RoarPerturbation(torch.nn.Module):
    def __init__(self, top_fraction: float, mean: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        if top_fraction < 0 or top_fraction > 1:
            raise ValueError(f"top_fraction should be in the interval [0, 1], but got {top_fraction} .")
        # the quantile e.g. 0.9 in numpy.quantile means top 10%
        self.quantile = 1 - top_fraction
        _mean: Optional[Tensor] = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        if _mean is not None:
            self.register_buffer("_mean", _mean)
            self._has_mean = True
        else:
            self._has_mean = False

    def forward(self, img: Tensor, smap: ndarray) -> Tensor:
        return self.perturb(img, smap)

    def perturb(self, img: Tensor, smap: ndarray) -> Tensor:
        # mean (broad-casted) shape: (num_samples, num_channels, height, width)
        if self._has_mean:
            mean = self.get_buffer("_mean").view(1, -1, 1, 1)
        else:
            mean = torch.mean(img, dim=[2, 3], keepdim=True)

        threshold = np.quantile(smap, self.quantile, axis=[1, 2], keepdims=True)
        if self.quantile == 0.0:
            # add a small shift so that the smallest pixel value will also be perturbed when quantile is 0.0
            threshold -= 1e-8
        # mask shape: (num_samples, 1, height, width).
        # 1 indicates that the pixel needs to be replaced by the mean value
        mask = img.new_tensor(smap > threshold)
        mask.unsqueeze_(1)

        output = mean * mask + img * (1 - mask)
        return output
