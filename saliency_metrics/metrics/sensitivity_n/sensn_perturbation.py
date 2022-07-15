from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class SensitivityNPerturbation:
    def __init__(self, n: int, num_masks: int = 100) -> None:
        self._n = n
        self.num_masks = num_masks
        self._masks: Optional[List[Tensor]] = None

    def _generate_random_masks(
        self, spatial_size: Tuple[int, int], device: Optional[Union[str, torch.device]] = None
    ) -> List[torch.Tensor]:
        """generate masking Tensor with.

        return masks
        """
        masks: List[torch.Tensor] = []
        h, w = spatial_size
        for _ in range(self.num_masks):
            inds = np.random.choice(h * w, self._n, replace=False)
            inds = np.unravel_index(inds, (h, w))
            mask = np.zeros(spatial_size)
            mask[inds] = 1
            mask = 1 - mask
            masks.append(torch.Tensor(mask, dtype=torch.float32, device=device))
        return masks

    def perturb(self, img: torch.Tensor, smap: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        if self._masks is None:
            spatial_size = img.shape[-2:]
            self._masks = self._generate_random_masks(spatial_size, device=img.device)
            batched_samples = []
            sum_attributions = []
            for i, x in self._masks:
                batched_samples.append(img * self._masks[i])
                sum_attributions.append((smap * self._masks[i]).sum())
            batched_samples.append(img)

            sum_attributions = torch.stack(sum_attributions).numpy()
            batched_samples = torch.stack(batched_samples)

        return batched_samples, sum_attributions
