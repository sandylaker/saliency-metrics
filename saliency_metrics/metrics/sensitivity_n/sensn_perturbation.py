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
        """generate ``num_masks`` many random mask tensors with ``n`` masks.

        Args:
            spatial_size: Spatial size of masking tensor
            device: device on which the torch.Tensor will be allocated

        return masking tensor
        """
        masks: List[torch.Tensor] = []
        h, w = spatial_size

        for _ in range(self.num_masks):
            inds = np.random.choice(h * w, self._n, replace=False)
            inds = np.unravel_index(inds, (h, w))
            mask = np.ones(spatial_size)
            mask[inds] = 0
            masks.append(torch.tensor(mask, dtype=torch.float32, device=device))
        return masks

    def perturb(self, img: torch.Tensor, smap: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Perturbation of image with masking tensor.

        Args:
            img: torch.Tensor representing the image to be perturbated
            smap: torch.Tensor representing the saliency map

        Returns:
            batched sample: torch.Tensor with shape(num_masks + 1, num_channels, height, width),
            where the last sample it the unperturbed image.
            sum_attributions: np.ndarray with shape (num_masks).
            Each element representing the sum of attributions of each random masked saliency map.
        """
        batched_sample: List[torch.Tensor] = []
        sum_attributions: List[torch.Tensor] = []

        if self._masks is None:
            spatial_size = img.shape[-2:]
            self._masks = self._generate_random_masks(spatial_size, device=img.device)
        for mask in self._masks:
            batched_sample.append(img * mask)
            sum_attributions.append((smap * mask).sum())
        batched_sample.append(img)
        sum_attributions = torch.stack(sum_attributions).numpy()
        batched_sample = torch.stack(batched_sample)
        return batched_sample, sum_attributions
