from typing import Iterator, List, Tuple

import torch


class ProgressivePerturbation:
    def __init__(
        self,
        input_tensor: torch.Tensor,
        replace_tensor: torch.Tensor,
        sorted_inds: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._current_tensor = input_tensor.clone()
        self._replace_tensor = replace_tensor
        self._row_inds, self._col_inds = sorted_inds
        self._num_pixels = self._row_inds.size()[0]

    def _perturb_by_inds(self, row_inds: torch.Tensor, col_inds: torch.Tensor) -> None:
        self._current_tensor[..., row_inds, col_inds] = self._replace_tensor[..., row_inds, col_inds]

    @property
    def current_tensor(self) -> torch.Tensor:
        return self._current_tensor.clone()

    def perturb(self, forward_batch_size: int = 128, perturb_step_size: int = 10) -> Iterator[torch.Tensor]:
        num_perturbed_pixels = 0
        while num_perturbed_pixels < self._num_pixels:
            perturbed_images_batch: List[torch.tensor] = []
            for _ in range(forward_batch_size):
                step_size = min(perturb_step_size, (self._num_pixels - num_perturbed_pixels))
                perturbed_row_indices = self._row_inds[num_perturbed_pixels : num_perturbed_pixels + step_size]
                perturbed_col_indices = self._col_inds[num_perturbed_pixels : num_perturbed_pixels + step_size]
                self._perturb_by_inds(perturbed_row_indices, perturbed_col_indices)
                perturbed_images_batch.append(self.current_tensor)
                num_perturbed_pixels += step_size
                if num_perturbed_pixels >= self._num_pixels:
                    break
            batch = torch.cat(perturbed_images_batch, dim=0)
            yield batch
