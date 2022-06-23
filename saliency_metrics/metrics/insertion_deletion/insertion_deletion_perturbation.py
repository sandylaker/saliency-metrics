from typing import Iterator, List

import torch


class ProgressivePerturbation:
    def __init__(
        self,
        input_tensor: torch.Tensor,
        replace_tensor: torch.Tensor,
        sorted_inds: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._current_tensor = input_tensor.clone()
        self._replace_tensor = replace_tensor
        self._row_inds, self._col_inds = sorted_inds
        self._num_pixels = self._row_inds.size()[0]

    def _perturb_by_inds(self, row_inds: torch.Tensor, col_inds: torch.Tensor) -> None:
        self._current_tensor[:, row_inds, col_inds] = self._replace_tensor[:, row_inds, col_inds]

    @property
    def current_tensor(self) -> torch.Tensor:
        return self._current_tensor

    def perturb(self, forward_batch_size: int = 128, perturb_step_size: int = 10) -> Iterator[torch.Tensor]:
        pixels_perturbed = 0
        while pixels_perturbed < self._num_pixels:
            forward_batch_count = 0
            perturbed_images_batch: List[torch.tensor] = []
            while forward_batch_count < forward_batch_size:
                step_size = min(perturb_step_size, (self._num_pixels - pixels_perturbed))
                perturbed_row_indices = self._row_inds[pixels_perturbed : pixels_perturbed + step_size]
                perturbed_col_indices = self._col_inds[pixels_perturbed : pixels_perturbed + step_size]
                self._perturb_by_inds(perturbed_row_indices, perturbed_col_indices)
                # TODO - check if only detach works - RESULT - doesn't work
                perturbed_images_batch.append(self.current_tensor.clone().detach())
                forward_batch_count += 1
                pixels_perturbed += step_size
                if pixels_perturbed >= self._num_pixels:
                    break
            batch = torch.stack(perturbed_images_batch)
            yield batch
