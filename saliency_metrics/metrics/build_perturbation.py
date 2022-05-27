from abc import abstractmethod
from typing import Dict, Optional, Protocol, Union, runtime_checkable

from mmcv import Registry
from numpy import ndarray
from torch import Tensor

__all__ = ["PERTURBATIONS", "build_perturbation", "Perturbation"]


@runtime_checkable
class Perturbation(Protocol):
    """Image perturbation protocol.

    Perturbation-based metrics should provide custom perturbation classes that implement this protocol.
    """

    def __call__(self, img: Union[Tensor, ndarray], smap: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        """Perturb an image according to the saliency map.

        This function is a simple wrapper of the :meth:`~Perturbation.perturb` function.

        Args:
            img: Input image.
            smap: saliency map.

        Returns:
            Perturbed image.
        """
        return self.perturb(img, smap)

    @abstractmethod
    def perturb(self, img: Union[Tensor, ndarray], smap: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        """Perturb an image according to the saliency map.

        Args:
            img: Input image.
            smap: saliency map.

        Returns:
            Perturbed image.
        """
        raise NotImplementedError


PERTURBATIONS = Registry("perturbations")


def build_perturbation(cfg: Dict, default_args: Optional[Dict] = None) -> Perturbation:
    """Build an image perturbation instance.

    Args:
        cfg: config dictionary that contains at least the field ``"type"``.
        default_args: Other default args.

    Returns:
        Image perturbation that implements the ``Perturbation`` protocol.
    """
    return PERTURBATIONS.build(cfg=cfg, default_args=default_args)
