from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import torch.nn as nn
from numpy import ndarray
from torch import Tensor

from ..models import get_module
from ..utils import resize_img

__all__ = ["AttributionMethod", "CaptumGradCAM"]


@runtime_checkable
class AttributionMethod(Protocol):
    """Protocol of attribution (also known as explanation) methods.

    This protocol is mainly used in the Sanity Check metric, where an attribution method must implement this protocol.
    """

    @abstractmethod
    def attribute(
        self,
        img: Tensor,
        target: Union[int, Sequence[int], Tensor],
        as_ndarray: bool = True,
        interpolate: bool = True,
        interpolate_args: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray]]]:
        """Attribute and produce the saliency map.

        .. note::
            The method performs attribution on single image, i.e., the local explanation.

        Args:
            img: Input image with shape ``(1, num_channels, height, width)``.
            target: Ground-truth target.
            as_ndarray: If True, then convert the saliency map into a ndarray.
            interpolate: if True, then resize and interpolate the saliency map to the image's spatial size.
            interpolate_args: Other arguments for interpolation. See also :func:`saliency_metrics.utils.resize_img`.
            **kwargs: Other keyword arguments for attribution.

        Returns:
            A saliency map or a tuple of saliency maps.
        """
        raise NotImplementedError


class CaptumGradCAM(AttributionMethod):
    """A wrapper class for ``captum.attr.LayerGradCam``.

    This class is only for internal testing.
    """

    def __init__(self, classifier: nn.Module, layer: str, **kwargs: Any) -> None:
        from captum.attr import LayerGradCam

        layer: nn.Module = get_module(classifier, layer)
        self._grad_cam = LayerGradCam(forward_func=classifier, layer=layer, **kwargs)

    def attribute(
        self,
        img: Tensor,
        target: Union[int, Sequence[int], Tensor],
        as_ndarray: bool = True,
        interpolate: bool = True,
        interpolate_args: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray]]]:
        height, width = img.shape[-2:]
        smap: Union[Tensor, Tuple[Tensor]] = self._grad_cam.attribute(img, target, **kwargs)

        smap_list = (
            [
                smap.detach(),
            ]
            if not isinstance(smap, (list, tuple))
            else [s.detach() for s in smap]
        )

        if interpolate:
            interpolate_args = dict() if interpolate_args is None else interpolate_args
            smap_list = [resize_img(s, output_shape=(height, width), **interpolate_args) for s in smap_list]
        if as_ndarray:
            smap_list = [s.detach().numpy() for s in smap_list]

        if len(smap_list) == 1:
            return smap_list[0]
        else:
            return tuple(smap_list)
