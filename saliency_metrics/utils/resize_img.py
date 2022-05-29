from typing import Any, Tuple, TypeVar

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

T = TypeVar("T", torch.Tensor, np.ndarray)


def resize_img(img: T, output_shape: Tuple[int, int], **kwargs: Any) -> T:
    """Resize and interpolate the image to the given shape.

    This function simply calls ``torch.nn.functional.interpolate`` if ``img`` is a ``torch.Tensor``,
    or ``cv2.resize`` if ``img`` is a ``numpy.ndarray``.

    .. note::
        If ``img`` is a ``numpy.ndarray``, then its data type must be ``numpy.uint8``.

    Args:
        img: Input image. Can be ``torch.Tensor`` with shape ``(num_samples, num_channels, height, width)`` or
            ``numpy.ndarray`` with shape ``(height, width, 3)`` or ``(height, width)`` .
        output_shape: output shape in the format of ``(out_height, out_width)``.
        **kwargs: other interpolation arguments. See also `interpolate`_ or `resize`_.

    Returns:
        The interpolated image.

    .. _interpolate: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html#torch.nn.functional.interpolate  # noqa
    .. _resize: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    """
    if isinstance(img, torch.Tensor):
        return interpolate(img, size=output_shape, **kwargs)
    else:
        return cv2.resize(img, dsize=output_shape, **kwargs)
