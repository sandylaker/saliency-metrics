import inspect
from typing import Dict, List, Optional, Union

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from mmcv import Registry, build_from_cfg

__all__ = ["PIPELINES", "build_pipeline"]

PIPELINES = Registry("pipelines")


def _register_albu_transforms() -> List:  # pragma: no cover
    albu_transforms = []
    for module_name in dir(A):
        if module_name.startswith("_"):
            continue
        transform = getattr(A, module_name)
        if inspect.isclass(transform) and issubclass(transform, A.BasicTransform):
            PIPELINES.register_module()(transform)
            albu_transforms.append(module_name)
    return albu_transforms


transforms = _register_albu_transforms()
PIPELINES.register_module(module=ToTensorV2)


def build_pipeline(cfg: Union[Dict, List], default_args: Optional[Dict] = None) -> Union[object, A.Compose]:
    """Build an albumentations_ pipeline for image augmentation.

    .. code-block:: python

        import numpy as np
        import albumentations as A
        from saliency_metrics.datasets import build_pipeline

        img = np.random.randint(0, 225, (250, 250), dtype=np.uint8)
        bboxes = [[10, 100, 10, 100]]
        labels = [0]

        # build single augmentation
        cfg_1 = dict(type="GaussianBlur", blur_limit=(3, 7), p=0.5)
        pipeline_1 = build_pipeline(cfg_1)
        img_1 = pipeline_1(image=img)["image"]

        # build multiple augmentations and perform transformation for bounding boxes
        cfg_2 = [
            dict(type="RandomCrop", height=200, width=200),
            dict(type="Resize", height=224, width=224),
            dict(type="ToTensorV2")
        ]
        default_args = dict(bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
        pipeline_2 = build_pipeline(cfg_2, default_args=default_args)
        output_2 = pipeline_2(image=img, bboxes=bboxes, labels=labels)
        img_2, bboxes_2, labels_2 = output_2["image"], output_2["bboxes"], output_2["labels"]


    Args:
        cfg: Config dictionary. If ``cfg`` is a dict, then the function returns a single ``albumentations``
            augmentation. If ``cfg`` is a list of dict, then the function first builds each augmentation respectively,
            and then compose them into an ``albumentations.Compose``.
        default_args: Other default arguments.

    .. _albumentations: https://albumentations.ai/docs/

    Returns:
        A single ``albumentations`` augmentation or ``albumentations.Compose``.
    """
    if isinstance(cfg, Dict):
        return build_from_cfg(cfg, PIPELINES)
    else:
        pipeline = []
        for transform_cfg in cfg:
            t = build_pipeline(transform_cfg)
            pipeline.append(t)
        if default_args is None:
            default_args = {}
        return A.Compose(pipeline, **default_args)
