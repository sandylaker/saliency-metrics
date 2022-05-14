from copy import deepcopy
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
from mmcv import Registry
from torchvision import models

__all__ = ["build_classifier", "TIMM_CLASSIFIERS", "TORCHVISION_CLASSIFIERS", "CUSTOM_CLASSIFIERS"]


def _preprocess_cfg(cfg: Dict, default_args: Optional[Dict] = None) -> Dict:
    """Override the `cfg` with `default_args`."""
    cfg = deepcopy(cfg)
    if default_args is not None:
        for name, value in default_args.items():
            cfg.setdefault(name, value)
    return cfg


def _build_timm_classifier(registry: Registry, cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    """Build a classifier from `timm` library."""
    cfg = _preprocess_cfg(cfg, default_args=default_args)
    model_name = cfg.pop("type")
    return timm.create_model(model_name, **cfg)


def _build_torchvision_classifier(registry: Registry, cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    """Build a classifier from `torchvision` library."""
    cfg = _preprocess_cfg(cfg, default_args=default_args)
    model_name = cfg.pop("type")
    ckpt_path = cfg.pop("checkpoint_path", None)
    _builder = getattr(models, model_name)
    model = _builder(**cfg)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    return model


TIMM_CLASSIFIERS = Registry("timm_classifiers", scope="timm", build_func=_build_timm_classifier)
TORCHVISION_CLASSIFIERS = Registry(
    "torchvision_classifiers", scope="torchvision", build_func=_build_torchvision_classifier
)
CUSTOM_CLASSIFIERS = Registry("custom_classifiers", scope="custom")


def build_classifier(cfg: Dict, default_args: Optional[Dict] = None) -> nn.Module:
    """Build a classifier.

    This function supports building a classifier from timm_, torchvision_ or **custom-defined**
    models. ``cfg`` must contain the field "type", which should have the format ``<scope>.<model_name>``. Specifically,
    ``scope`` can be one of ``timm``, ``torchvision``, or ``custom``. When building a custom-defined classifier, the
    model should be already registered under ``saliency_metrics.models.CUSTOM_CLASSIFIERS`` registry. When building a
    classifier from ``torchvision`` or ``timm``, the ``model_name`` should be the name of corresponding builder
    function, e.g., ``resnet18``. I.e., ``cfg = dict(type="torchvision.resnet18,)`` is equivalent to call
    ``torchvision.models.resnet18``.

    .. _timm: https://rwightman.github.io/pytorch-image-models/

    .. _torchvision: https://pytorch.org/vision/stable/index.html

    Examples:
        Build a `torchvision` classifier:

        .. code-block:: python

            from torchvision.models.resnet import ResNet
            from saliency_metrics.models import build_classifier

            cfg_1 = dict(type="torchvision.resnet18, num_classes=2, pretrained=False)
            model = build_classifier(cfg_1)
            assert isinstance(model, ResNet)

        Build a `timm` classifier:

        .. code-block:: python

            from timm.models.efficientnet import EfficientNet
            from saliency_metrics.models import build_classifier

            cfg_2 = dict(type="timm.efficientnet_b0, num_classes=2)
            model = build_classifier(cfg_2)
            assert isinstance(model, EfficientNet)

        Build a `custom` classifier:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from saliency_metrics.models import CUSTOM_CLASSIFIERS, build_classifier

            # First register the class
            @CUSTOM_CLASSIFIERS.register_module()
            class MLP(nn.Module):
                def __init__(self, hidden_size: int = 10) -> None:
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(10, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 2))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.layers(x)

            # Now the instance of the class can be built from a config
            cfg_3 = dict(type="custom.MLP")
            model = build_classifier(cfg_3, default_args=dict(hidden_size=5))
            assert isinstance(model, MLP)

    Args:
        cfg: A config dict that contains the arguments for building a classifier. It should at least contain the
            field "type".
        default_args: Other default arguments.

    Returns:
        The classifier.
    """
    cfg = deepcopy(cfg)
    if "type" not in cfg:
        raise ValueError("Key 'type' must be contained in the config.")
    # timm and torchvision registries are actually empty. If using default build_from_cfg function to build
    # from CLASSIFIERS, it will raise an KeyError. We need to find the scope and call the corresponding build function
    scope, model_name = Registry.split_scope_key(cfg["type"])
    if scope is None:
        raise ValueError(f"type must be in format <scope>.<model_name>, but got {cfg['type']}.")
    if scope == "":
        raise ValueError("scope must not be an empty string.")
    cfg.update({"type": model_name})

    if scope == "timm":
        return TIMM_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    elif scope == "torchvision":
        return TORCHVISION_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    elif scope == "custom":
        return CUSTOM_CLASSIFIERS.build(cfg=cfg, default_args=default_args)
    else:
        raise ValueError(f"Invalid scope name, should be one of 'timm', 'torchvision', 'custom', but got {scope}.")
