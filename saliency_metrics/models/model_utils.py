from typing import Optional

import torch.nn as nn


def get_module(model: nn.Module, module: str) -> Optional[nn.Module]:
    r"""Get a specific layer in a model.

    This function is adapted from `TorchRay <https://github.com/facebookresearch/TorchRay>`_.
    :attr:`module` is the name of a module (as given by the :func:`named_modules` function for :class:`torch.nn.Module`
    objects). The function searches for a module with the name :attr:`module` and returns a :class:`torch.nn.Module`
    if found; otherwise, ``None`` is returned.

    Examples:
        .. code-block:: python

            from saliency_metrics.models import build_classifier, get_module

            cfg = dict(type="torchvision.resnet18", num_classes=2)
            model = build_classifier(cfg)

            # get the last block
            _ = get_module(model, "layer4.1")
            # get the last BN layer
            _ = get_module(model, "layer4.1.bn2")

    Args:
        model: Model in which to search for layer.
        module: Name of layer.

    Returns:
        Specific ``nn.Module`` layer (``None`` if the layer isn't found).
    """
    if not isinstance(module, str):
        raise TypeError(f"module can only be a str, but got {module.__class__.__name__}")

    if module == "":
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None


def freeze_module(model: nn.Module, module: Optional[str] = None, eval_mode: bool = True) -> None:
    """Freeze a specific module of the model.

    This function freezes a specific layer of the model by setting the `requires_grad` flag of its parameters to False.
    It also converts the whole model into `eval` mode, if `eval_mode` is True.

    Examples:
        .. code-block:: python

            from saliency_metrics.models import build_classifier, get_module

            model_1 = build_classifier(dict(type="timm.resnet18", num_classes=2))
            freeze_module(model_1, "fc", eval_mode=True)
            assert not model_1.training
            assert not model_1.fc.weight.requires_grad

            model_2 = build_classifier(dict(type="timm.resnet18", num_classes=2))
            freeze_module(model_2, None)
            for p in model_2.parameters():
                assert not p.requires_grad

    Args:
        model: Model to be processed.
        module: The name of the target module. If None, the target module to be frozen is the entire model.
        eval_mode: If True, turns the **entire** model into `eval` mode.

    Returns:
        None
    """
    if eval_mode:
        model.eval()

    target_module = model if module is None else get_module(model, module)
    for p in target_module.parameters():
        p.requires_grad = False
