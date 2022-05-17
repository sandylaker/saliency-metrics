from typing import Dict, Optional

from mmcv import Registry
from torch.utils.data import Dataset

from .image_folder import ImageFolder

DATASETS = Registry("datasets")
# Sphinx will throw errors when using the decorator to register a subclass,
# and the inherited methods will not be displayed.
# Therefore, we register the class by calling the function
DATASETS.register_module(module=ImageFolder)


def build_dataset(cfg: Dict, default_args: Optional[Dict] = None) -> Dataset:
    """Build a dataset.

    Args:
        cfg: A config dict. It should at least contain the field "type", which is the registered name of the dataset.
        default_args: Other default arguments.

    Returns:
        An instance of ``torch.utils.data.Dataset``.
    """
    return DATASETS.build(cfg, default_args=default_args)
