from abc import ABCMeta, abstractmethod
from typing import Dict

from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def get_ind_to_cls(self) -> Dict[int, str]:
        """Get the dictionary mapping indices to class names.

        Returns:
            A dict mapping indices to class names.
        """
        pass

    @abstractmethod
    def get_cls_to_ind(self) -> Dict[str, int]:
        """Get the dictionary mapping class names to indices.

        Returns:
            A dict mapping class names to indices.
        """
        pass
