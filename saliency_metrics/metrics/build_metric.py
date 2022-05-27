from abc import abstractmethod
from typing import Dict, Optional, Protocol, Union, runtime_checkable

from mmcv import Registry
from numpy import ndarray
from torch import Tensor

from .serializable_result import SerializableResult

__all__ = ["ReInferenceMetric", "ReTrainingMetric", "METRICS", "build_metric"]


@runtime_checkable
class ReInferenceMetric(Protocol):
    """Re-inference based metric.

    A Metric implementing this protocol performs per-sample evaluation at inference time. Specifically, it first
    perturbs the input image according to the saliency map and then measure the degradation of the model's prediction.
    """

    _result: SerializableResult

    @abstractmethod
    def update(self, single_result: Dict) -> None:
        """Given the evaluation result on a single sample, update the cached result (for the whole dataset).

        Args:
            single_result: Evaluation result on a single sample.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, img: Union[Tensor, ndarray], smap: Union[Tensor, ndarray], target: Union[Tensor, int]) -> Dict:
        """Perform evaluation on a single sample.

        Args:
            img: Input image.
            smap: Saliency map.
            target: Ground-truth target.

        Returns:
            Evaluation result on a single sample.
        """
        raise NotImplementedError

    def get_result(self) -> SerializableResult:
        """Get the cached result.

        Returns:
            The final evaluation result for the whole dataset.
        """
        return self._result


@runtime_checkable
class ReTrainingMetric(Protocol):
    """Re-training based metric.

    A metric implementing this protocol re-trains a model on a perturbed dataset and evaluate the performance
    degradation.
    """

    _result: SerializableResult

    @abstractmethod
    def evaluate(self, cfg: Dict, dist_args: Optional[Dict] = None) -> None:
        """Perform re-training evaluation on the whole dataset.

        Args:
            cfg: Config dictionary. It specifies the hyper-parameters of e.g., dataset, model, optimizer, lr-scheduler,
                max epochs etc.
            dist_args: DDP training hyper-parameters e.g. ``nproc_per_node``, ``backend`` etc. See also: `Parallel`_.

        Returns:
            None

        .. _Parallel: https://pytorch.org/ignite/generated/ignite.distributed.launcher.Parallel.html
        """
        raise NotImplementedError

    def get_result(self):
        """Get the cached result.

        Returns:
            The final evaluation result for the whole dataset.
        """
        return self._result


METRICS = Registry("Metrics")


def build_metric(cfg: Dict, default_args: Optional[Dict] = None) -> Union[ReInferenceMetric, ReTrainingMetric]:
    """Build an evaluation metric.

    Args:
        cfg: Config dictionary.
        default_args: Other default arguments.

    Returns:
        A evaluation metric.
    """
    return METRICS.build(cfg=cfg, default_args=default_args)
