from datetime import timedelta
from logging import Logger
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.utils import setup_logger
from torch.optim import Optimizer

from ..build_perturbations import Perturbation

__all__ = [
    "get_train_step_fn",
    "get_eval_step_fn",
    "prob_transform",
    "logits_transform",
    "TrainStatsTextLogger",
    "MetricsTextLogger",
]


def get_train_step_fn(
    classifier: nn.Module,
    ptb_fn: Perturbation,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: Union[str, torch.device],
) -> Callable[[Engine, Dict], float]:  # pragma: no cover
    def _train_step_fn(engine: Engine, batch: Dict) -> float:
        classifier.train()
        img: torch.Tensor = batch["img"].to(device)
        smap: np.ndarray = batch["smap"]
        target: torch.Tensor = batch["target"].to(device)

        img = ptb_fn(img, smap)
        pred = classifier(img)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return _train_step_fn


def get_eval_step_fn(
    classifier: nn.Module, ptb_fn: Perturbation, device: Union[str, torch.device]
) -> Callable[[Engine, Dict], Dict[str, torch.Tensor]]:  # pragma: no cover
    def _eval_step_fn(engine: Engine, batch: Dict) -> Dict:
        classifier.eval()
        with torch.no_grad():
            img: torch.Tensor = batch.pop("img")
            smap: np.ndarray = batch.pop("smap")
            target: torch.Tensor = batch.pop("target")

            img = img.to(device)
            target = target.to(device)

            img = ptb_fn(img, smap)
            pred = classifier(img)

            batch.update({"pred": pred, "target": target})
            return batch

    return _eval_step_fn


def prob_transform(
    batch: Dict[str, Union[str, torch.Tensor, np.ndarray]]
) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    pred: torch.Tensor = batch["pred"]
    pred = pred.sigmoid()
    target: torch.Tensor = batch["target"]
    return (pred > 0.5).to(target), target


def logits_transform(
    batch: Dict[str, Union[str, torch.Tensor, np.ndarray]]
) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    return batch["pred"], batch["target"]


class TrainStatsTextLogger:  # pragma: no cover
    def __init__(self, interval: int = 1, logger: Optional[Logger] = None) -> None:  # pragma: no cover
        self._interval = interval

        self._logger = logger if logger is not None else setup_logger("saliency-metrics")
        self._timer = Timer(average=True)

    def _log_stats(self, engine: Engine, optimizer: Optimizer) -> None:  # pragma: no cover
        loss = engine.state.output
        iter_time = self._timer.value()
        max_iters = engine.state.max_epochs * engine.state.epoch_length
        remain_iters = max(0, max_iters - engine.state.iteration)
        eta_str = str(timedelta(seconds=int(remain_iters * iter_time)))

        log_str = f"Top fraction: {engine.state.top_fraction:.2f} Trial: {engine.state.trial} "
        log_str += f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}] "
        iter_in_epoch = engine.state.epoch_length % engine.state.epoch_length
        iter_in_epoch = iter_in_epoch if iter_in_epoch != 0 else engine.state.epoch_length

        log_str += f"Iteration [{iter_in_epoch}/{engine.state.epoch_length}]: "
        log_str += f"batch time: {iter_time:.4f}, eta: {eta_str}, "
        log_str += f"lr: {optimizer.param_groups[0]['lr']:.2e}, "
        log_str += f"loss: {loss:.4f} "
        self._logger.info(log_str)

    def attach(self, engine: Engine, optimizer: Optimizer) -> None:  # pragma: no cover
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self._interval), self._log_stats, optimizer)
        self._timer.attach(
            engine,
            start=Events.EPOCH_STARTED(once=1),
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )


class MetricsTextLogger:
    def __init__(self, logger: Optional[Logger] = None) -> None:  # pragma: no cover
        self._logger = logger if logger is not None else setup_logger("saliency-metrics")

    def _log_metrics(
        self, evaluator: Engine, evaluator_name: str, trainer: Optional[Engine] = None
    ) -> None:  # pragma: no cover
        if trainer is not None:
            epoch = trainer.state.epoch
            max_epochs = trainer.state.max_epochs
            log_str = f"Epoch [{epoch}/{max_epochs}]: "
        else:
            log_str = ""
        log_str += f"{evaluator_name} metrics: "
        log_str += "; ".join([f"{name}: {val:.4f}" for name, val in evaluator.state.metrics.items()])
        self._logger.info(log_str)

    def attach(self, evaluator: Engine, evaluator_name: str, trainer: Optional[Engine] = None) -> None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self._log_metrics, evaluator_name, trainer=trainer)
