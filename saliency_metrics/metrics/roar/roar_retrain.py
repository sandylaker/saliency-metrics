import os.path as osp
from copy import deepcopy
from datetime import datetime
from typing import List

import ignite.distributed as idist
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.handlers.checkpoint import Checkpoint, DiskSaver
from ignite.handlers.param_scheduler import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.metrics import Accuracy
from ignite.utils import manual_seed, setup_logger
from mmcv import Config
from mmcv.runner import build_optimizer

from saliency_metrics.datasets import build_dataset
from saliency_metrics.models import build_classifier
from ..build_perturbations import build_perturbation
from .roar_retrain_utils import *  # noqa:F403


def roar_single_trial(
    local_rank: int, cfg: Config, top_fraction: float, trial: int, test_acc_list: List[float], seed: int = 42
) -> None:  # pragma: no cover
    rank = idist.get_rank()
    manual_seed(seed + rank + trial)
    device = idist.device()

    logger = setup_logger(
        "saliency-metrics", filepath=osp.join(cfg.work_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )

    if trial == 0:
        logger.info(f"Config:\n{cfg.pretty_text}")

    train_set = build_dataset(cfg.data["train"])
    val_set = build_dataset(cfg.data["val"])
    test_set = build_dataset(cfg.data["test"])

    val_data_loader_cfg = deepcopy(cfg.data["data_loader"])
    val_data_loader_cfg.update({"batch_size": val_data_loader_cfg["batch_size"] * 4, "shuffle": False})

    train_loader = idist.auto_dataloader(train_set, **cfg.data["data_loader"])
    val_loader = idist.auto_dataloader(val_set, **val_data_loader_cfg)
    test_loader = idist.auto_dataloader(test_set, **val_data_loader_cfg)
    epoch_length = len(train_loader)

    classifier = build_classifier(cfg.classifier)
    classifier.to(device)
    classifier = idist.auto_model(classifier, sync_bn=cfg.get("sync_bn", False))
    ptb_fn = build_perturbation(cfg.ptb_fn)
    if isinstance(ptb_fn, nn.Module):
        ptb_fn = ptb_fn.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = build_optimizer(model=classifier, cfg=cfg.optimizer)
    optimizer = idist.auto_optim(optimizer)

    trainer = Engine(get_train_step_fn(classifier, ptb_fn, criterion, optimizer, device))
    trainer.state.top_fraction = top_fraction
    trainer.state.trial = trial
    trainer.logger = logger

    eval_step_fn = get_eval_step_fn(classifier, ptb_fn, device)
    evaluator_val = Engine(eval_step_fn)
    evaluator_val.logger = logger

    evaluator_test = Engine(eval_step_fn)
    evaluator_test.logger = logger

    @idist.one_rank_only(rank=0)
    def add_roar_result(engine_test: Engine, acc_list: List[float]) -> None:
        acc_list.append(engine_test.state.metrics["accuracy"])

    evaluator_test.add_event_handler(Events.COMPLETED, add_roar_result, test_acc_list)

    pbar_val = ProgressBar(persist=True)
    pbar_val.attach(evaluator_val)
    pbar_test = ProgressBar(persist=True)
    pbar_test.attach(evaluator_test)

    val_metrics = {"accuracy": Accuracy(output_transform=prob_transform, device=device)}
    test_metrics = {"accuracy": Accuracy(output_transform=prob_transform, device=device)}
    for name, metric in val_metrics.items():
        metric.attach(engine=evaluator_val, name=name)
    for name, metric in test_metrics.items():
        metric.attach(engine=evaluator_test, name=name)

    val_metrics_logger = MetricsTextLogger(logger=logger)
    test_metrics_logger = MetricsTextLogger(logger=logger)
    val_metrics_logger.attach(evaluator_val, evaluator_name="val", trainer=trainer)
    test_metrics_logger.attach(evaluator_test, evaluator_name="test", trainer=trainer)

    # trainer handlers
    def run_validation(engine_train: Engine, engine_val: Engine) -> None:
        engine_val.run(val_loader)

    def run_test(engine_train: Engine, engine_test: Engine) -> None:
        engine_test.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.val_interval), run_validation, evaluator_val)
    trainer.add_event_handler(Events.COMPLETED, run_test, evaluator_test)

    cycle_size = cfg.max_epochs * epoch_length
    lr = cfg.optimizer["lr"]
    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer, param_name="lr", start_value=lr, end_value=lr * 0.01, cycle_size=cycle_size
    )
    lr_scheduler = create_lr_scheduler_with_warmup(
        lr_scheduler, warmup_start_value=0.01 * lr, warmup_duration=1000, warmup_end_value=lr
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    to_save = {"classifier": classifier}
    ckpt_dir = osp.join(cfg.work_dir, "ckpts", f"{top_fraction}_trial_{trial}")
    save_handler = DiskSaver(ckpt_dir, require_empty=True)
    score_fn = Checkpoint.get_default_score_fn("accuracy")
    ckpt_handler = Checkpoint(
        to_save,
        save_handler,
        n_saved=1,
        score_name="accuracy",
        score_function=score_fn,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        greater_or_equal=True,
    )
    evaluator_val.add_event_handler(Events.COMPLETED, ckpt_handler)

    train_stats_logger = TrainStatsTextLogger(interval=cfg.log_interval, logger=logger)
    train_stats_logger.attach(trainer, optimizer)

    trainer.run(data=train_loader, max_epochs=cfg.max_epochs)
