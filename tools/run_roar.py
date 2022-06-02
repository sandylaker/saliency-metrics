import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Dict

from ignite.utils import setup_logger
from mmcv import Config, DictAction, mkdir_or_exist

from saliency_metrics.metrics import ReTrainingMetric, build_metric


def parse_args() -> Namespace:
    parser = ArgumentParser("Run ROAR.")
    parser.add_argument("config", help="Path to the config file.")
    parser.add_argument("--work-dir", default="workdirs/roar/", help="Output directory for storing files.")
    parser.add_argument("--backend", help="DDP backend")
    parser.add_argument("--nproc-per-node", type=int, help="Number of processes per node.")
    parser.add_argument("--nnodes", type=int, help="Number of nodes.")
    parser.add_argument("--node-rank", type=int, help="Node rank.")
    parser.add_argument("--master-addr", help="Master node TCP/IP address.")
    parser.add_argument("--master-port", help="Master node port.")
    parser.add_argument("--init-method", help="Initialization method for process groups.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override some settings in the used config, "
        "the key-value pair in xxx=yyy will be merged into config file.",
    )

    return parser.parse_args()


def run_roar(cfg: Config, dist_args: Dict) -> None:
    roar: ReTrainingMetric = build_metric(cfg.metric)
    roar.evaluate(cfg, dist_args=dist_args)
    result = roar.get_result()

    file_path = osp.join(cfg["work_dir"], "roar_result.json")
    result.dump(file_path)
    logger = setup_logger("saliency-metrics")
    logger.info(f"ROAR finished. The result is saved to {file_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg: Config = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mkdir_or_exist(args.work_dir)
    cfg.dump(osp.join(args.work_dir, "roar_config.py"))
    cfg["work_dir"] = args.work_dir

    dist_args = {
        "backend": args.backend,
        "nproc_per_node": args.nproc_per_node,
        "nnodes": args.nndoes,
        "node_rank": args.node_rank,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "init_method": args.init_method,
    }

    run_roar(cfg=cfg, dist_args=dist_args)
