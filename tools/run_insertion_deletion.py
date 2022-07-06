from argparse import ArgumentParser, Namespace

from saliency_metrics.metrics.insertion_deletion.insertion_deletion import run_insertion_deletion


def parse_args() -> Namespace:
    parser = ArgumentParser("Run Insertion Deletion")
    parser.add_argument("--work-dir", default="workdirs/insertion_deletion", help="Output directory for storing files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_insertion_deletion(args.work_dir)
