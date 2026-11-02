"""Chạy đánh giá theo người từ bảng đặc trưng đã lưu."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wesad.evaluation import run_experiment
from wesad.models import MODEL_NAMES
from wesad.utils import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--features")
    parser.add_argument("--schema")
    parser.add_argument("--output")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--task", choices=["multiclass", "binary", "both"])
    parser.add_argument("--nested", action="store_true", default=None)
    parser.add_argument("--outer-subjects", nargs="+")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    for argument, key in (("features", "features"), ("schema", "schema"), ("output", "output_dir"),
                          ("models", "models"), ("nested", "nested")):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    if args.task:
        config["tasks"] = ["multiclass", "binary"] if args.task == "both" else [args.task]
    result = run_experiment(config, args.resume, args.outer_subjects, args.allow_partial)
    print(f"Run status: {result['status']}")


if __name__ == "__main__":
    main()
