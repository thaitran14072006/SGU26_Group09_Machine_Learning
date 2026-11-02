"""Trích xuất từng subject; --resume dùng lại kết quả còn khớp mã băm."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wesad.extraction import run_extraction
from wesad.utils import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/features_chest.yaml")
    parser.add_argument("--local", help="YAML containing only data_root")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_extraction(read_config(args.config, args.local), args.subjects, args.resume)


if __name__ == "__main__":
    main()
