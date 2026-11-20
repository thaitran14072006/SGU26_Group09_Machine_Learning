"""Chạy đúng một thí nghiệm tiếp theo trong quy trình LR."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wesad.lr_workflow import main

if __name__ == '__main__':
    main()
