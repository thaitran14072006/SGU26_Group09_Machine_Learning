"""Verify existing Nam G4 output, compare with G3, export figures and handoff ZIP; no fitting."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wesad.g4_nam import prepare_config, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/g4_nam_knn_svm_tuning.yaml')
    p.add_argument('--features')
    p.add_argument('--schema')
    p.add_argument('--output', help='Existing RUN output, not a new report directory')
    p.add_argument('--outer-subjects', nargs='+')
    p.add_argument('--g3-metrics', default='reports/g3_group/metrics_by_subject.csv')
    a = p.parse_args()
    try:
        c = prepare_config(a.config, features=a.features, schema=a.schema, output=a.output,
                           outer_subjects=a.outer_subjects)
        print(report(c, a.g3_metrics))
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
