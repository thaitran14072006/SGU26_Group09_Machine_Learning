"""Nam G4: preflight, nested KNN/SVM, source snapshot, verification."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wesad.g4_nam import prepare_config, preflight, run, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/g4_nam_knn_svm_tuning.yaml')
    p.add_argument('--features', help='Override local path, never the locked data hash')
    p.add_argument('--schema')
    p.add_argument('--output')
    scope = p.add_mutually_exclusive_group()
    scope.add_argument('--outer-subjects', nargs='+', help='Declared main shard; train still has 14 people')
    scope.add_argument('--quick-subject', help='Separate smoke run; forbidden in main merge')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    try:
        c = prepare_config(a.config, features=a.features, schema=a.schema, output=a.output,
                           outer_subjects=a.outer_subjects, quick_subject=a.quick_subject)
        if a.preflight_only:
            frame, _ = preflight(c)
            print(f"PASS: Python 3.12.14; {len(frame)} windows; 68 features; 15 input subjects; "
                  f"{len(c['outer_subjects'])} outer test subjects; 4 candidates/model; shared core.")
        elif a.verify_only:
            print(verify(c))
        else:
            print(run(c, resume=a.resume))
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
