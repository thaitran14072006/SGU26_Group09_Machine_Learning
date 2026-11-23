"""Run EXP-009 using the common nested LOSO engine and independent G4 audit."""
import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wesad.evaluation import run_experiment
from wesad.g4 import inputs, candidates, verify_run
from wesad.utils import read_config, resolve_path, write_csv, write_json

DEFAULT = 'experiments/02_tuning/EXP-009_RF/config.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config)
    if (config['models'] != ['rf'] or config['nested'] is not True
            or config['balance'] is not False or config['inner_splits'] != 3):
        raise ValueError('EXP-009 requires RF-only, nested 3-fold CV, without balancing')
    inputs(config)
    candidates(config, 'rf')
    result = run_experiment(config, resume=args.resume)
    output = resolve_path(config['output_dir'])
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in config['source_paths']:
            archive.write(ROOT / name, name)
    (output / 'config.yaml').write_text(resolve_path(args.config).read_text(encoding='utf-8'), encoding='utf-8')
    _, metrics, _, proof = verify_run(output, config, kind='nested')
    write_csv(output / 'verified_metrics_by_subject.csv', metrics)
    write_json(output / 'verification.json', {**proof, 'scope': 'main', 'experiment_id': 'EXP-009'})
    print(f"EXP-009 {result['status']}: {proof['n_folds']} verified folds", flush=True)


if __name__ == '__main__':
    main()
