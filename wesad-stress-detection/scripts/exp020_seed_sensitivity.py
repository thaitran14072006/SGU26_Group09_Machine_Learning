"""Chạy bổ sung EXP020 với seed 43/44, giữ nguyên giao thức đã chốt."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    from wesad.utils import read_config, read_json, write_json, environment, file_hash
    from wesad.evaluation import run_experiment
    from exp020_train import verify
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[43, 44], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reference = ROOT / 'experiments/04_additional/EXP-020/results'
    config = read_config(reference / 'config.yaml')
    assert environment() == read_json(reference / 'environment.json'), 'Môi trường khác lượt gốc.'
    assert file_hash(ROOT / config['features']) == config['feature_sha256']
    config['seed'] = args.seed
    config['output_dir'] = f'experiments/04_additional/EXP-020/seed_sensitivity/20261120/seed_{args.seed}'
    config['source_paths'] = ['src/wesad/data.py', 'src/wesad/evaluation.py',
                              'src/wesad/models.py', 'src/wesad/utils.py',
                              'scripts/exp020_seed_sensitivity.py']
    out = ROOT / config['output_dir']
    if out.exists() and not args.resume:
        raise FileExistsError(f'Lượt này đã có: {out}. Dùng --resume nếu bị ngắt.')
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    write_json(out / 'run_note.json', {
        'experiment': 'EXP-020', 'member': 'như', 'date': '20/11/2026',
        'purpose': 'kiểm tra độ nhạy theo seed', 'seed': args.seed,
        'reference_seed': 42, 'comparison_seeds_declared_before_training': [42, 43, 44],
        'changed_model_setting': 'seed only', 'outer_folds': 30, 'inner_fits': 360,
        'interpretation': 'same 15 subjects; seeds are not independent subject samples; no best-seed selection',
        'python_executable': sys.executable,
    })
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in config['source_paths']:
            z.write(ROOT / p, p)
        z.write(out / 'config.yaml', 'config.yaml')
    print('output:', out, flush=True)
    run_experiment(config, resume=args.resume, outer_subjects=config['outer_subjects'])
    verify(out)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
