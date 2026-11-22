"""Chạy bổ sung EXP020 với seed 47–50, giữ nguyên giao thức đã chốt."""
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
    parser.add_argument('--seed', type=int, choices=[47, 48, 49, 50], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reference = ROOT / 'experiments/04_additional/EXP-020/results'
    config = read_config(reference / 'config.yaml')
    assert environment() == read_json(reference / 'environment.json'), 'Môi trường khác lượt gốc.'
    assert file_hash(ROOT / config['features']) == config['feature_sha256']
    config['seed'] = args.seed
    config['output_dir'] = f'experiments/04_additional/EXP-020/seed_sensitivity/20261122_r2/seed_{args.seed}'
    config['source_paths'] = ['src/wesad/data.py', 'src/wesad/evaluation.py',
                              'src/wesad/models.py', 'src/wesad/utils.py',
                              'scripts/exp020_batch_r2.py']
    out = ROOT / config['output_dir']
    if out.exists() and not args.resume:
        raise FileExistsError(f'Lượt này đã có: {out}. Dùng --resume nếu bị ngắt.')
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    write_json(out / 'run_note.json', {
        'experiment': 'EXP-020', 'member': 'như', 'date': '22/11/2026',
        'purpose': 'kiểm tra độ nhạy theo seed', 'seed': args.seed,
        'reference_seed': 42, 'comparison_seeds_declared_before_training': [42, 43, 44, 45, 46, 47, 48, 49, 50],
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
    import traceback
    seed = sys.argv[sys.argv.index('--seed')+1]
    logdir=ROOT/f'experiments/04_additional/EXP-020/seed_sensitivity/20261122_r2/seed_{seed}'
    logdir.mkdir(parents=True,exist_ok=True)
    # Let the normal guard create the run; write events to a sibling journal.
    journal=logdir.parent/f'seed_{seed}_journal.jsonl'
    logdir.rmdir() if not any(logdir.iterdir()) else None
    def event(status,note):
        with journal.open('a',encoding='utf8') as f:
            f.write(json.dumps({'date':'22/11/2026','member':'như','seed':int(seed),'status':status,'note':note},ensure_ascii=False)+'\n')
    event('bắt đầu','chạy đủ 15 người, 2 bài toán')
    try:
        main()
    except Exception as error:
        event('lỗi',str(error))
        (journal.parent/f'seed_{seed}_error.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
    else:
        event('xong','đủ 30 fold, kiểm tra đạt')
