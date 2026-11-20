"""Audit EXP-009 evidence and prepare paired tables for the report notebook."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from wesad.g4 import inputs, verify_run
from wesad.g3 import verify_run as verify_baseline
from wesad.utils import read_config, read_json, write_csv, write_json, file_hash

EXPERIMENT = ROOT / 'experiments/02_tuning/Mai_20-11-2026-EXP009_RF'
METRICS = ['macro_f1', 'balanced_accuracy']
KEYS = ['task', 'subject_id']


def historical(config):
    base = ROOT / 'outputs/g4_group_tuning_complete_20261110'
    sources = [base / 'source_runs/run', base / 'source_runs/g4_rf_mlp_additional_outer11_20261110']
    predictions, metrics, params, proofs = [], [], {}, []
    for directory in sources:
        p, m, best, proof = verify_run(directory, config, kind='nested', allow_partial=True)
        predictions.append(p.loc[p.model.eq('rf')])
        metrics.append(m.loc[m.model.eq('rf')])
        params.update({k: v for k, v in best.items() if '/rf/' in k})
        proofs.append({'source': str(directory.relative_to(ROOT)), 'proof': proof})
    p, m = pd.concat(predictions, ignore_index=True), pd.concat(metrics, ignore_index=True)
    assert len(m) == 30 and len(params) == 30 and not m.duplicated(KEYS).any()
    assert not p.duplicated(['task', 'subject_id', 'window_id']).any()
    merged = pd.read_csv(base / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    merged = merged.loc[merged.model.eq('rf')]
    sort = ['task', 'subject_id', 'window_id']
    pd.testing.assert_frame_equal(p.sort_values(sort).reset_index(drop=True), merged.sort_values(sort).reset_index(drop=True))
    write_csv(EXPERIMENT / 'results_historical.csv', m)
    write_json(EXPERIMENT / 'best_params_historical.json', params)
    write_json(EXPERIMENT / 'historical_provenance.json', {
        'status': 'verified_historical_30_folds',
        'report_created': '2026-11-20', 'training_date_claim': 'Historical G4 sources; not a new run on 20/11.',
        'sources': proofs,
        'merged_predictions_sha256': file_hash(base / 'predictions.csv'),
    })
    return m, params


def prepare():
    config = read_config(EXPERIMENT / 'config.yaml')
    frame, columns = inputs(config)
    history, historical_params = historical(config)
    source = ROOT / config['output_dir']
    p, m, best, proof = verify_run(source, config, kind='nested')
    baseline_dir = ROOT / 'outputs/exp005_rf_full'
    baseline_config = read_config(baseline_dir / 'config.yaml')
    baseline_proof, baseline = verify_baseline(baseline_dir, baseline_config)
    assert baseline_config['feature_sha256'] == config['feature_sha256']
    assert len(m) == len(baseline) == 30 and len(p) == len(frame) * 2
    pd.testing.assert_frame_equal(
        baseline[KEYS + METRICS].sort_values(KEYS).reset_index(drop=True),
        pd.read_csv(ROOT / 'experiments/01_baselines/EXP-005_RF/results_rerun.csv')[KEYS + METRICS].sort_values(KEYS).reset_index(drop=True),
    )
    # Retain fold-level evidence so the final folder can be audited independently.
    for name in ['folds', 'config.json', 'environment.json', 'splits.json', 'source_snapshot.zip',
                 'best_params.json', 'summary.json', 'run.log', 'run_status.json', 'verification.json']:
        path = source / name
        if path.is_dir(): shutil.copytree(path, EXPERIMENT / name, dirs_exist_ok=True)
        else: shutil.copy2(path, EXPERIMENT / name)
    write_csv(EXPERIMENT / 'results.csv', m)
    write_csv(EXPERIMENT / 'metrics_by_subject.csv', m)
    write_csv(EXPERIMENT / 'predictions.csv', p)
    write_csv(EXPERIMENT / 'results_baseline.csv', baseline)
    paired = baseline[KEYS + METRICS].merge(m[KEYS + METRICS], on=KEYS, suffixes=('_baseline', '_tuned'), validate='one_to_one')
    assert len(paired) == 30
    for metric in METRICS:
        paired[metric + '_delta'] = paired[metric + '_tuned'] - paired[metric + '_baseline']
    write_csv(EXPERIMENT / 'comparison_exp005.csv', paired)
    repeat = history[KEYS + METRICS].merge(m[KEYS + METRICS], on=KEYS, suffixes=('_historical', '_rerun'), validate='one_to_one')
    for metric in METRICS:
        repeat[metric + '_delta'] = repeat[metric + '_rerun'] - repeat[metric + '_historical']
    write_csv(EXPERIMENT / 'comparison_historical.csv', repeat)
    rows = []
    for variant, data in [('baseline_EXP005', baseline), ('historical_G4', history), ('rerun_EXP009', m)]:
        for task, group in data.groupby('task'):
            r = {'variant': variant, 'task': task, 'n_subjects': len(group)}
            for metric in METRICS:
                r[metric + '_mean'] = float(group[metric].mean())
                r[metric + '_std'] = float(group[metric].std(ddof=1))
            r['fit_and_search_seconds_mean'] = float(group.seconds.mean())
            rows.append(r)
    summary = pd.DataFrame(rows)
    write_csv(EXPERIMENT / 'comparison_summary.csv', summary)
    cv_rows, params_rows, class_rows = [], [], []
    for task in config['tasks']:
        labels = [1, 2, 3] if task == 'multiclass' else [0, 1]
        part = p.loc[p.task.eq(task)]
        report = classification_report(part.y_true, part.y_pred, labels=labels, output_dict=True, zero_division=0)
        write_json(EXPERIMENT / f'confusion_matrix_{task}.json', {'labels': labels, 'matrix': confusion_matrix(part.y_true, part.y_pred, labels=labels).tolist()})
        for label in labels:
            class_rows.append({'task': task, 'label': label, **report[str(label)]})
        for subject in config['expected_subjects']:
            key = f'{task}/rf/{subject}'
            params_rows.append({'task': task, 'subject_id': subject, **best[key]})
            cv = pd.read_csv(source / f'folds/{task}/rf/{subject}/inner_cv.csv')
            cv.insert(0, 'subject_id', subject); cv.insert(0, 'task', task)
            cv_rows.append(cv)
    parameter_table = pd.DataFrame(params_rows)
    parameter_table['configuration'] = [json.dumps(best[f'{r.task}/rf/{r.subject_id}'], sort_keys=True) for r in parameter_table.itertuples()]
    write_csv(EXPERIMENT / 'best_params_by_subject.csv', parameter_table)
    write_csv(EXPERIMENT / 'candidate_frequency.csv', parameter_table.groupby(['task', 'configuration'], dropna=False).size().reset_index(name='n_subjects'))
    write_csv(EXPERIMENT / 'inner_search_all.csv', pd.concat(cv_rows, ignore_index=True))
    write_csv(EXPERIMENT / 'classification_report.csv', pd.DataFrame(class_rows))
    differences = repeat[[metric + '_delta' for metric in METRICS]].abs().to_numpy()
    provenance = {
        'experiment_id': 'EXP-009', 'author': 'Mai', 'report_date': '2026-11-20',
        'status': 'verified_new_rerun_and_historical_analysis',
        'training_source': config['output_dir'], 'training_proof': proof,
        'baseline_source': str(baseline_dir.relative_to(ROOT)), 'baseline_proof': baseline_proof,
        'data': {'n_rows': len(frame), 'n_features': len(columns), 'n_subjects': frame.subject_id.nunique(),
                 'sha256': config['feature_sha256'], 'schema_sha256': file_hash(ROOT / config['schema'])},
        'historical_comparison': {'n_folds': len(repeat), 'max_abs_metric_difference': float(differences.max()),
            'n_changed_metric_folds': int((differences > 1e-10).any(axis=1).sum()),
            'n_changed_parameter_folds': sum(best[k] != historical_params[k] for k in best)},
        'report_script_sha256': file_hash(Path(__file__)),
        'execution': {
            'method': 'Three processes on disjoint outer subjects, identical config/signature; final sequential resume and audit.',
            'outer_subject_partitions': [
                ['S10', 'S14', 'S2', 'S5', 'S8'],
                ['S11', 'S15', 'S3', 'S6', 'S9'],
                ['S13', 'S16', 'S17', 'S4', 'S7'],
            ],
            'worker_command': '.venv-g1/Scripts/python.exe -X utf8 scripts/run_experiment.py --config experiments/02_tuning/Mai_20-11-2026-EXP009_RF/config.yaml --outer-subjects <partition> --resume',
            'final_command': '.venv-g1/Scripts/python.exe -X utf8 scripts/run_exp009_rf.py --resume',
        },
        'limitations': ['No statistical significance test.', 'Search/fit timings are not prediction latency.',
                       'No final deployment model selected or trained by this experiment.',
                       'git_commit is unavailable when the local checkout has no usable Git metadata.'],
    }
    write_json(EXPERIMENT / 'provenance.json', provenance)
    # Repeat audit on the delivered copy, not just on transient output files.
    _, delivered, _, _ = verify_run(EXPERIMENT, config, kind='nested')
    assert len(delivered) == 30
    print(summary.to_string(index=False))
    print('Delivered evidence verified:', len(delivered), 'folds;', len(p), 'predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-only', action='store_true')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.destination:
        EXPERIMENT = args.destination.resolve()
        if not EXPERIMENT.is_relative_to(ROOT / 'outputs'):
            raise ValueError('Report destination must be inside outputs/')
    if args.historical_only:
        historical(read_config(EXPERIMENT / 'config.yaml'))
        print('Historical RF: verified 30 folds')
    else:
        prepare()
