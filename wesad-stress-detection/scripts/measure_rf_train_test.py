"""Refit historical RF folds to add train scores, checking saved test predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from wesad.data import load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from wesad.utils import environment, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    source = ROOT / args.source
    destination = ROOT / args.destination
    manifest = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    verified = json.loads((source / 'verification.json').read_text(encoding='utf-8'))
    config = manifest['settings']
    assert verified['status'] == 'passed' and verified['n_folds'] == 30
    assert config['models'] == ['rf'] and not config['nested']
    current = environment()
    assert current['packages'] == manifest['environment']['packages'], 'Use the original training environment'
    defaults = read_config(ROOT / 'configs/defaults.yaml')
    table, schema = ROOT / defaults['features'], ROOT / defaults['schema']
    assert hashlib.sha256(table.read_bytes()).hexdigest() == manifest['data_sha256'] == config['feature_sha256']
    frame, columns = load_feature_table(table, schema, config['expected_subjects'])
    assert columns == manifest['feature_columns']
    X = frame[columns].astype(float)
    historical = pd.read_csv(source / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    historical = historical.loc[historical.model.eq('rf')]
    assert not historical.duplicated(['task', 'subject_id', 'window_id']).any()
    rows, matched = [], 0
    for task in ['binary', 'multiclass']:
        y, labels = task_target(frame, task)
        for subject in config['expected_subjects']:
            train = frame.subject_id.ne(subject)
            test = ~train
            assert frame.loc[train, 'subject_id'].nunique() == 14
            fold = source / 'folds' / task / 'rf' / subject
            done = json.loads((fold / 'complete.json').read_text(encoding='utf-8'))
            assert done['signature'] == manifest['signature']
            for name, digest in done['files'].items():
                assert hashlib.sha256((fold / name).read_bytes()).hexdigest() == digest
            details = json.loads((fold / 'details.json').read_text(encoding='utf-8'))
            params = details['best_params']
            model = build_pipeline('rf', config)
            model.set_params(**params)
            with threadpool_limits(limits=1):
                model.fit(X.loc[train], y.loc[train])
                train_pred = model.predict(X.loc[train])
                test_pred = model.predict(X.loc[test])
            actual = frame.loc[test, ['subject_id', 'window_id']].copy()
            actual['y_true'], actual['y_pred'] = y.loc[test].to_numpy(), test_pred
            saved = historical.loc[historical.task.eq(task) & historical.subject_id.eq(subject)]
            paired = actual.merge(saved, on=['subject_id', 'window_id'], suffixes=('_new', '_old'), validate='one_to_one')
            assert len(paired) == len(actual) == len(saved)
            for name in ['y_true', 'y_pred']:
                assert np.array_equal(paired[name + '_new'], paired[name + '_old']), f'Historical mismatch: {task}/{subject}'
            matched += len(paired)
            row = {'task': task, 'subject_id': subject, 'variant': 'balanced' if config['balance'] else 'baseline',
                   'n_train': int(train.sum()), 'n_test': int(test.sum())}
            for split, truth, pred in [('train', y.loc[train], train_pred), ('test', y.loc[test], test_pred)]:
                row[f'{split}_macro_f1'] = f1_score(truth, pred, average='macro', labels=labels, zero_division=0)
                row[f'{split}_balanced_accuracy'] = balanced_accuracy_score(truth, pred)
            for metric in ['macro_f1', 'balanced_accuracy']:
                row[f'gap_{metric}'] = row[f'train_{metric}'] - row[f'test_{metric}']
            rows.append(row)
            print(f'{task}/{subject}: test predictions match', flush=True)
    assert len(rows) == 30 and matched == len(historical) == 12378
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / 'train_test_metrics.csv', index=False)
    audit = {'status': 'passed', 'n_folds': 30, 'matched_test_predictions': matched,
             'source': source.relative_to(ROOT).as_posix(), 'historical_signature': manifest['signature'],
             'method': 'refit_historical_params_and_verify_test_predictions', 'environment': current}
    (destination / 'train_test_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified 30 folds; train/test metrics saved.', flush=True)


if __name__ == '__main__':
    main()
