"""Reconstruct selected outer-fold RF fits and verify historical test predictions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from wesad.data import load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from wesad.utils import environment


def main():
    package = Path(__file__).resolve().parent
    report = ROOT / 'outputs/exp009_rf_mai_20261120_report'
    proof = json.loads((report / 'verification.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'passed' and proof['n_folds'] == 30
    config = proof['producer']['settings']
    feature_path = ROOT / 'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
    assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == proof['producer']['data_sha256']
    historical_environment = proof['producer']['environment']
    current = environment()
    assert current['packages'] == historical_environment['packages'], 'Use the original .venv-g1 environment'
    frame, columns = load_feature_table(feature_path, expected_subjects=config['expected_subjects'])
    X = frame[columns].astype(float)
    rows = []
    matched = 0
    for task in ['binary', 'multiclass']:
        y, labels = task_target(frame, task)
        for subject in config['expected_subjects']:
            folder = report / 'folds' / task / 'rf' / subject
            done = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            assert done['signature'] == proof['producer']['signature']
            for name, digest in done['files'].items():
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            params = json.loads((folder / 'details.json').read_text(encoding='utf-8'))['best_params']
            train = frame.subject_id.ne(subject)
            test = ~train
            assert frame.loc[train, 'subject_id'].nunique() == 14
            model = build_pipeline('rf', config)
            model.set_params(**params)
            with threadpool_limits(limits=1):
                model.fit(X.loc[train], y.loc[train])
                train_pred = model.predict(X.loc[train])
                test_pred = model.predict(X.loc[test])
            saved = pd.read_csv(folder / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
            actual = frame.loc[test, ['subject_id', 'window_id']].copy()
            actual['y_true'] = y.loc[test].to_numpy()
            actual['y_pred'] = test_pred
            paired = actual.merge(saved, on=['subject_id', 'window_id'], suffixes=('_new', '_old'), validate='one_to_one')
            assert len(paired) == len(actual) == len(saved)
            for name in ['y_true', 'y_pred']:
                assert np.array_equal(paired[name + '_new'], paired[name + '_old']), f'Historical prediction mismatch: {task}/{subject}'
            matched += len(paired)
            row = {'task': task, 'subject_id': subject, 'n_train': int(train.sum()), 'n_test': int(test.sum()), 'best_params': json.dumps(params, sort_keys=True)}
            for split, truth, pred in [('train', y.loc[train], train_pred), ('test', y.loc[test], test_pred)]:
                row[split + '_macro_f1'] = f1_score(truth, pred, average='macro', labels=labels, zero_division=0)
                row[split + '_balanced_accuracy'] = balanced_accuracy_score(truth, pred)
            for metric in ['macro_f1', 'balanced_accuracy']:
                row['gap_' + metric] = row['train_' + metric] - row['test_' + metric]
            rows.append(row)
            print(f'{task}/{subject}: test predictions match; train F1={row["train_macro_f1"]:.4f}, test F1={row["test_macro_f1"]:.4f}', flush=True)
    assert len(rows) == 30 and matched == proof['n_predictions']
    result = pd.DataFrame(rows)
    result.to_csv(package / 'train_test_selected_folds.csv', index=False)
    audit = {'status': 'passed', 'method': 'refit_saved_best_params_without_repeating_inner_search', 'historical_run_date': config['run_date'], 'historical_signature': proof['producer']['signature'], 'n_folds': 30, 'matched_test_predictions': matched, 'environment': current}
    (package / 'train_test_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: all historical predictions match; wrote train/test metrics.', flush=True)


if __name__ == '__main__':
    main()
