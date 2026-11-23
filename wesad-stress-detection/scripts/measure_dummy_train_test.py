"""Add Dummy train/test scores while verifying the saved test predictions."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from wesad.data import load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from wesad.utils import file_hash, read_config


def main():
    package = ROOT / 'experiments/01_baselines/EXP-001_Dummy'
    config = read_config(package / 'config.yaml')
    source = ROOT / config['output_dir']
    manifest = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    verified = json.loads((source / 'verification.json').read_text(encoding='utf-8'))
    assert verified['status'] == 'passed' and verified['n_folds'] == 30
    assert config['models'] == ['dummy'] and not config['balance'] and not config['nested']
    assert file_hash(ROOT / config['features']) == manifest['data_sha256'] == config['feature_sha256']
    frame, columns = load_feature_table(ROOT / config['features'], ROOT / config['schema'], config['expected_subjects'])
    assert len(columns) == 68 and columns == manifest['feature_columns']
    X = frame[columns].astype(float)
    saved = pd.read_csv(source / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    assert set(saved.model) == {'dummy'}
    assert not saved.duplicated(['task', 'subject_id', 'window_id']).any()
    old_scores = pd.read_csv(package / 'results.csv')
    assert len(old_scores) == 30 and not old_scores.duplicated(['task', 'subject_id']).any()
    records, label_counts, matched = [], [], 0
    for task in ['binary', 'multiclass']:
        y, labels = task_target(frame, task)
        for subject in config['expected_subjects']:
            train, test = frame.subject_id.ne(subject), frame.subject_id.eq(subject)
            assert frame.loc[train, 'subject_id'].nunique() == 14
            pipeline = build_pipeline('dummy', config)
            assert pipeline['model'].strategy == 'most_frequent'
            pipeline.fit(X.loc[train], y.loc[train])
            train_pred, test_pred = pipeline.predict(X.loc[train]), pipeline.predict(X.loc[test])
            actual = frame.loc[test, ['subject_id', 'window_id']].copy()
            actual['y_true'], actual['y_pred'] = y.loc[test].to_numpy(), test_pred
            original = saved.loc[saved.task.eq(task) & saved.subject_id.eq(subject)]
            pair = actual.merge(original, on=['subject_id', 'window_id'], suffixes=('_new', '_old'), validate='one_to_one')
            assert len(pair) == len(actual) == len(original)
            assert np.array_equal(pair.y_true_new, pair.y_true_old) and np.array_equal(pair.y_pred_new, pair.y_pred_old)
            matched += len(pair)
            record = {'task': task, 'subject_id': subject, 'n_train': int(train.sum()), 'n_test': int(test.sum()), 'predicted_class': int(test_pred[0])}
            for split, truth, pred in [('train', y.loc[train], train_pred), ('test', y.loc[test], test_pred)]:
                record[f'{split}_accuracy'] = accuracy_score(truth, pred)
                record[f'{split}_macro_f1'] = f1_score(truth, pred, labels=labels, average='macro', zero_division=0)
                record[f'{split}_balanced_accuracy'] = balanced_accuracy_score(truth, pred)
                counts = truth.value_counts()
                for label in labels:
                    label_counts.append({'task': task, 'subject_id': subject, 'split': split, 'label': label, 'count': int(counts.get(label, 0))})
            previous = old_scores.loc[old_scores.task.eq(task) & old_scores.subject_id.eq(subject)].iloc[0]
            for metric in ['macro_f1', 'balanced_accuracy']:
                assert np.isclose(record['test_' + metric], previous[metric], atol=1e-12, rtol=0)
                record['gap_' + metric] = record['train_' + metric] - record['test_' + metric]
            records.append(record)
    assert len(records) == 30 and matched == len(saved) == 12378
    result = pd.DataFrame(records)
    result.to_csv(package / 'train_test_metrics.csv', index=False)
    pd.DataFrame(label_counts).to_csv(package / 'train_test_class_counts.csv', index=False)
    columns = [f'{split}_{metric}' for metric in ['accuracy', 'macro_f1', 'balanced_accuracy'] for split in ['train', 'test']]
    result.groupby('task')[columns].agg(['mean', 'std']).to_csv(package / 'train_test_summary.csv')
    proof = {'status': 'passed', 'n_folds': 30, 'matched_test_predictions': matched,
             'method': 'fit_same_most_frequent_dummy_and_match_historical_test_predictions',
             'computed_at': datetime.now(timezone(timedelta(hours=7))).isoformat(),
             'source': source.relative_to(ROOT).as_posix(), 'historical_signature': manifest['signature'],
             'data_sha256': file_hash(ROOT / config['features'])}
    (package / 'train_test_verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: Dummy train/test scores for 30 folds; all 12,378 historical test predictions match.')


if __name__ == '__main__':
    main()
