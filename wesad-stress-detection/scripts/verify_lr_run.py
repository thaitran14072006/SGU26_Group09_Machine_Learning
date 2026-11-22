"""Recompute LR run metrics and verify data, splits and frozen dependencies."""
import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wesad.data import SUBJECTS, load_feature_table
from wesad.lr_workflow import (folder, identity, plan, prerequisites,
                               validate_completed)
from wesad.utils import file_hash, fingerprint, read_config


def verify(run_id):
    cfg = plan()
    spec = next(s for s in cfg['experiments'] if s['run_id'] == run_id)
    receipt = validate_completed(spec)
    dependency = prerequisites(spec['stage'], cfg)
    assert receipt['signature'] == fingerprint(identity(spec, cfg, dependency))
    defaults = read_config(ROOT / 'configs/defaults.yaml')
    assert file_hash(ROOT / defaults['features']) == cfg['feature_sha256']
    frame, features = load_feature_table(ROOT / defaults['features'],
                                        ROOT / defaults['schema'], SUBJECTS)
    out = folder(spec)
    metrics = pd.read_csv(out / 'results.csv')
    pred = pd.read_csv(out / 'predictions.csv', dtype={'window_id': str})
    details = json.loads((out / 'fold_details.json').read_text(encoding='utf8'))
    summary = pd.read_csv(out / 'summary.csv').set_index('task')
    notebook = json.loads((out / 'experiment.ipynb').read_text(encoding='utf8'))
    assert notebook['nbformat'] == 4 and notebook['cells']
    for cell in notebook['cells']:
        assert cell['cell_type'] in {'markdown', 'code', 'raw'}
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert len(metrics) == 30 and len(pred) == 12378 and len(details) == 30
    assert set(metrics.task) == set(pred.task) == {'binary', 'multiclass'}
    assert not metrics.duplicated(['task', 'subject_id']).any()
    assert not pred.duplicated(['task', 'subject_id', 'window_id']).any()
    for task in ('multiclass', 'binary'):
        labels = [1, 2, 3] if task == 'multiclass' else [0, 1]
        part = pred[pred.task == task]
        expected = frame[['subject_id', 'window_id', 'label_original']].copy()
        expected['window_id'] = expected.window_id.astype(str)
        merged = expected.merge(part, on=['subject_id', 'window_id'],
                                how='outer', validate='one_to_one', indicator=True)
        assert len(merged) == 6189 and merged['_merge'].eq('both').all()
        truth = (merged.label_original == 2).astype(int) if task == 'binary' else merged.label_original
        assert np.array_equal(truth, merged.y_true) and merged.y_pred.isin(labels).all()
        assert set(metrics.loc[metrics.task == task, 'subject_id']) == set(SUBJECTS)
        for subject in SUBJECTS:
            p = part[part.subject_id == subject]
            m = metrics[(metrics.task == task) & (metrics.subject_id == subject)].iloc[0]
            d = details[f'{task}/{subject}']
            assert len(p) == m.n_test
            assert np.isclose(m.macro_f1, f1_score(p.y_true, p.y_pred, labels=labels,
                                                  average='macro', zero_division=0))
            assert np.isclose(m.balanced_accuracy, balanced_accuracy_score(p.y_true, p.y_pred))
            assert d['confusion_matrix'] == confusion_matrix(p.y_true, p.y_pred, labels=labels).tolist()
            assert d['outer_test_subjects'] == [subject]
            assert set(d['outer_train_subjects']) == set(SUBJECTS) - {subject}
            assert not d['fatal_warnings'] and len(d['inner_splits']) == 3
            seen = []
            for split in d['inner_splits']:
                a, b = set(split['train_subjects']), set(split['validation_subjects'])
                assert not a & b and subject not in a | b
                assert a | b == set(SUBJECTS) - {subject}
                seen.extend(b)
            assert len(seen) == 14 and set(seen) == set(SUBJECTS) - {subject}
            assert len(d['selected_features']) == m.selected_k
            assert set(d['selected_features']) <= set(features)
            candidates = d['inner_candidates']
            assert all(np.isfinite(c['mean_macro_f1']) for c in candidates)
            assert np.isclose(max(c['mean_macro_f1'] for c in candidates), d['inner_macro_f1'])
            if dependency:
                recipe = dependency['recipes'][f'{task}/{subject}']
                assert d['inner_splits'] == json.loads((folder(next(
                    s for s in cfg['experiments'] if s['run_id'] == recipe['source_run'])) /
                    'fold_details.json').read_text(encoding='utf8'))[f'{task}/{subject}']['inner_splits']
                assert d['method'] == recipe['method']
                assert d['selection_params'] == recipe['selection_params']
                assert len(candidates) == 1
                assert d['weight_policy'] == spec.get('weight_policy', recipe['weight_policy'])
                assert d['C'] == spec.get('C', 1.0)
            else:
                assert len(candidates) == len(spec['candidates'])
                assert max(candidates, key=lambda c: c['mean_macro_f1'])['params'] == d['selection_params']
        for metric in ('macro_f1', 'balanced_accuracy'):
            values = metrics.loc[metrics.task == task, metric]
            assert np.isclose(summary.loc[task, metric + '_mean'], values.mean())
            assert np.isclose(summary.loc[task, metric + '_std'], values.std(ddof=1))
    return {'run_id': run_id, 'status': 'passed', 'folds': len(metrics),
            'oof_predictions': len(pred), 'acceptance_sha256': file_hash(out / 'acceptance.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))
