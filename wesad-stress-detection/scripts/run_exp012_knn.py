"""Compare fixed EXP-007 KNN parameters with/without train-only oversampling."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from wesad.data import load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from wesad.utils import environment, file_hash, fingerprint, read_config


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_inputs(config):
    reference = ROOT / config['reference_dir']
    files = {name: reference / name for name in ['config.yaml', 'best_params_by_subject.csv', 'predictions.csv', 'results.csv', 'result_provenance.json']}
    ref_config = read_config(files['config.yaml'])
    provenance = json.loads(files['result_provenance.json'].read_text(encoding='utf-8'))
    assert ref_config['models'] == ['knn'] and ref_config['nested'] and not ref_config['balance']
    assert provenance['scope'] == 'main complete'
    assert ref_config['feature_sha256'] == provenance['data_sha256'] == config['feature_sha256']
    assert ref_config['seed'] == config['seed']
    table, schema = ROOT / config['features'], ROOT / config['schema']
    assert file_hash(table) == config['feature_sha256']
    frame, columns = load_feature_table(table, schema, config['expected_subjects'])
    assert len(columns) == 68 and set(frame.subject_id) == set(ref_config['expected_subjects'])
    params = pd.read_csv(files['best_params_by_subject.csv'], dtype={'subject_id': str})
    scores = pd.read_csv(files['results.csv'], dtype={'subject_id': str})
    predictions = pd.read_csv(files['predictions.csv'], dtype={'subject_id': str, 'window_id': str})
    expected = {(task, subject) for task in config['tasks'] for subject in config['expected_subjects']}
    for data in [params, scores]:
        assert len(data) == 30 and not data.duplicated(['task', 'subject_id']).any()
        assert set(zip(data.task, data.subject_id)) == expected
    assert set(predictions.model) == set(scores.model) == {'knn'}
    assert not predictions.duplicated(['task', 'subject_id', 'window_id']).any()
    assert len(predictions) == len(frame) * 2
    allowed = {(item['model__n_neighbors'][0], item['model__metric'][0], item['model__weights'][0]) for item in ref_config['search_spaces']['knn']}
    for row in params.itertuples():
        assert (row.n_neighbors, row.metric, row.weights) in allowed
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        for subject in config['expected_subjects']:
            truth = frame.loc[frame.subject_id.eq(subject), ['subject_id', 'window_id']].copy()
            truth['expected_y'] = y.loc[frame.subject_id.eq(subject)].to_numpy()
            saved = predictions.loc[predictions.task.eq(task) & predictions.subject_id.eq(subject)]
            joined = truth.merge(saved, on=['subject_id', 'window_id'], validate='one_to_one')
            assert len(joined) == len(truth) == len(saved)
            assert np.array_equal(joined.expected_y, joined.y_true)
            original = scores.loc[scores.task.eq(task) & scores.subject_id.eq(subject)].iloc[0]
            assert np.isclose(f1_score(saved.y_true, saved.y_pred, labels=labels, average='macro', zero_division=0), original.macro_f1, atol=1e-12, rtol=0)
            assert np.isclose(balanced_accuracy_score(saved.y_true, saved.y_pred), original.balanced_accuracy, atol=1e-12, rtol=0)
    return frame, columns, params, predictions, files


def run(config_path, quick_subject=None, validate_only=False):
    config_path = Path(config_path).resolve()
    cfg = read_config(config_path)
    assert cfg['experiment_id'] == 'EXP-012' and cfg['sampler'] == 'RandomOverSampler'
    assert cfg['tasks'] == ['binary', 'multiclass'] and cfg['sampling_strategy'] == 'auto'
    frame, columns, params, historical, reference_files = load_inputs(cfg)
    if validate_only:
        print('PASS: data, 30 parameter sets and all EXP-007 test metrics verified.')
        return
    subjects = cfg['expected_subjects']
    if quick_subject:
        assert quick_subject in subjects
        subjects = [quick_subject]
    output = ROOT / cfg['output_dir']
    if quick_subject:
        output = output / ('quick_' + quick_subject)
    identity = {'config': cfg, 'data_sha256': file_hash(ROOT / cfg['features']),
                'schema_sha256': file_hash(ROOT / cfg['schema']),
                'reference_sha256': {name: file_hash(path) for name, path in reference_files.items()},
                'source_sha256': {name: file_hash(ROOT / name) for name in cfg['source_paths']},
                'environment': environment(), 'scope': 'quick' if quick_subject else 'main', 'subjects': subjects}
    signature = fingerprint(identity)
    manifest = output / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))['signature'] == signature, 'Settings changed. Choose a new output_dir.'
    else:
        # A main run may share its parent with independent quick-subject runs.
        # Do not accept unrelated files or incomplete quick-run directories.
        unexpected = []
        for child in output.iterdir() if output.exists() else []:
            quick_manifest = child / 'manifest.json'
            allowed = child.is_dir() and child.name in {'quick_' + s for s in cfg['expected_subjects']} and quick_manifest.is_file()
            if allowed:
                quick_identity = json.loads(quick_manifest.read_text(encoding='utf-8'))
                allowed = quick_identity.get('scope') == 'quick' and quick_identity.get('config') == cfg
            if not allowed:
                unexpected.append(child.name)
        if unexpected:
            raise ValueError('Output contains files without a main manifest: ' + ', '.join(unexpected))
        save_json(manifest, {**identity, 'signature': signature, 'run_date': datetime.now(timezone(timedelta(hours=7))).isoformat()})
    X = frame[columns].astype(float)
    all_scores, all_predictions, all_counts = [], [], []
    total_reference_changes = 0
    for task in cfg['tasks']:
        y, labels = task_target(frame, task)
        for subject in subjects:
            directory = output / 'folds' / task / subject
            complete = directory / 'complete.json'
            reused = False
            if complete.exists():
                done = json.loads(complete.read_text(encoding='utf-8'))
                reused = done['signature'] == signature and all((directory / name).exists() and file_hash(directory / name) == digest for name, digest in done['files'].items())
            if not reused:
                train, test = frame.subject_id.ne(subject), frame.subject_id.eq(subject)
                assert frame.loc[train, 'subject_id'].nunique() == 14 and not set(frame.loc[train, 'subject_id']) & set(frame.loc[test, 'subject_id'])
                selected = params.loc[params.task.eq(task) & params.subject_id.eq(subject)].iloc[0]
                fixed = {'model__n_neighbors': int(selected.n_neighbors), 'model__metric': selected.metric, 'model__weights': selected.weights}
                rows, prediction_rows, count_rows, caught_warnings = [], [], [], []
                reference_changes = 0
                for variant, balanced in [('plain', False), ('balanced', True)]:
                    pipeline = build_pipeline('knn', {'seed': cfg['seed'], 'balance': balanced})
                    pipeline.set_params(**fixed)
                    started = perf_counter()
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                        warnings.simplefilter('always')
                        pipeline.fit(X.loc[train], y.loc[train])
                        train_pred = pipeline.predict(X.loc[train])
                        test_pred = pipeline.predict(X.loc[test])
                    caught_warnings.extend(str(w.message) for w in caught)
                    row = {'task': task, 'subject_id': subject, 'variant': variant, 'n_train': int(train.sum()), 'n_test': int(test.sum()), 'seconds': perf_counter() - started, **fixed}
                    for split, truth, pred in [('train', y.loc[train], train_pred), ('test', y.loc[test], test_pred)]:
                        row[('' if split == 'test' else 'train_') + 'accuracy'] = accuracy_score(truth, pred)
                        row[('' if split == 'test' else 'train_') + 'macro_f1'] = f1_score(truth, pred, labels=labels, average='macro', zero_division=0)
                        row[('' if split == 'test' else 'train_') + 'balanced_accuracy'] = balanced_accuracy_score(truth, pred)
                    rows.append(row)
                    pred = frame.loc[test, ['subject_id', 'window_id']].copy()
                    pred['task'], pred['variant'] = task, variant
                    pred['y_true'], pred['y_pred'] = y.loc[test].to_numpy(), test_pred
                    prediction_rows.append(pred)
                    if not balanced:
                        old = historical.loc[historical.task.eq(task) & historical.subject_id.eq(subject)]
                        joined = pred.merge(old, on=['task', 'subject_id', 'window_id'], suffixes=('_new', '_old'), validate='one_to_one')
                        assert len(joined) == len(old) == len(pred)
                        reference_changes = int(joined.y_pred_new.ne(joined.y_pred_old).sum())
                    before = y.loc[train].value_counts()
                    sampled_y = y.loc[train].iloc[pipeline['sampler'].sample_indices_] if balanced else y.loc[train]
                    after = sampled_y.value_counts()
                    assert not balanced or after.nunique() == 1
                    row['n_train_resampled'] = len(sampled_y)
                    for label in labels:
                        count_rows.append({'task': task, 'subject_id': subject, 'variant': variant, 'label': label, 'before': int(before.get(label, 0)), 'after': int(after.get(label, 0))})
                directory.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(directory / 'results.csv', index=False)
                pd.concat(prediction_rows, ignore_index=True).to_csv(directory / 'predictions.csv', index=False)
                pd.DataFrame(count_rows).to_csv(directory / 'class_counts.csv', index=False)
                save_json(directory / 'details.json', {'params_from': 'EXP-007', 'fixed_params': fixed, 'train_subjects': sorted(set(frame.loc[train, 'subject_id'])), 'test_subject': subject, 'plain_prediction_changes_vs_exp007': reference_changes, 'warnings': sorted(set(caught_warnings))})
                files = ['results.csv', 'predictions.csv', 'class_counts.csv', 'details.json']
                save_json(complete, {'signature': signature, 'files': {name: file_hash(directory / name) for name in files}})
            details = json.loads((directory / 'details.json').read_text(encoding='utf-8'))
            total_reference_changes += details['plain_prediction_changes_vs_exp007']
            all_scores.append(pd.read_csv(directory / 'results.csv'))
            all_predictions.append(pd.read_csv(directory / 'predictions.csv', dtype={'subject_id': str, 'window_id': str}))
            all_counts.append(pd.read_csv(directory / 'class_counts.csv'))
            print(f'{task}/{subject}: {"reused" if reused else "completed"} plain + balanced', flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    assert len(scores) == len(subjects) * 4
    assert not scores.duplicated(['task', 'subject_id', 'variant']).any()
    assert not predictions.duplicated(['task', 'subject_id', 'window_id', 'variant']).any()
    for row in scores.itertuples():
        pred = predictions.loc[predictions.task.eq(row.task) & predictions.subject_id.eq(row.subject_id) & predictions.variant.eq(row.variant)]
        labels = [0, 1] if row.task == 'binary' else [1, 2, 3]
        assert len(pred) == row.n_test
        for value, actual in [(row.accuracy, accuracy_score(pred.y_true, pred.y_pred)), (row.macro_f1, f1_score(pred.y_true, pred.y_pred, average='macro', labels=labels, zero_division=0)), (row.balanced_accuracy, balanced_accuracy_score(pred.y_true, pred.y_pred))]:
            assert np.isclose(value, actual, atol=1e-12, rtol=0)
    plain = scores.loc[scores.variant.eq('plain')]
    balanced = scores.loc[scores.variant.eq('balanced')]
    comparison = plain.merge(balanced, on=['task', 'subject_id'], suffixes=('_plain', '_balanced'), validate='one_to_one')
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        comparison[metric + '_delta'] = comparison[metric + '_balanced'] - comparison[metric + '_plain']
    metrics = ['accuracy', 'macro_f1', 'balanced_accuracy', 'train_accuracy', 'train_macro_f1', 'train_balanced_accuracy']
    summary = scores.groupby(['task', 'variant'])[metrics].agg(['mean', 'std'])
    scores.to_csv(output / 'results.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    comparison.to_csv(output / 'comparison_exp007.csv', index=False)
    summary.to_csv(output / 'summary.csv')
    pd.concat(all_counts, ignore_index=True).to_csv(output / 'class_counts.csv', index=False)
    proof = {'status': 'passed', 'scope': identity['scope'], 'outer_folds': len(subjects) * 2,
             'paired_result_rows': len(scores), 'n_predictions': len(predictions), 'signature': signature,
             'plain_prediction_changes_vs_exp007': total_reference_changes,
             'comparison_scope': 'paired refits in current environment; fixed EXP-007 parameters',
             'sampling_scope': 'outer train only, after median imputation and scaling',
             'train_metric_scope': 'original outer-training rows, before duplication'}
    save_json(output / 'verification.json', proof)
    if not quick_subject:
        destination = ROOT / cfg['export_dir']
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['results.csv', 'predictions.csv', 'comparison_exp007.csv', 'summary.csv', 'class_counts.csv', 'verification.json']:
            (destination / name).write_bytes((output / name).read_bytes())
        exported = params.copy()
        exported.to_csv(destination / 'best_params_by_subject.csv', index=False)
    print(f'PASS: {proof["outer_folds"]} outer folds; historical plain prediction changes={total_reference_changes}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/03_balancing/EXP-012_KNN/config.yaml')
    parser.add_argument('--quick-subject')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.quick_subject, args.validate_only)
