"""Basic RF/MLP nested tuning on the accepted G2 table; separate from G3."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid

from wesad.data import SUBJECTS, load_feature_table
from wesad.evaluation import run_experiment, task_target
from wesad.utils import (PROJECT_ROOT, file_hash, read_json, resolve_path,
                         write_csv, write_json)


def require(value, message):
    if not value:
        raise ValueError(message)


def preflight(config):
    require(config.get('protocol') == 'g4_nhu_nested_physiological68_v1', 'Wrong G4 protocol')
    require(config.get('nested') is True and config.get('inner_splits') == 3, 'Use nested GroupKFold(3)')
    require(config['models'] and set(config['models']) <= {'rf', 'mlp'}
            and len(config['models']) == len(set(config['models']))
            and config['tasks'] == ['multiclass', 'binary'], 'Run assigned RF/MLP models, both tasks')
    require(config.get('outer_subjects') == ['S4','S9','S11','S17'], 'Use assigned outer subjects S4/S9/S11/S17')
    require(config.get('seed') == 42 and config.get('normalization') == 'global_train', 'Wrong seed/normalization')
    require(config.get('sensor_set') == 'combined_physiological68', 'Use combined physiological68')
    require(not config.get('balance') and not config.get('model_factories')
            and not config.get('model_params') and config.get('variance_threshold') is None,
            'Keep tuning separate from balancing/selection or G3 factories')
    require(config['expected_subjects'] == list(SUBJECTS), 'All 15 subjects required')
    schema = read_json(resolve_path(config['schema']))
    reference = read_json(PROJECT_ROOT / 'configs/feature_schema.json')
    require(schema['feature_version'] == config['feature_version'] == 'physiological_68_v1'
            and schema['feature_columns'] == reference['feature_columns']
            and schema['sensor_groups'] == reference['sensor_groups'], 'Wrong schema')
    require(config['feature_sha256'] == read_json_hash()
            and file_hash(resolve_path(config['features'])) == config['feature_sha256'], 'G2 checksum differs')
    frame, columns = load_feature_table(resolve_path(config['features']), resolve_path(config['schema']), SUBJECTS)
    require(len(columns) == 68 and list(frame) == reference['output_column_order'], 'Wrong input columns')
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        for subject in SUBJECTS:
            require(set(y[frame.subject_id != subject]) == set(labels), 'Outer train missing class')
    for model in config['models']:
        require(len(list(ParameterGrid(config['search_spaces'][model]))) == 4, 'Exactly four candidates required')
    return frame


def read_json_hash():
    # Keep the accepted G3/G2 input lock; never choose data using outer scores.
    from wesad.utils import read_config
    return read_config('configs/defaults.yaml')['feature_sha256']


def check_splits(detail, subject):
    train = set(SUBJECTS) - {subject}
    require(set(detail['outer_train_subjects']) == train
            and detail['outer_test_subjects'] == [subject], 'Outer subject leakage')
    require(len(detail['inner_splits']) == 3, 'Expected three inner splits')
    validations = []
    for split in detail['inner_splits']:
        a, b = set(split['train_subjects']), set(split['validation_subjects'])
        require(a and b and not a & b and a | b == train, 'Inner subject leakage/incomplete split')
        validations.extend(split['validation_subjects'])
    require(len(validations) == 14 and set(validations) == train, 'Inner validation must cover each train subject once')


def verify(config, out, quick=False):
    frame = preflight(config)
    manifest = read_json(out / 'config.json')
    summary = read_json(out / 'summary.json')
    require(summary['status'] in {'partial','complete'}, 'Invalid run status')
    predictions = pd.read_csv(out / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    require(not predictions.duplicated(['task','model','subject_id','window_id']).any(), 'Duplicate OOF keys')
    verified = []
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        for model in config['models']:
            part = predictions[(predictions.task == task) & (predictions.model == model)]
            subjects = set(part.subject_id)
            require(subjects and subjects <= set(SUBJECTS)
                    and (quick or subjects == set(config['outer_subjects'])), 'Missing assigned subjects')
            for subject in sorted(subjects):
                folder = out / 'folds' / task / model / subject
                completion = read_json(folder / 'complete.json')
                require(completion['signature'] == manifest['signature'], 'Fold signature mismatch')
                require({'predictions.csv','metrics.json','details.json','inner_cv.csv'} <= set(completion['files']), 'Missing fold artifacts')
                for name, digest in completion['files'].items():
                    path = (folder / name).resolve()
                    require(path.is_relative_to(folder.resolve()) and file_hash(path) == digest, 'Fold checksum mismatch')
                detail = read_json(folder / 'details.json')
                check_splits(detail, subject)
                if model == 'mlp' and config.get('record_mlp_loss'):
                    optimization = detail['mlp_optimization']
                    require(len(optimization['loss_curve']) == optimization['n_iter']
                            and np.isfinite(optimization['loss_curve']).all(), 'Invalid MLP loss trace')
                candidates = list(ParameterGrid(config['search_spaces'][model]))
                require(detail['best_params'] in candidates, 'Best params outside declared candidates')
                inner = pd.read_csv(folder / 'inner_cv.csv')
                require(len(inner) == 4 and np.isfinite(inner.mean_test_score).all(), 'Invalid inner results')
                winner = inner.iloc[int(np.argmax(inner.mean_test_score))]
                require(all(json.loads(json.dumps(detail['best_params'][key])) ==
                            json.loads(json.dumps(value)) for key, value in candidates[int(winner.name)].items()), 'Best params differ from inner winner')
                g = part[part.subject_id == subject]
                reference = frame[frame.subject_id == subject][['window_id']].copy()
                reference['expected_y'] = y[frame.subject_id == subject].to_numpy()
                joined = reference.merge(g, on='window_id', how='outer', validate='one_to_one', indicator=True)
                require((joined['_merge'] == 'both').all() and (joined.expected_y == joined.y_true).all()
                        and g.y_pred.isin(labels).all(), 'Wrong OOF rows/labels')
                metric = read_json(folder / 'metrics.json')
                for key, value in [('macro_f1',f1_score(g.y_true,g.y_pred,labels=labels,average='macro',zero_division=0)),
                                   ('balanced_accuracy',balanced_accuracy_score(g.y_true,g.y_pred))]:
                    require(np.isclose(value,metric[key],atol=1e-12,rtol=0), 'Metric differs from predictions')
                require(detail['confusion_matrix'] == confusion_matrix(g.y_true,g.y_pred,labels=labels).tolist(), 'Wrong confusion matrix')
                verified.append({**metric, 'best_inner_macro_f1':float(winner.mean_test_score),
                                 'best_params':json.dumps(detail['best_params'],sort_keys=True),
                                 'n_warnings':len(detail['warnings'])})
            stored = next(r for r in summary['results'] if r['task']==task and r['model']==model)
            rows = [r for r in verified if r['task']==task and r['model']==model]
            for key in ('macro_f1','balanced_accuracy'):
                values = [r[key] for r in rows]
                require(np.isclose(stored[key+'_mean'],np.mean(values),atol=1e-12,rtol=0), 'Wrong summary mean')
                if len(values)>1:
                    require(np.isclose(stored[key+'_std'],np.std(values,ddof=1),atol=1e-12,rtol=0), 'Wrong summary SD')
    write_csv(out / 'verified_metrics_by_subject.csv',pd.DataFrame(verified))
    result = {'status':'passed','scope':'quick' if quick else 'assigned_outer4',
              'complete_15_subjects':False,
              'folds':len(verified),'predictions':len(predictions),'data_sha256':config['feature_sha256']}
    write_json(out / 'verification.json',result)
    return result


def run(config, resume=False, subject=None, verify_only=False):
    preflight(config)
    out = resolve_path(config['output_dir'])
    require(not subject or 'quick' in out.name.lower(), 'One-subject checks need a quick output folder')
    if not verify_only:
        import yaml
        run_experiment(config,resume=resume,outer_subjects=[subject] if subject else config['outer_subjects'])
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in config['source_paths']:
                archive.write(resolve_path(path),path)
    return verify(config,out,quick=bool(subject))
