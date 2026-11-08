"""G3 baseline contract, independent OOF checks, and group prediction assembly."""
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

from wesad.data import SUBJECTS, load_feature_table
from wesad.evaluation import run_experiment, task_target
from wesad.models import MODEL_NAMES, build_pipeline
from wesad.utils import (PROJECT_ROOT, file_hash, read_json, resolve_path,
                         write_json, write_csv, environment, fingerprint)

PROTOCOL = 'g3_defaults_physiological68_v1'
KEY = ['task', 'model', 'sensor_set', 'normalization', 'subject_id', 'window_id']
SOURCE_PATHS = {'src/wesad/data.py','src/wesad/evaluation.py','src/wesad/models.py',
                'src/wesad/utils.py','src/wesad/g3.py','scripts/run_g3_baselines.py',
                'scripts/merge_g3_predictions.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(config):
    require(config.get('protocol') == PROTOCOL, 'Use the G3 physiological68 defaults protocol')
    require(set(config.get('source_paths', [])) == SOURCE_PATHS
            and len(config['source_paths']) == len(SOURCE_PATHS), 'G3 source scope differs')
    require(config.get('source_hash_normalization') == 'LF', 'G3 source uses declared LF normalization')
    require(not config.get('nested') and not config.get('balance'), 'G3 defaults exclude tuning/balancing')
    require(not config.get('model_factories') and not config.get('search_spaces')
            and config.get('variance_threshold') is None, 'No factories/search/selection in G3 defaults')
    require(config.get('zero_division') == 0 and config.get('missing_train_class') == 'error', 'Metric/class policy differs')
    require(config.get('missing_test_class') == 'fixed_labels_for_macro_f1_observed_classes_for_balanced_accuracy', 'Test-class policy differs')
    require(config.get('seed') == 42 and config.get('normalization') == 'global_train'
            and config.get('sensor_set') == 'combined_physiological68', 'Seed/normalization/sensor policy differs')
    require(set(config['expected_subjects']) == set(SUBJECTS)
            and len(config['expected_subjects']) == 15, 'Expect all 15 subjects')
    require(set(config['tasks']) == {'multiclass', 'binary'} and len(config['tasks']) == 2, 'Run both tasks')
    require(config['models'] and len(set(config['models'])) == len(config['models'])
            and set(config['models']) <= set(MODEL_NAMES), 'Invalid model list')
    for name, params in config.get('model_params', {}).items():
        require(name in MODEL_NAMES and set(params) <= {'random_state', 'n_jobs'}, 'Only seed/parallelism overrides are defaults')
        require(all((key == 'random_state' and value == 42) or (key == 'n_jobs' and value == 1)
                    for key, value in params.items()), 'Unexpected default override')
    table, schema = resolve_path(config['features']), resolve_path(config['schema'])
    require(file_hash(table) == config['feature_sha256'], 'Feature checksum differs from locked G2 table')
    contract = read_json(schema)
    require(config['feature_version'] == contract['feature_version'] == 'physiological_68_v1', 'Not physiological68')
    require(len(contract['feature_columns']) == 68, 'Expected exactly 68 features')
    frame, columns = load_feature_table(table, schema, SUBJECTS)
    require(list(frame.columns) == contract['output_column_order'], 'Feature/metadata column order differs')
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        for subject in SUBJECTS:
            require(set(y[frame.subject_id != subject]) == set(labels), f'{task}/{subject}: train missing class')
    return frame, columns


def verify_run(directory, config, allow_quick=False):
    directory = Path(directory)
    frame, columns = check_config(config)
    manifest = read_json(directory / 'config.json')
    identity = {k:manifest[k] for k in ('settings','data_sha256','feature_columns','code_sha256','environment')}
    require(fingerprint(identity) == manifest['signature'], 'Invalid run signature')
    archive = directory / 'source_snapshot.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            require(set(z.namelist()) == SOURCE_PATHS, 'Source snapshot inventory differs')
            hashes={n:hashlib.sha256(z.read(n).replace(b'\r\n',b'\n')).hexdigest() for n in z.namelist()}
            require(fingerprint(hashes)==manifest['code_sha256'], 'Source snapshot code differs')
    require(manifest['data_sha256'] == config['feature_sha256'] and manifest['feature_columns'] == columns, 'Run data/schema differs')
    settings = {k: v for k, v in config.items() if k not in {'features', 'schema', 'output_dir'}}
    require(manifest['settings'] == settings, 'Run settings differ')
    summary = read_json(directory / 'summary.json')
    if not allow_quick:
        require(summary['status'] == 'complete', 'Quick/partial run cannot be a main group result')
    predictions = pd.read_csv(directory / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    require(not predictions.duplicated(KEY).any(), 'Duplicate prediction keys')
    require(set(predictions.model) == set(config['models']) and set(predictions.task) == set(config['tasks']), 'Missing or extra models/tasks')
    require(set(predictions.sensor_set) == {config['sensor_set']}
            and set(predictions.normalization) == {'global_train'}, 'Prediction sensor/normalization differs')
    metrics, warnings_by_fold = [], []
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        reference = frame[['subject_id', 'window_id']].copy()
        reference['y_true'] = y
        for model in config['models']:
            part = predictions.loc[(predictions.task == task) & (predictions.model == model)]
            subjects = set(part.subject_id)
            require(subjects <= set(SUBJECTS), 'Unknown predicted subject')
            if not allow_quick:
                require(subjects == set(SUBJECTS), 'Incomplete LOSO subjects')
            expected = reference.loc[reference.subject_id.isin(subjects)]
            checked = expected.merge(part, on=['subject_id', 'window_id'], how='outer', validate='one_to_one', indicator=True, suffixes=('_expected', ''))
            require((checked['_merge'] == 'both').all() and (checked.y_true_expected == checked.y_true).all(), 'OOF keys or true labels differ')
            require(part.y_pred.isin(labels).all(), 'Predictions contain invalid classes')
            for subject in sorted(subjects):
                folder = directory / 'folds' / task / model / subject
                done = read_json(folder / 'complete.json')
                require(done['signature'] == manifest['signature'] and {'predictions.csv','metrics.json','details.json'} <= set(done['files']), 'Fold signature/inventory differs')
                for name, digest in done['files'].items():
                    path = (folder / name).resolve()
                    require(path.is_relative_to(folder.resolve()) and file_hash(path) == digest, 'Fold checksum differs')
                detail = read_json(folder / 'details.json')
                require(set(detail['outer_train_subjects']) == set(SUBJECTS) - {subject}
                        and len(detail['outer_train_subjects']) == 14
                        and detail['outer_test_subjects'] == [subject], 'Outer train/test subject leakage')
                require(not detail['inner_splits'] and not detail['best_params'], 'Tuning found in G3 default result')
                g = part.loc[part.subject_id == subject]
                fold_predictions = pd.read_csv(folder / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
                pd.testing.assert_frame_equal(g.reset_index(drop=True), fold_predictions.reset_index(drop=True))
                metric = read_json(folder / 'metrics.json')
                f1 = float(f1_score(g.y_true, g.y_pred, labels=labels, average='macro', zero_division=0))
                ba = float(balanced_accuracy_score(g.y_true, g.y_pred))
                require(np.isclose(f1, metric['macro_f1'], atol=1e-12, rtol=0)
                        and np.isclose(ba, metric['balanced_accuracy'], atol=1e-12, rtol=0), 'Stored metric differs from predictions')
                require(confusion_matrix(g.y_true,g.y_pred,labels=labels).tolist() == detail['confusion_matrix'], 'Fold confusion matrix differs')
                metrics.append({**metric, 'sensor_set':config['sensor_set'], 'normalization':'global_train'})
                if detail['warnings']:
                    warnings_by_fold.append({'task':task, 'model':model, 'subject_id':subject, 'warnings':detail['warnings']})
            stored = [r for r in summary['results'] if r['task']==task and r['model']==model]
            require(len(stored)==1, 'Missing/duplicate summary group')
            group_metrics = [r for r in metrics if r['task']==task and r['model']==model]
            for score in ('macro_f1','balanced_accuracy'):
                values = [r[score] for r in group_metrics]
                require(np.isclose(stored[0][score+'_mean'],np.mean(values),rtol=0,atol=1e-12), 'Summary mean differs')
                expected_std=float(np.std(values,ddof=1)) if len(values)>1 else None
                require((expected_std is None and stored[0][score+'_std'] is None) or
                        (expected_std is not None and np.isclose(stored[0][score+'_std'],expected_std,rtol=0,atol=1e-12)), 'Summary standard deviation differs')
            require(stored[0]['confusion_matrix']==confusion_matrix(part.y_true,part.y_pred,labels=labels).tolist(), 'Summary confusion matrix differs')
    return {'status':'passed', 'scope':'quick' if allow_quick else 'main',
            'n_features':68, 'n_subjects':len(set(predictions.subject_id)),
            'n_predictions':len(predictions), 'n_folds':len(metrics),
            'data_sha256':manifest['data_sha256'], 'code_sha256':manifest['code_sha256'],
            'warnings_by_fold':warnings_by_fold, 'models':config['models'], 'tasks':config['tasks']}, pd.DataFrame(metrics)


def run_defaults(config, resume=False, quick_subject=None):
    check_config(config)
    out = resolve_path(config['output_dir'])
    if quick_subject:
        require('quick' in out.name.lower(), 'Quick output directory must contain quick')
    result = run_experiment(config, resume=resume, outer_subjects=[quick_subject] if quick_subject else None)
    verification, metrics = verify_run(out, config, allow_quick=bool(quick_subject))
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (out / 'environment.txt').write_text(json.dumps(environment(), indent=2), encoding='utf-8')
    write_json(out / 'model_parameters.json', {name:build_pipeline(name,config)['model'].get_params() for name in config['models']})
    write_json(out / 'verification.json', verification)
    write_csv(out / 'verified_metrics_by_subject.csv', metrics)
    write_json(out / 'scope.json', {'scope':'quick' if quick_subject else 'main', 'group_G3_complete':False,
        'remaining_models':sorted(set(MODEL_NAMES)-set(config['models']))})
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(resolve_path(p) for p in config['source_paths']):
            z.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    return result, verification


def merge_runs(directories, output, reference_config):
    """Accept complete model batches; never promote quick results or fabricate missing models."""
    output = Path(output)
    require(not output.exists(), 'Use a new group output directory')
    predictions, records, manifests = [], [], []
    code = None
    for directory in map(Path, directories):
        scope = read_json(directory/'scope.json')
        require(scope['scope'] == 'main', 'Quick results cannot be merged')
        require((directory/'source_snapshot.zip').is_file(), 'Missing source evidence')
        config = yaml.safe_load((directory/'config.yaml').read_text(encoding='utf-8'))
        # Paths can differ on another machine; identities and protocol cannot.
        for key in ('features','schema','output_dir'):
            config[key] = reference_config[key]
        expected = {**reference_config, 'models':config['models']}
        require({k:v for k,v in config.items() if k not in {'features','schema','output_dir'}} ==
                {k:v for k,v in expected.items() if k not in {'features','schema','output_dir'}}, 'Different G3 protocol settings')
        verification, metrics = verify_run(directory, config)
        manifest = read_json(directory/'config.json')
        if code is None:
            code = verification['code_sha256']
        require(code == verification['code_sha256'], 'Different G3 source code; inspect before merging')
        env = manifest['environment']
        require(env['python'].split()[0] == '3.12.14'
                and env['packages'] == environment()['packages'], 'Different Python patch/libraries')
        predictions.append(pd.read_csv(directory/'predictions.csv',dtype={'subject_id':str,'window_id':str}))
        records.append(metrics)
        manifests.append(manifest)
    require(bool(predictions), 'No model packages supplied')
    joined = pd.concat(predictions,ignore_index=True)
    require(not joined.duplicated(KEY).any(), 'Duplicate model/task/subject/window predictions')
    missing = sorted(set(MODEL_NAMES)-set(joined.model))
    summary=[]
    for (task,model),g in pd.concat(records).groupby(['task','model']):
        pred=joined.loc[(joined.task==task)&(joined.model==model)]
        labels=[1,2,3] if task=='multiclass' else [0,1]
        summary.append({'task':task,'model':model,'n_subjects':len(g),
            'macro_f1_mean':float(g.macro_f1.mean()),'macro_f1_std':float(g.macro_f1.std(ddof=1)),
            'balanced_accuracy_mean':float(g.balanced_accuracy.mean()),'balanced_accuracy_std':float(g.balanced_accuracy.std(ddof=1)),
            'labels':labels,'confusion_matrix':confusion_matrix(pred.y_true,pred.y_pred,labels=labels).tolist()})
    output.mkdir(parents=True)
    write_csv(output/'predictions.csv',joined)
    write_csv(output/'metrics_by_subject.csv',pd.concat(records,ignore_index=True))
    result={'status':'complete' if not missing else 'waiting_for_models','missing_models':missing,
        'data_sha256':reference_config['feature_sha256'],'results':summary,'source_packages':manifests}
    write_json(output/'summary.json',result)
    return result
