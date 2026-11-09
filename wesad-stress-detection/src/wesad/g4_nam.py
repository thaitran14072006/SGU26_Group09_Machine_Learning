"""Nam G4 handoff around the unchanged shared numerical core.

Full 15-subject runs and explicit outer shards are supported. Quick is always
separate, cannot be packaged/merged as main. No public flag bypasses the data,
environment or training-core locks.
"""
import hashlib
import json
from pathlib import Path
import zipfile
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from wesad.data import SUBJECTS
from wesad.evaluation import run_experiment
from wesad.g4 import candidates, inputs, verify_run
from wesad.utils import (environment, file_hash, fingerprint, read_config, read_json,
                         resolve_path, write_csv, write_json)

DATA_SHA = '2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba'
N_ROWS = 6189
PACKAGES = {'numpy': '2.5.4', 'pandas': '3.1.0', 'PyYAML': '6.0.3', 'scipy': '1.18.1',
            'scikit-learn': '1.9.1', 'imbalanced-learn': '0.14.2'}
CORE = {
    'src/wesad/data.py': '530e3a15ab3ffe9737fef0526be9acbdb6049e7e2d83c185a427956175ed3f61',
    'src/wesad/evaluation.py': '32189d0f01cf80b96a29346546c02f05fe6b6cca19fda73d23d3c8be055a79d7',
    'src/wesad/models.py': 'bd5105be66ce32df12f2bac928339a7aea1c313419c2c6672ec53778b516da05',
    'src/wesad/utils.py': '0dd1d6d5e063bc59572ce9adebcaccf1d1e811cd74267130bd68f0a0a19620fa',
    'src/wesad/g4.py': 'd9cca29d4426b1dc916a9eaf4db88ba231058f7660b7536c08ac303fc9964134',
}
SOURCE_PATHS = set(CORE) | {'src/wesad/g4_nam.py', 'scripts/run_nam_g4.py', 'scripts/report_nam_g4.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_environment():
    env = environment()
    require(env['python'].split()[0] == '3.12.14', 'Use Python 3.12.14 with an explicit interpreter path')
    require(env['packages'] == PACKAGES,
            f"Locked dependencies differ. Expected {PACKAGES}; found {env['packages']}")


def check_sources():
    for name, expected in CORE.items():
        actual = hashlib.sha256(resolve_path(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        require(actual == expected, f'Shared source changed: {name}. Review against base 8011c94 before training; do not edit hash locks.')


def prepare_config(path, *, features=None, schema=None, output=None, outer_subjects=None, quick_subject=None):
    c = read_config(path)
    require(not (outer_subjects and quick_subject), 'Choose quick OR assigned outer subjects')
    for key, val in [('features', features), ('schema', schema), ('output_dir', output)]:
        if val is not None:
            c[key] = val
    if quick_subject:
        c['outer_subjects'] = [quick_subject]
        c['run_kind'] = 'quick'
        if not output:
            c['output_dir'] += '_quick_' + quick_subject
    elif outer_subjects is not None:
        c['outer_subjects'] = list(outer_subjects)
        c['run_kind'] = 'main' if set(outer_subjects) == set(SUBJECTS) else 'main_shard'
        require(bool(output) or c['run_kind'] == 'main', 'For an outer shard specify a new --output directory')
    else:
        c['outer_subjects'] = c.get('outer_subjects', list(SUBJECTS))
    return c


def preflight(c):
    check_environment()
    check_sources()
    require(c.get('protocol') == 'g4_nam_nested_physiological68_v1', 'Wrong Nam G4 protocol')
    require(c.get('feature_sha256') == DATA_SHA, 'Do not change the accepted G2 checksum')
    require(c.get('models') == ['knn', 'svm'], 'Use KNN and SVM together')
    require(c.get('nested') is True and c.get('inner_splits') == 3 and c.get('balance') is False,
            'Main tuning uses nested CV(3), balance=false')
    require(not c.get('model_factories') and not c.get('model_params') and c.get('variance_threshold') is None,
            'Use shared defaults; no factory, selection, or model overrides')
    require(c.get('zero_division') == 0 and c.get('missing_train_class') == 'error', 'Wrong metric/class policy')
    require(set(c.get('source_paths', [])) == SOURCE_PATHS and len(c['source_paths']) == len(SOURCE_PATHS)
            and c.get('source_hash_normalization') == 'LF', 'Wrong source snapshot inventory')
    selected = c.get('outer_subjects', [])
    require(selected and len(selected) == len(set(selected)) and set(selected) <= set(SUBJECTS), 'Invalid outer subjects')
    kind = c.get('run_kind')
    require(kind in {'main', 'main_shard', 'quick'}, 'Declare main/main_shard/quick')
    require((kind != 'main' or set(selected) == set(SUBJECTS))
            and (kind != 'main_shard' or len(selected) < 15), 'Run scope differs from assigned subjects')
    if kind == 'quick':
        require(len(selected) == 1 and 'quick' in resolve_path(c['output_dir']).name.lower(), 'Quick needs one subject and a separate quick output')
    else:
        require('quick' not in resolve_path(c['output_dir']).name.lower(), 'Main output must not be a quick folder')
    frame, columns = inputs(c)
    require(len(frame) == N_ROWS, 'Unexpected number of windows in locked table')
    schema = read_json(resolve_path(c['schema']))
    reference = read_json(resolve_path('configs/feature_schema.json'))
    require(columns == reference['feature_columns'] and schema['sensor_groups'] == reference['sensor_groups'], 'G2 feature/group schema differs')
    require(frame.label_fraction.ge(.9).all() and np.allclose(frame.end_s-frame.start_s, 60), 'Window duration/label purity differs')
    for model in c['models']:
        candidates(c, model)
    for task in c['tasks']:
        labels = {1, 2, 3} if task == 'multiclass' else {0, 1}
        y = frame.label_original if task == 'multiclass' else frame.label_binary
        for subject in SUBJECTS:
            require(set(y[frame.subject_id != subject]) == labels, 'Outer train missing a class')
    return frame, columns


def _check_identity(c, out):
    m = read_json(out / 'config.json')
    settings = {k: v for k, v in c.items() if k not in {'features', 'schema', 'output_dir'}}
    require(m['settings'] == settings, 'Config/scope does not match this output. Reuse the exact original command/config')
    hashes = {n: hashlib.sha256(resolve_path(n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for n in c['source_paths']}
    require(fingerprint(hashes) == m['code_sha256'], 'Runtime sources changed. Restore the original version to verify/resume')
    return m


def snapshot(c, out):
    _check_identity(c, out)
    path = out / 'source_snapshot.zip'
    if path.exists():
        with zipfile.ZipFile(path) as z:
            require(set(z.namelist()) == SOURCE_PATHS, 'Existing snapshot inventory differs')
            for name in SOURCE_PATHS:
                require(z.read(name).replace(b'\r\n', b'\n') == resolve_path(name).read_bytes().replace(b'\r\n', b'\n'), 'Existing snapshot differs; do not overwrite')
        return
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(SOURCE_PATHS):
            z.write(resolve_path(name), name)


def verify(c):
    preflight(c)
    require(c['run_kind'] != 'quick', 'Quick is a smoke check only; it cannot receive main verification or handoff')
    out = resolve_path(c['output_dir'])
    _check_identity(c, out)
    pred, metrics, best, proof = verify_run(out, c, kind='nested', allow_partial=c['run_kind'] == 'main_shard')
    expected = {(t, m, s) for t in c['tasks'] for m in c['models'] for s in c['outer_subjects']}
    require(set(zip(metrics.task, metrics.model, metrics.subject_id)) == expected, 'Missing or extra assigned folds')
    stored = read_json(out / 'summary.json')
    for task in c['tasks']:
        for model in c['models']:
            group = metrics[(metrics.task == task) & (metrics.model == model)]
            r = next(x for x in stored['results'] if x['task'] == task and x['model'] == model)
            require(r['n_subjects'] == len(group), 'Summary subject count differs')
            for score in ('macro_f1', 'balanced_accuracy'):
                require(np.isclose(r[score+'_mean'], group[score].mean(), atol=1e-12, rtol=0), 'Summary mean differs')
                sd = r[score+'_std']
                require(sd is None if len(group) == 1 else np.isclose(sd, group[score].std(ddof=1), atol=1e-12, rtol=0), 'Summary SD differs')
    write_csv(out / 'verified_metrics_by_subject.csv', metrics)
    result = {'status': 'passed', 'scope': 'full_15_subjects' if c['run_kind'] == 'main' else 'assigned_outer_subjects',
              'complete_15_subjects': c['run_kind'] == 'main', 'models': c['models'], 'tasks': c['tasks'],
              'outer_subjects': c['outer_subjects'], 'folds': len(metrics), 'predictions': len(pred),
              'data_sha256': c['feature_sha256'], 'training_core_sha256': proof['training_core_sha256'],
              'warnings_by_fold': proof['warnings_by_fold']}
    write_json(out / 'scope.json', {k: result[k] for k in ('scope', 'complete_15_subjects', 'models', 'tasks', 'outer_subjects')})
    write_json(out / 'verification.json', result)
    return result


def run(c, resume=False):
    preflight(c)
    out = resolve_path(c['output_dir'])
    require(not out.is_file(), 'Output must be a directory')
    if out.exists() and any(out.iterdir()):
        require((out/'config.json').is_file() and resume, 'Existing output requires --resume and a matching config manifest; otherwise use a new --output')
        _check_identity(c, out)
    # Capture source bytes before fitting. Existing archives are checked, not replaced.
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'source_snapshot.zip').exists():
        with zipfile.ZipFile(out/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
            for name in sorted(SOURCE_PATHS):
                z.write(resolve_path(name), name)
    elif (out/'config.json').exists():
        snapshot(c, out)
    try:
        result = run_experiment(c, resume=resume, outer_subjects=c['outer_subjects'])
    finally:
        if (out/'config.json').exists():
            snapshot(c, out)
            import yaml
            (out/'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    if c['run_kind'] == 'quick':
        return {'status': 'quick_completed_not_main_verified', 'summary_status': result['status'],
                'output': str(out), 'note': 'Do not merge or submit quick as main. Run an official full/shard command next.'}
    return verify(c)


def _comparison(c, out, metrics_path):
    source = resolve_path(metrics_path)
    old = pd.read_csv(source)
    required = {'task', 'model', 'subject_id', 'macro_f1', 'balanced_accuracy'}
    require(required <= set(old), 'G3 metrics missing columns')
    old = old[old.model.isin(c['models']) & old.task.isin(c['tasks'])].copy()
    require(not old.duplicated(['task', 'model', 'subject_id']).any(), 'Duplicate G3 fold keys')
    keys = {(t, m, s) for t in c['tasks'] for m in c['models'] for s in SUBJECTS}
    require(set(zip(old.task, old.model, old.subject_id)) == keys, 'Reference G3 must have all 60 KNN/SVM folds')
    values = old[['macro_f1', 'balanced_accuracy']].to_numpy(float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid G3 scores')
    current = pd.read_csv(out/'verified_metrics_by_subject.csv')
    cols = ['task', 'model', 'subject_id', 'macro_f1', 'balanced_accuracy']
    paired = current[cols].merge(old[cols], on=['task', 'model', 'subject_id'],
                                 suffixes=('_g4', '_g3'), validate='one_to_one')
    require(len(paired) == len(current), 'Missing paired G3 subjects')
    rows = []
    for score in ('macro_f1', 'balanced_accuracy'):
        paired['delta_'+score] = paired[score+'_g4']-paired[score+'_g3']
    for (task, model), g in paired.groupby(['task', 'model']):
        row = {'task': task, 'model': model, 'n_subjects': len(g)}
        for score in ('macro_f1', 'balanced_accuracy'):
            for suffix in ('g3', 'g4'):
                row[score+'_'+suffix+'_mean'] = float(g[score+'_'+suffix].mean())
                row[score+'_'+suffix+'_std'] = float(g[score+'_'+suffix].std(ddof=1)) if len(g)>1 else None
            row['delta_'+score+'_mean'] = float(g['delta_'+score].mean())
        rows.append(row)
    write_csv(out/'paired_g3_g4_by_subject.csv', paired)
    write_csv(out/'default_vs_tuned.csv', pd.DataFrame(rows))
    write_json(out/'comparison_provenance.json', {'g3_metrics_path': str(source), 'g3_metrics_sha256': file_hash(source),
               'scope': c['outer_subjects'], 'note': 'Pairs only the same people. Reference is the accepted G3 metrics file; this does not reverify G3 prediction packages. No significance test.'})
    return rows


def export_figures(out, rows):
    """Dependency-free SVG charts from verified scores/matrices; never fit again."""
    fig = out/'figures'; fig.mkdir(exist_ok=True)
    def document(w,h,body):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>'+body+'</svg>'
    body='<text x="25" y="30" font-family="Arial" font-size="19">G3 vs G4: paired Macro-F1 (same subjects)</text>'
    for i,r in enumerate(rows):
        y=65+i*90
        label=f"{r['task']} / {r['model']} (n={r['n_subjects']})"
        body+=f'<text x="25" y="{y}" font-family="Arial" font-size="14">{escape(label)}</text>'
        for j,(k,col) in enumerate([('g3','#52667A'),('g4','#167D8D')]):
            v=r['macro_f1_'+k+'_mean']; yy=y+8+j*22
            body+=f'<rect x="270" y="{yy}" width="{v*500:.2f}" height="17" fill="{col}"/><text x="{280+v*500:.2f}" y="{yy+14}" font-family="Arial" font-size="13">{k.upper()} {v:.4f}</text>'
    body+='<text x="25" y="440" font-family="Arial" font-size="13">Score range 0–1. Descriptive comparison; no statistical significance claim.</text>'
    (fig/'g3_vs_g4_macro_f1.svg').write_text(document(900,465,body),encoding='utf-8')
    for r in read_json(out/'summary.json')['results']:
        labels=r['labels'];cm=np.array(r['confusion_matrix']);n=len(labels);body=''
        title=f"{r['task']} / {r['model']}: OOF confusion matrix (n={r['n_subjects']} people)"
        body+=f'<text x="25" y="32" font-family="Arial" font-size="18">{escape(title)}</text><text x="180" y="65" font-family="Arial" font-size="14">Predicted label</text>'
        for i,label in enumerate(labels):
            body+=f'<text x="{180+i*100}" y="100" font-family="Arial">{label}</text><text x="100" y="{163+i*100}" font-family="Arial">{label}</text>'
            for j in range(n):
                v=int(cm[i,j]); a=.12+.7*v/max(int(cm.max()),1)
                body+=f'<rect x="{135+j*100}" y="{115+i*100}" width="95" height="95" fill="#167D8D" fill-opacity="{a:.3f}"/><text x="{180+j*100}" y="{168+i*100}" text-anchor="middle" font-family="Arial">{v}</text>'
        body+='<text x="25" y="90" font-family="Arial" font-size="13">True label</text>'
        (fig/f"confusion_{r['task']}_{r['model']}.svg").write_text(document(650,440,body),encoding='utf-8')


def report(c, g3_metrics):
    proof = verify(c)
    out = resolve_path(c['output_dir'])
    rows = _comparison(c, out, g3_metrics)
    export_figures(out, rows)
    log_rows = []
    for row in rows:
        log_rows.append({**row, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'stage': 'G4', 'scope': proof['scope'], 'protocol': c['protocol'],
            'data_sha256': c['feature_sha256'], 'outer_split': 'LeaveOneGroupOut',
            'inner_split': 'GroupKFold(3)', 'candidates_per_model': 4,
            'selection_metric': 'inner Macro-F1', 'balance': False,
            'normalization': 'global_train', 'best_params': 'best_params.json (one winner per outer fold)',
            'note': 'Descriptive paired comparison; keep negative deltas; not a significance test'})
    write_csv(out/'experiment_log.csv', pd.DataFrame(log_rows))
    # Hash every file actually handed over; archive sits next to the run, not inside it.
    inventory = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and p.name != 'handoff_checksums.json'}
    write_json(out/'handoff_checksums.json', inventory)
    archive = out.parent/(out.name+'_handoff.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, out.name+'/'+p.relative_to(out).as_posix())
    return {'status': 'handoff_ready', 'verification': proof, 'zip_path': str(archive),
            'zip_sha256': file_hash(archive), 'note': 'Submit the whole ZIP. A main shard is not full 15-subject G4.'}
