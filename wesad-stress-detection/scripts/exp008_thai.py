"""Six preregistered SVM candidates; subject-separated inner and outer CV."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from wesad.data import SUBJECTS, load_feature_table
from wesad.utils import environment, file_hash, fingerprint, read_config, write_json

BASE = ROOT / 'experiments/02_tuning/EXP-008_SVM_Thai'


def setup():
    cfg = read_config(ROOT / 'configs/exp008_thai.yaml')
    default = read_config(ROOT / 'configs/defaults.yaml')
    assert file_hash(ROOT / default['features']) == cfg['feature_sha256']
    frame, columns = load_feature_table(ROOT / default['features'], ROOT / default['schema'], SUBJECTS)
    assert len(frame) == 6189 and len(columns) == 68
    return cfg, frame, columns


def score(y, pred, labels):
    return dict(macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)))


def signature(cfg, spec):
    return fingerprint(dict(plan=cfg, run=spec, environment=environment(),
        sources={n: file_hash(ROOT/n) for n in ['scripts/exp008_thai.py', 'src/wesad/data.py', 'src/wesad/utils.py']}))


def verify(run):
    cfg, frame, columns = setup()
    spec = next(s for s in cfg['runs'] if s['run_id'] == run)
    out = BASE / 'runs' / run
    receipt = json.loads((out/'acceptance.json').read_text())
    assert receipt['signature'] == signature(cfg, spec)
    for name, digest in receipt['files'].items():
        assert file_hash(out/name) == digest, name
    metrics = pd.read_csv(out/'results.csv')
    predictions = pd.read_csv(out/'predictions.csv', dtype={'window_id': str})
    details = json.loads((out/'fold_details.json').read_text())
    summary = pd.read_csv(out/'summary.csv').set_index('task')
    assert len(metrics) == 30 and len(predictions) == 12378 and len(details) == 30
    assert not predictions.duplicated(['task','subject_id','window_id']).any()
    assert not metrics.duplicated(['task','subject_id']).any()
    for task in ['multiclass', 'binary']:
        labels = [1,2,3] if task == 'multiclass' else [0,1]
        p = predictions[predictions.task == task]
        match = frame.merge(p, on=['subject_id','window_id'], how='outer', validate='one_to_one', indicator=True)
        assert len(match) == 6189 and match['_merge'].eq('both').all()
        y = match.label_original if task == 'multiclass' else (match.label_original == 2).astype(int)
        assert np.array_equal(y, match.y_true) and match.y_pred.isin(labels).all()
        assert set(metrics.loc[metrics.task == task, 'subject_id']) == set(SUBJECTS)
        for subject in SUBJECTS:
            part = p[p.subject_id == subject]
            m = metrics[(metrics.task == task) & (metrics.subject_id == subject)].iloc[0]
            d = details[f'{task}/{subject}']
            for key, value in score(part.y_true, part.y_pred, labels).items():
                assert np.isclose(m[key], value)
            assert m.n_test == len(part) and not d['warnings']
            assert d['confusion_matrix'] == confusion_matrix(part.y_true, part.y_pred, labels=labels).tolist()
            assert d['test_subject'] == subject and set(d['train_subjects']) == set(SUBJECTS)-{subject}
            seen = []
            for split in d['inner_splits']:
                a,b = set(split['train']),set(split['validation'])
                assert not a & b and subject not in a|b and a|b == set(SUBJECTS)-{subject}
                seen.extend(b)
            assert len(seen) == 14 and set(seen) == set(SUBJECTS)-{subject}
            assert len(d['inner_scores']) == 3 and np.isfinite(d['inner_scores']).all()
            assert np.isclose(m.inner_macro_f1, np.mean(d['inner_scores']))
        for name in ['macro_f1','balanced_accuracy']:
            vals = metrics.loc[metrics.task == task, name]
            assert np.isclose(summary.loc[task, name+'_mean'], vals.mean())
            assert np.isclose(summary.loc[task, name+'_std'], vals.std(ddof=1))
    return {'run_id': run, 'status':'passed', 'folds':30, 'predictions':12378,
            'acceptance_sha256':file_hash(out/'acceptance.json')}


def run_one(run):
    cfg, frame, columns = setup()
    spec = next(s for s in cfg['runs'] if s['run_id'] == run)
    out = BASE/'runs'/run
    if (out/'acceptance.json').exists():
        print(verify(run)); return
    sig = signature(cfg, spec)
    work = ROOT/'outputs/exp008_thai'/run
    work.mkdir(parents=True, exist_ok=True)
    manifest = work/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['signature'] == sig, 'Checkpoint signature mismatch'
    else:
        write_json(manifest, {'signature':sig,'plan':cfg,'run':spec,'environment':environment(),
            'started_at':datetime.now(timezone.utc).isoformat()})
    pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
        SVC(C=spec['C'], gamma=spec['gamma'], kernel='rbf', class_weight=None,
            cache_size=256, max_iter=-1, random_state=42))
    X, groups = frame[columns].astype(float), frame.subject_id
    folds = []
    for task in ['multiclass','binary']:
        y = frame.label_original if task == 'multiclass' else (frame.label_original == 2).astype(int)
        labels = [1,2,3] if task == 'multiclass' else [0,1]
        for subject in SUBJECTS:
            checkpoint = work/f'{task}_{subject}.json'
            if checkpoint.exists():
                body = json.loads(checkpoint.read_text())
                assert body['signature'] == sig
                assert body['digest'] == fingerprint({k:v for k,v in body.items() if k!='digest'})
                folds.append(body); continue
            train, test = np.flatnonzero(groups.ne(subject)), np.flatnonzero(groups.eq(subject))
            inner_scores, splits = [], []
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter('always', ConvergenceWarning)
                for a,b in GroupKFold(3).split(X.iloc[train], y.iloc[train], groups.iloc[train]):
                    ia,ib = train[a],train[b]
                    model = clone(pipe).fit(X.iloc[ia], y.iloc[ia])
                    inner_scores.append(score(y.iloc[ib], model.predict(X.iloc[ib]), labels)['macro_f1'])
                    splits.append({'train':sorted(set(groups.iloc[ia])), 'validation':sorted(set(groups.iloc[ib]))})
                model = clone(pipe).fit(X.iloc[train], y.iloc[train])
                predicted = model.predict(X.iloc[test])
                fitted = model.predict(X.iloc[train])
            assert not caught, [str(w.message) for w in caught]
            metric = dict(task=task, model='svm', subject_id=subject, n_test=len(test),
                **score(y.iloc[test], predicted, labels), train_macro_f1=score(y.iloc[train], fitted, labels)['macro_f1'],
                inner_macro_f1=float(np.mean(inner_scores)))
            pred = frame.iloc[test][['subject_id','window_id']].copy()
            pred['task'],pred['model'],pred['y_true'],pred['y_pred'] = task,'svm',y.iloc[test].to_numpy(),predicted
            body = dict(signature=sig, metric=metric, predictions=pred.to_dict(orient='records'),
                detail=dict(train_subjects=sorted(set(groups.iloc[train])),test_subject=subject,
                inner_splits=splits,inner_scores=inner_scores,warnings=[],
                confusion_matrix=confusion_matrix(y.iloc[test], predicted, labels=labels).tolist()))
            body['digest'] = fingerprint(body)
            write_json(checkpoint, body)
            folds.append(body)
            print(f'{run} {task}/{subject}: F1={metric["macro_f1"]:.4f}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([f['metric'] for f in folds])
    predictions = pd.DataFrame([p for f in folds for p in f['predictions']])
    metrics.to_csv(out/'results.csv',index=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    write_json(out/'fold_details.json', {f'{f["metric"]["task"]}/{f["metric"]["subject_id"]}':f['detail'] for f in folds})
    write_json(out/'config.json',json.loads(manifest.read_text()))
    summary = metrics.groupby('task')[['macro_f1','balanced_accuracy']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv',index=False)
    fig,ax = plt.subplots(figsize=(6,4))
    ax.bar(summary.index, summary.macro_f1_mean,yerr=summary.macro_f1_std,capsize=5)
    ax.set(ylim=(0,1.05), ylabel='Macro-F1 mean +/- subject SD',title=f'EXP-008/{run}: C={spec["C"]}, gamma={spec["gamma"]}')
    fig.tight_layout(); fig.savefig(out/'summary.png',dpi=140); plt.close(fig)
    md = f'# EXP-008 / {run}\n\n## Hypothesis\nC={spec["C"]}, gamma={spec["gamma"]} changes RBF margin/locality; improvement is not guaranteed.\n\n## Protocol\n68 features; median imputation and scaling fit inside train only. 15 outer LOSO subjects, two tasks, inner GroupKFold(3). One fixed candidate per run; six candidates compared by inner Macro-F1 in the final report. Not six independent search algorithms.\n\n## Results\n{summary.to_string()}\n\n## Interpretation\nUse results.csv for subject variation; no selection on outer test scores.\n\n## Limitations\nSmall reused laboratory benchmark; no independent external holdout. Earlier G4 results remain historical.\n'
    code = "from pathlib import Path\nimport pandas as pd\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/exp008_thai.yaml').exists())\n"+f"OUT = ROOT / {out.relative_to(ROOT).as_posix()!r}\nprint(pd.read_csv(OUT/'summary.csv').to_string(index=False))\n# Reproduce: .venv/Scripts/python.exe scripts/exp008_thai.py --run {run}\n"
    write_json(out/'experiment.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'id':'research','cell_type':'markdown','metadata':{},'source':md.splitlines(True)},
        {'id':'results','cell_type':'code','metadata':{'output_origin':'Generated from verified saved results, not a Jupyter cell execution'},'source':code.splitlines(True),'execution_count':None,
         'outputs':[{'output_type':'stream','name':'stdout','text':summary.to_string().splitlines(True)}]}]})
    write_json(out/'acceptance.json',{'signature':sig,'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
        'files':{p.name:file_hash(p) for p in out.iterdir() if p.name!='acceptance.json'}})
    checked = verify(run)
    log = ROOT/'experiments/experiments_logs.xlsx'
    assert not log.with_name('~$'+log.name).exists(), 'Close Excel before updating log'
    book = load_workbook(log); sheet = book['Experiment Log']
    existing = {r[0] for r in sheet.iter_rows(min_row=2, values_only=True)}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    rel = out.relative_to(ROOT).as_posix()
    for task,row in summary.iterrows():
        key = f'EXP-008/{run}/{task}'
        if key not in existing:
            sheet.append([key,datetime.now(timezone.utc).isoformat(),'Thai','tuning',task,'SVM','physiological68',
                'LOSO + inner GroupKFold(3)',row.macro_f1_mean,row.balanced_accuracy_mean,
                'Verified 15 outer folds; fixed C/gamma candidate',rel,rel+'/experiment.ipynb',rel+'/config.json',commit,'EXP-008'])
    book.save(log)
    print(json.dumps(checked),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    if args.verify: print(json.dumps(verify(args.run)))
    else: run_one(args.run)
