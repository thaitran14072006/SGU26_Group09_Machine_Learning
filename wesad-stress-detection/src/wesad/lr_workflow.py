"""Thực nghiệm LR theo giai đoạn, chọn cấu hình bằng inner CV theo người."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from threadpoolctl import threadpool_limits
import yaml

from wesad.data import SUBJECTS, load_feature_table
from wesad.utils import PROJECT_ROOT as ROOT, environment, file_hash, fingerprint

PLAN_PATH = ROOT / 'configs/lr_sequential_workflow.yaml'
STATE_PATH = ROOT / 'experiments/lr_workflow_state.json'
TASKS = ('multiclass', 'binary')
STAGES = ('feature_selection', 'balancing', 'tuning')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def plan():
    value = yaml.safe_load(PLAN_PATH.read_text(encoding='utf-8'))
    if value['order'] != list(STAGES):
        raise ValueError('Feature Selection phải hoàn tất trước cân bằng lớp và tuning.')
    ids = [r['run_id'] for r in value['experiments']]
    if len(ids) != len(set(ids)):
        raise ValueError('Trùng mã run.')
    for stage in STAGES:
        if not 3 <= sum(r['stage'] == stage for r in value['experiments']) <= 5:
            raise ValueError('Mỗi giai đoạn cần 3–5 thí nghiệm.')
    return value


def folder(spec):
    bases = {'EXP-016': '04_additional/EXP-016_LR_Features',
             'EXP-011': '03_balancing/EXP-011_LR',
             'supplementary/LR_Regularization': 'supplementary/LR_Regularization'}
    return ROOT / 'experiments' / bases[spec['experiment_id']] / 'runs' / spec['run_id']


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Giữ cột đầu tiên theo schema, loại cột tương quan cao với cột đã giữ."""
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
        kept = []
        for j in range(self.n_features_in_):
            if not kept or not np.any(corr[j, kept] > self.threshold):
                kept.append(j)
        self.support_ = np.isin(np.arange(self.n_features_in_), kept)
        return self

    def transform(self, X):
        check_is_fitted(self, 'support_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Sai số cột khi transform.')
        return X[:, self.support_]

    def get_support(self):
        check_is_fitted(self, 'support_')
        return self.support_.copy()


class WeightedLR(ClassifierMixin, BaseEstimator):
    """Tính trọng số lớp chỉ từ nhãn của lần fit hiện tại."""
    def __init__(self, C=1.0, weight_policy='none', max_iter=2000, random_state=42):
        self.C, self.weight_policy = C, weight_policy
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        if self.weight_policy == 'none':
            weights = None
        elif self.weight_policy == 'balanced':
            weights = 'balanced'
        elif self.weight_policy == 'sqrt_balanced':
            weights = {int(c): float(np.sqrt(len(y) / (len(classes) * n)))
                       for c, n in zip(classes, counts)}
        else:
            raise ValueError('Trọng số lớp không hợp lệ.')
        self.fit_class_weights_ = weights
        self.estimator_ = LogisticRegression(C=self.C, class_weight=weights,
            max_iter=self.max_iter, random_state=self.random_state, solver='lbfgs').fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def selector(method, seed):
    if method == 'anova':
        return SelectKBest(f_classif)
    if method == 'mutual_info':
        return SelectKBest(partial(mutual_info_classif, random_state=seed))
    if method == 'correlation':
        return CorrelationSelector()
    raise ValueError('Không có phương pháp chọn đặc trưng này.')


def build(spec, cfg, frozen=None):
    recipe = dict(frozen or {})
    if spec['stage'] == 'feature_selection':
        method = spec['method']
        key = 'selection__threshold' if method == 'correlation' else 'selection__k'
        grid = {key: spec['candidates']}
    else:
        method, grid = recipe['method'], {}
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('selection', selector(method, cfg['seed'])),
        ('model', WeightedLR(C=spec.get('C', 1.0),
            weight_policy=spec.get('weight_policy', recipe.get('weight_policy', 'none')),
            max_iter=cfg['max_iter'], random_state=cfg['seed']))])
    if frozen:
        pipe.set_params(**recipe['selection_params'])
    return pipe, grid, method


def grouped_splits(X, y, groups, outer_subject):
    splits = list(GroupKFold(n_splits=3).split(X, y, groups))
    evidence = []
    for a, b in splits:
        ga, gb = set(groups.iloc[a]), set(groups.iloc[b])
        if ga & gb or outer_subject in ga | gb:
            raise ValueError('Phát hiện rò rỉ người giữa các tập.')
        evidence.append({'train_subjects': sorted(ga), 'validation_subjects': sorted(gb)})
    return splits, evidence


def winner(records):
    """Xếp hạng theo inner score, không đọc các điểm outer test."""
    return max(records, key=lambda r: r['inner_macro_f1'])


def validate_completed(spec):
    target = folder(spec)
    receipt = load(target / 'acceptance.json')
    if receipt['status'] != 'complete' or receipt['folds'] != 30:
        raise ValueError('Chưa hoàn thành đủ fold.')
    for name, digest in receipt['files'].items():
        if file_hash(target / name) != digest:
            raise ValueError(f'Bằng chứng đã thay đổi: {spec["run_id"]}/{name}')
    return receipt


def freeze(stage, cfg):
    specs = [r for r in cfg['experiments'] if r['stage'] == stage]
    for spec in specs:
        receipt = validate_completed(spec)
        manifest = load(folder(spec) / 'config.json')
        signed = {k: manifest[k] for k in ('protocol','run','dependency','source_sha256','environment')}
        if (manifest['protocol'] != cfg or manifest['run'] != spec
                or receipt['data_sha256'] != cfg['feature_sha256']
                or fingerprint(signed) != receipt['signature']):
            raise ValueError('Hồ sơ không thuộc đúng phiên bản quy trình hiện hành.')
    completed = [load(folder(s) / 'fold_details.json') for s in specs]
    recipes = {}
    for task in TASKS:
        for subject in SUBJECTS:
            key = f'{task}/{subject}'
            candidates = [d[key] for d in completed]
            if len({fingerprint(c['inner_splits']) for c in candidates}) != 1:
                raise ValueError('Các run dùng inner split khác nhau.')
            best = winner(candidates)
            recipes[key] = {k: best[k] for k in ('method', 'selection_params', 'weight_policy', 'C')}
            recipes[key].update({'source_run': best['run_id'], 'inner_macro_f1': best['inner_macro_f1']})
    payload = {'stage': stage, 'decision_metric': 'inner_macro_f1',
               'source_receipts': {s['run_id']: file_hash(folder(s) / 'acceptance.json') for s in specs},
               'recipes': recipes}
    path = ROOT / 'experiments' / f'lr_{stage}_gate.json'
    if path.exists() and load(path) != payload:
        raise ValueError('Quyết định đã khóa thay đổi; cần phiên bản quy trình mới.')
    if not path.exists():
        dump(path, payload)
    return payload


def prerequisites(stage, cfg):
    if stage == 'feature_selection':
        return None
    previous = 'feature_selection' if stage == 'balancing' else 'balancing'
    if previous == 'balancing':
        freeze('feature_selection', cfg)
    return freeze(previous, cfg)


def audit_baselines(cfg, frame):
    reports = []
    for base in ['01_baselines/EXP-001_Dummy/runs/20261114', '01_baselines/EXP-002_LR']:
        target = ROOT / 'experiments' / base
        if load(target / 'config.json')['data_sha256'] != cfg['feature_sha256']:
            raise ValueError('Baseline khác dữ liệu.')
        metrics = pd.read_csv(target / 'results.csv')
        predictions = pd.read_csv(target / 'predictions.csv')
        if len(metrics) != 30 or metrics.duplicated(['task', 'subject_id']).any():
            raise ValueError('Baseline thiếu hoặc trùng fold.')
        for task in TASKS:
            part = predictions[predictions.task == task]
            if len(part) != len(frame):
                raise ValueError('Sai số lượng dự đoán baseline.')
            merged = frame[['subject_id', 'window_id', 'label_original']].merge(
                part, on=['subject_id', 'window_id'], validate='one_to_one')
            expected = (merged.label_original == 2).astype(int) if task == 'binary' else merged.label_original
            if len(merged) != len(frame) or not np.array_equal(expected, merged.y_true):
                raise ValueError('Nhãn hoặc khóa baseline không khớp dữ liệu.')
            labels = [0, 1] if task == 'binary' else [1, 2, 3]
            for subject in SUBJECTS:
                p = part[part.subject_id == subject]
                row = metrics[(metrics.task == task) & (metrics.subject_id == subject)].iloc[0]
                score = f1_score(p.y_true, p.y_pred, labels=labels, average='macro', zero_division=0)
                if not np.isclose(score, row.macro_f1, atol=1e-10):
                    raise ValueError('Điểm baseline không khớp dự đoán.')
                if len(p) != row.n_test or not np.isclose(balanced_accuracy_score(p.y_true,p.y_pred),row.balanced_accuracy,atol=1e-10):
                    raise ValueError('Balanced Accuracy hoặc số cửa sổ baseline sai.')
        reports.append({'folder': base, 'status': 'verified', 'results_sha256': file_hash(target/'results.csv')})
    return reports


def identity(spec, cfg, dependency):
    sources = [Path(__file__), ROOT/'src/wesad/data.py', ROOT/'src/wesad/utils.py',
               ROOT/'scripts/run_lr_workflow.py']
    return {'protocol': cfg, 'run': spec, 'dependency': dependency,
        'source_sha256': fingerprint({p.relative_to(ROOT).as_posix():file_hash(p) for p in sources}),
        'environment': environment()}


def notebook(spec, out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    means = summary.set_index('task')['macro_f1_mean']
    errors = summary.set_index('task')['macro_f1_std']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index, means, yerr=errors, capsize=5, color=['#2c7fb8', '#41ab5d'])
    ax.set(ylim=(0, 1.05), ylabel='Macro-F1', title=f'{spec["experiment_id"]} — {spec["run_id"]}')
    fig.tight_layout()
    fig.savefig(out / 'summary_metrics.png', dpi=150)
    plt.close(fig)
    rel = out.relative_to(ROOT).as_posix()
    cells = []
    def md(s):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)})
    def code(s, outputs=None):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(keepends=True),
                      'execution_count': None,
                      'outputs': outputs or []})
    md(f'# {spec["experiment_id"]} — {spec["run_id"]}\n\n## Mục tiêu\nĐánh giá tác động của {spec.get("method", spec.get("weight_policy", "C="+str(spec.get("C"))))} trong quy trình LR theo người.\n\n## Giả thuyết\nThay đổi cấu hình có thể cải thiện khả năng khái quát, nhưng mức thay đổi phụ thuộc từng người.')
    md('## Thiết lập\n- 6.189 cửa sổ, 68 đặc trưng đầu vào và 15 người.\n- Hai bài toán: multiclass và binary.\n- Outer LOSO, inner GroupKFold 3 phần.\n- Median, scaling và selection chỉ fit trên train trong từng lần CV.\n- LR lbfgs, max_iter=2000, seed=42.\n- Cấu hình lựa chọn dựa trên inner Macro-F1, outer test chỉ dùng đánh giá.\n- Điểm tổng hợp là mean ± SD theo người, không phải theo cửa sổ.')
    code("from pathlib import Path\nimport sys\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/wesad/lr_workflow.py').is_file())\nsys.path.insert(0, str(ROOT / 'src'))\nfrom wesad.lr_workflow import run_one\nRUN_TRAIN = False\n"+f"FOLDER = ROOT / {rel!r}\nif RUN_TRAIN:\n    run_one({spec['run_id']!r})\n")
    md('## Kết quả LOSO')
    code("metrics = pd.read_csv(FOLDER / 'results.csv')\nassert len(metrics) == 30\ndisplay(metrics.groupby('task')[['macro_f1', 'balanced_accuracy']].agg(['mean', 'std']))",
         [{'output_type':'stream','name':'stdout','text':[summary.to_string(index=False)+'\n']}])
    code("display(Image(filename=str(FOLDER / 'summary_metrics.png')))",[
        {'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode((out/'summary_metrics.png').read_bytes()).decode(),'text/plain':['Macro-F1 theo bài toán; cột sai số là SD giữa người.']}}])
    findings = '\n'.join(f'- {r.task}: Macro-F1 {r.macro_f1_mean:.6f} ± {r.macro_f1_std:.6f}; Balanced Accuracy {r.balanced_accuracy_mean:.6f} ± {r.balanced_accuracy_std:.6f}.' for r in summary.itertuples())
    md('## Nhận xét\n'+findings+'\n\nKết quả tổng hợp không đồng nghĩa cải thiện trên mọi người. Đối chiếu `results.csv` theo subject và `fold_details.json` để xem đặc trưng, cảnh báo và cấu hình trong từng fold.\n\n## Giới hạn\nDữ liệu chỉ gồm 15 người trong phòng thí nghiệm. Chưa có tập kiểm tra ngoài WESAD.\n\n## Quyết định tiếp theo\nHoàn tất tập phương án đã định cho giai đoạn hiện tại. Chỉ chuyển bước khi đủ kết quả và kiểm tra chất lượng; không chọn cấu hình theo điểm outer test.')
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i+1}'
    dump(out/'experiment.ipynb', {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5})


def append_log(spec, out, summary):
    from openpyxl import load_workbook
    path = ROOT/'experiments/experiments_logs.xlsx'
    if path.with_name('~$'+path.name).exists():
        raise RuntimeError('Nhật ký Excel đang mở. Đóng Excel để cập nhật kết quả.')
    book = load_workbook(path)
    sheet = book['Experiment Log']
    existing = {r[0] for r in sheet.iter_rows(min_row=2, values_only=True)}
    stamp = datetime.now(timezone.utc).isoformat()
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True,text=True,check=True).stdout.strip()
    changed = False
    for row in summary.itertuples():
        key = f'{spec["experiment_id"]}/{spec["run_id"]}/{row.task}'
        if key not in existing:
            rel = out.relative_to(ROOT).as_posix()
            sheet.append([key, stamp, 'Thái', spec['stage'], row.task, 'LR','physiological68','LOSO + inner GroupKFold(3)',
                row.macro_f1_mean,row.balanced_accuracy_mean,'Hoàn thành 15 fold; lựa chọn theo inner CV',
                rel,rel+'/experiment.ipynb',rel+'/config.json',commit,spec['experiment_id']])
            changed = True
    if changed:
        book.save(path)


def publish(spec, cfg, work, signature):
    out = folder(spec)
    out.mkdir(parents=True,exist_ok=True)
    records, predictions, details = [], [], {}
    for task in TASKS:
        for subject in SUBJECTS:
            f = load(work/'folds'/f'{task}_{subject}.json')
            body = {k:v for k,v in f.items() if k!='sha256'}
            if f['sha256'] != fingerprint(body) or f['signature'] != signature:
                raise ValueError('Fold bị thay đổi hoặc khác danh tính run.')
            if f['detail']['fatal_warnings']:
                raise ValueError('Chưa hội tụ. Dừng ở giai đoạn hiện tại.')
            records.append(f['metric'])
            predictions.extend(f['predictions'])
            details[f'{task}/{subject}'] = f['detail']
    metrics, pred = pd.DataFrame(records), pd.DataFrame(predictions)
    if len(metrics)!=30 or len(pred)!=12378 or pred.duplicated(['task','subject_id','window_id']).any():
        raise ValueError('Thiếu hoặc trùng kết quả LOSO.')
    for task in TASKS:
        if set(metrics.loc[metrics.task==task,'subject_id']) != set(SUBJECTS):
            raise ValueError('Sai danh sách người test.')
    fields = ['macro_f1','balanced_accuracy']
    summary = metrics.groupby('task')[fields].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    metrics.to_csv(out/'results.csv',index=False)
    pred.to_csv(out/'predictions.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    dump(out/'fold_details.json',details)
    dump(out/'config.json',load(work/'manifest.json'))
    notebook(spec,out,summary)
    append_log(spec,out,summary)
    files = ['results.csv','predictions.csv','summary.csv','fold_details.json','config.json','experiment.ipynb','summary_metrics.png']
    dump(out/'acceptance.json',{'status':'complete','folds':30,'data_sha256':cfg['feature_sha256'],
         'signature':signature,'completed_at':datetime.now(timezone.utc).isoformat(),
         'files':{n:file_hash(out/n) for n in files}})
    return out


@contextmanager
def lock():
    target = ROOT/'outputs/lr_workflow/active.lock'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps({'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat()}))
    try:
        yield
    finally:
        target.unlink(missing_ok=True)


def run_one(run_id=None):
    with lock():
        return _run_one(run_id)


def _run_one(run_id):
    cfg = plan()
    defaults = yaml.safe_load((ROOT/'configs/defaults.yaml').read_text(encoding='utf-8'))
    table = ROOT/defaults['features']
    if file_hash(table)!=cfg['feature_sha256']:
        raise ValueError('Checksum dữ liệu khác bảng đã chốt.')
    frame, columns = load_feature_table(table,ROOT/defaults['schema'],SUBJECTS)
    if len(frame)!=6189 or len(columns)!=68:
        raise ValueError('Sai kích thước dữ liệu.')
    baseline_audit = audit_baselines(cfg,frame)
    pending = []
    for spec in cfg['experiments']:
        if (folder(spec)/'acceptance.json').exists():
            validate_completed(spec)
        else:
            pending.append(spec)
    if not pending:
        freeze('tuning',cfg)
        write_state(cfg)
        print('WORKFLOW_COMPLETE',flush=True)
        return None
    spec = next((s for s in cfg['experiments'] if s['run_id']==run_id),None) if run_id else pending[0]
    if spec is None:
        raise ValueError('Không có run này.')
    if (folder(spec)/'acceptance.json').exists():
        return folder(spec)
    dependency = prerequisites(spec['stage'],cfg)
    work = ROOT/'outputs/lr_workflow'/spec['run_id']
    work.mkdir(parents=True,exist_ok=True)
    ident = identity(spec,cfg,dependency)
    signature = fingerprint(ident)
    if (work/'manifest.json').exists() and load(work/'manifest.json')['signature']!=signature:
        raise ValueError('Mã/cấu hình/môi trường đã đổi. Không ghi đè run; cần phiên bản mới.')
    if not (work/'manifest.json').exists():
        dump(work/'manifest.json',{'signature':signature,**ident,'baseline_audit':baseline_audit,
            'started_at':datetime.now(timezone.utc).isoformat()})
    X, groups = frame[columns].astype(float),frame.subject_id
    for task in TASKS:
        y = (frame.label_original==2).astype(int) if task=='binary' else frame.label_original.astype(int)
        labels = [0,1] if task=='binary' else [1,2,3]
        for subject in SUBJECTS:
            output = work/'folds'/f'{task}_{subject}.json'
            if output.exists():
                old = load(output)
                if old['signature']!=signature or old['sha256']!=fingerprint({k:v for k,v in old.items() if k!='sha256'}):
                    raise ValueError('Fold checkpoint khác chữ ký hoặc đã bị sửa.')
                continue
            train,test=np.flatnonzero(groups.ne(subject)),np.flatnonzero(groups.eq(subject))
            if groups.iloc[train].nunique()!=14 or set(groups.iloc[train])&set(groups.iloc[test]):
                raise ValueError('Sai cách chia LOSO.')
            recipe = dependency['recipes'][f'{task}/{subject}'] if dependency else None
            estimator,grid,method = build(spec,cfg,recipe)
            splits,evidence=grouped_splits(X.iloc[train],y.iloc[train],groups.iloc[train],subject)
            started=perf_counter()
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                warnings.simplefilter('always')
                search=GridSearchCV(estimator,grid,cv=splits,n_jobs=1,error_score='raise',refit=True,
                    scoring=make_scorer(f1_score,average='macro',labels=labels,zero_division=0),return_train_score=False)
                search.fit(X.iloc[train],y.iloc[train])
                predicted=search.predict(X.iloc[test])
                fitted=search.predict(X.iloc[train])
            fatal=sorted({str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)})
            if fatal or not np.isfinite(search.cv_results_['mean_test_score']).all():
                raise ValueError(f'Run chưa đạt chất lượng ở {task}/{subject}: {fatal}')
            chosen=np.array(columns)[search.best_estimator_['selection'].get_support()].tolist()
            if not chosen or set(chosen)-set(columns):
                raise ValueError('Bộ đặc trưng không hợp lệ.')
            metric={'task':task,'model':'lr','subject_id':subject,'n_test':len(test),
                'macro_f1':float(f1_score(y.iloc[test],predicted,labels=labels,average='macro',zero_division=0)),
                'balanced_accuracy':float(balanced_accuracy_score(y.iloc[test],predicted)),
                'train_macro_f1':float(f1_score(y.iloc[train],fitted,labels=labels,average='macro',zero_division=0)),
                'inner_macro_f1':float(search.best_score_),'selected_k':len(chosen),'seconds':perf_counter()-started}
            params={k:v for k,v in search.best_estimator_.get_params().items() if k in ['selection__k','selection__threshold']}
            detail={'run_id':spec['run_id'],'method':method,'selection_params':params,
                'weight_policy':search.best_estimator_['model'].weight_policy,'C':search.best_estimator_['model'].C,
                'inner_macro_f1':float(search.best_score_),'outer_train_subjects':sorted(set(groups.iloc[train])),
                'outer_test_subjects':[subject],'inner_splits':evidence,'selected_features':chosen,
                'inner_candidates':[{'params':param,'mean_macro_f1':float(mean),'std_macro_f1':float(sd)} for param,mean,sd in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score'])],
                'fatal_warnings':fatal,'warnings':sorted({str(w.message) for w in caught}),
                'confusion_matrix':confusion_matrix(y.iloc[test],predicted,labels=labels).tolist()}
            preds=frame.iloc[test][['subject_id','window_id']].copy()
            preds['task'],preds['model']=task,'lr'
            preds['y_true'],preds['y_pred']=y.iloc[test].to_numpy(),predicted
            body={'signature':signature,'metric':metric,'detail':detail,'predictions':preds.to_dict(orient='records')}
            dump(output,{**body,'sha256':fingerprint(body)})
            print(f'{spec["run_id"]} {task}/{subject}: {len(chosen)} features, inner={search.best_score_:.4f}',flush=True)
    out=publish(spec,cfg,work,signature)
    write_state(cfg)
    stage_specs=[s for s in cfg['experiments'] if s['stage']==spec['stage']]
    if all((folder(s)/'acceptance.json').exists() for s in stage_specs):
        freeze(spec['stage'],cfg)
        write_state(cfg)
    print('COMPLETED',spec['run_id'],out,flush=True)
    return out


def write_state(cfg):
    rows=[]
    for spec in cfg['experiments']:
        done=(folder(spec)/'acceptance.json').exists()
        rows.append({'run_id':spec['run_id'],'stage':spec['stage'],'experiment_id':spec['experiment_id'],
                     'status':'complete' if done else 'pending','folder':folder(spec).relative_to(ROOT).as_posix()})
    dump(STATE_PATH,{'protocol':cfg['protocol'],'plan_sha256':file_hash(PLAN_PATH),
        'updated_at':datetime.now(timezone.utc).isoformat(),'experiments':rows,
        'complete':all(r['status']=='complete' for r in rows)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default=None)
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        cfg=plan()
        write_state(cfg)
        print(STATE_PATH.read_text(encoding='utf-8'))
    else:
        run_one(args.run)


if __name__=='__main__':
    main()
