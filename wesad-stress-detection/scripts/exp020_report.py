"""Kiểm tra, so sánh và đóng gói kết quả thật của EXP-020."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import shutil
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from wesad.utils import read_json, read_config, write_json, write_csv, file_hash, environment
from wesad.data import SUBJECTS, load_feature_table
from wesad.evaluation import task_target
from wesad.models import build_pipeline
from exp020_train import verify

PACKAGE = ROOT/'experiments/04_additional/EXP-020'
REFERENCE = ROOT/'experiments/02_tuning/EXP-010/results'


def compare(run):
    current = pd.read_csv(run/'verified_metrics_by_subject.csv')
    rows = []
    for exp,folder in [('EXP-006',ROOT/'experiments/01_baselines/EXP-006/results'),('EXP-010',REFERENCE)]:
        old = pd.read_csv(folder/'metrics_by_subject.csv').rename(columns={'test_subject':'subject_id'})
        pair = current.merge(old,on=['task','subject_id'],validate='one_to_one',suffixes=('_new','_reference'))
        assert len(pair) == 30 and pair.n_test_new.eq(pair.n_test_reference).all()
        pair.insert(0,'reference',exp)
        for metric in ['macro_f1','balanced_accuracy']:
            pair['delta_'+metric] = pair[metric+'_new']-pair[metric+'_reference']
        rows.append(pair)
    paired = pd.concat(rows,ignore_index=True)
    write_csv(run/'comparison_by_subject.csv',paired)
    summary = []
    for (reference,task),part in paired.groupby(['reference','task']):
        for metric in ['macro_f1','balanced_accuracy']:
            d = part['delta_'+metric]
            summary.append(dict(reference=reference,task=task,metric=metric,
                reference_mean=float(part[metric+'_reference'].mean()),
                new_mean=float(part[metric+'_new'].mean()),new_std=float(part[metric+'_new'].std(ddof=1)),
                mean_delta=float(d.mean()),improved=int((d>1e-12).sum()),
                worse=int((d < -1e-12).sum()),equal=int((abs(d)<=1e-12).sum())))
    table = pd.DataFrame(summary)
    write_csv(run/'comparison_summary.csv',table)
    return current,paired,table


def refit_train_test(run):
    config = read_config(run/'config.yaml')
    assert environment() == read_json(run/'environment.json')
    frame,features = load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    params = read_json(run/'best_params.json')
    original = pd.read_csv(run/'predictions.csv',dtype={'window_id':str,'subject_id':str})
    rows = []
    folder = run/'train_test';folder.mkdir()
    for task in config['tasks']:
        y,labels = task_target(frame,task)
        for subject in SUBJECTS:
            train = frame.subject_id.ne(subject);test = ~train
            estimator = build_pipeline('mlp',config).set_params(**params[f'{task}/mlp/{subject}'])
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter('always')
                estimator.fit(frame.loc[train,features],y[train])
                a = estimator.predict(frame.loc[train,features])
                b = estimator.predict(frame.loc[test,features])
            ref = original[(original.task==task)&(original.subject_id==subject)]
            pred = frame.loc[test,['window_id']].copy();pred['predicted'] = b
            pair = pred.merge(ref,on='window_id',how='outer',validate='one_to_one',indicator=True)
            assert pair['_merge'].eq('both').all() and pair.predicted.eq(pair.y_pred).all()
            record = dict(task=task,subject_id=subject,n_train=int(train.sum()),n_test=int(test.sum()),
                train_macro_f1=float(f1_score(y[train],a,labels=labels,average='macro',zero_division=0)),
                test_macro_f1=float(f1_score(y[test],b,labels=labels,average='macro',zero_division=0)),
                train_balanced_accuracy=float(balanced_accuracy_score(y[train],a)),
                test_balanced_accuracy=float(balanced_accuracy_score(y[test],b)))
            rows.append(record)
            write_json(folder/f'{task}_{subject}.json',dict(**record,labels=labels,
                train_confusion_matrix=confusion_matrix(y[train],a,labels=labels).tolist(),
                test_confusion_matrix=confusion_matrix(y[test],b,labels=labels).tolist(),
                best_params=params[f'{task}/mlp/{subject}'],warnings=[str(w.message) for w in caught],
                reference_predictions_match=True))
    table = pd.DataFrame(rows)
    write_csv(folder/'metrics.csv',table)
    write_json(folder/'verification.json',dict(status='passed',scope='supplementary fixed-parameter refit',
        refits=30,all_test_predictions_match=True,source_sha256=file_hash(__file__),
        date=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%d/%m/%Y'),
        note='train/test from the same refit; not scores captured during original nested run'))
    return table


def repeatability(runs):
    reference = pd.read_csv(runs[0]/'predictions.csv').sort_values(['task','subject_id','window_id']).reset_index(drop=True)
    base_config = read_config(runs[0]/'config.yaml')
    table = []
    for i,run in enumerate(runs,1):
        verify(run)
        config = read_config(run/'config.yaml')
        for key in ['search_spaces','model_params','seed','nested','inner_splits','feature_sha256','balance']:
            assert config[key] == base_config[key]
        assert read_json(run/'environment.json') == read_json(runs[0]/'environment.json')
        # Launcher only changed to append a UUID after an actual folder collision.
        with zipfile.ZipFile(run/'source_snapshot.zip') as z,zipfile.ZipFile(runs[0]/'source_snapshot.zip') as b:
            for name in ['src/wesad/data.py','src/wesad/evaluation.py','src/wesad/models.py','src/wesad/utils.py']:
                assert z.read(name) == b.read(name)
        pred = pd.read_csv(run/'predictions.csv').sort_values(['task','subject_id','window_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(pred.drop(columns='y_pred'),reference.drop(columns='y_pred'))
        differences = int(pred.y_pred.ne(reference.y_pred).sum())
        assert read_json(run/'best_params.json') == read_json(runs[0]/'best_params.json')
        for s in read_json(run/'summary.json')['results']:
            table.append(dict(exp_chung='EXP-020',run_id=f'EXP-020-R{i:02}',
                date=read_json(run/'notebook_provenance.json')['date'],task=s['task'],seed=config['seed'],
                outer_folds=30,prediction_differences=differences,
                macro_f1_mean=s['macro_f1_mean'],macro_f1_std=s['macro_f1_std'],
                balanced_accuracy_mean=s['balanced_accuracy_mean'],balanced_accuracy_std=s['balanced_accuracy_std'],
                source_run=run.relative_to(ROOT).as_posix()))
    result = pd.DataFrame(table)
    write_csv(runs[0]/'repeatability.csv',result)
    write_json(runs[0]/'repeatability_verification.json',dict(status='passed',independent_runs=len(runs),
        outer_fits=30*len(runs),inner_fits=360*len(runs),
        all_predictions_match=bool(result.prediction_differences.eq(0).all()),
        interpretation='same-seed reproducibility, not variability across seeds',
        launcher_change='R02/R03 append UUID to avoid output folder collision; shared ML code and grid unchanged'))
    return result


def figures(run,metrics,paired,train_test):
    out=run/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,task in enumerate(['multiclass','binary']):
        part=paired[(paired.task==task)&(paired.reference=='EXP-010')].set_index('subject_id').loc[list(SUBJECTS)]
        for j,metric in enumerate(['macro_f1','balanced_accuracy']):
            ax=axes[i,j]
            ax.plot(SUBJECTS,part[metric+'_reference'],'o-',label='EXP-010')
            ax.plot(SUBJECTS,part[metric+'_new'],'s-',label='EXP-020')
            ax.set(title=f'{task}: {metric}',ylim=(0,1.05),xlabel='outer test subject')
            ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'exp010_vs_exp020.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,task in enumerate(['multiclass','binary']):
        part=train_test[train_test.task==task].set_index('subject_id').loc[list(SUBJECTS)]
        for j,metric in enumerate(['macro_f1','balanced_accuracy']):
            ax=axes[i,j]
            ax.plot(SUBJECTS,part['train_'+metric],'o-',label='train (14 subjects)')
            ax.plot(SUBJECTS,part['test_'+metric],'s-',label='test (held-out subject)')
            ax.set(title=f'{task}: {metric}',ylim=(0,1.05),xlabel='outer test subject')
            ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'train_vs_test.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    convergence=[];inners=[]
    for ax,task in zip(axes,['multiclass','binary']):
        for subject in SUBJECTS:
            folder=run/'folds'/task/'mlp'/subject
            detail=read_json(folder/'details.json');loss=detail['mlp_optimization']
            ax.plot(np.arange(1,len(loss['loss_curve'])+1),loss['loss_curve'],alpha=.7,label=subject)
            convergence.append(dict(task=task,subject_id=subject,n_iter=loss['n_iter'],max_iter=loss['max_iter'],
                final_train_loss=loss['loss'],reached_max_iter=loss['reached_max_iter'],
                warning_count=len(detail['warnings']),warnings=' | '.join(detail['warnings']),
                hidden_sizes=str(detail['best_params']['model__hidden_layer_sizes']),alpha=detail['best_params']['model__alpha']))
            inner=pd.read_csv(folder/'inner_cv.csv');inner.insert(0,'outer_subject',subject);inner.insert(0,'task',task);inners.append(inner)
        ax.set(title=f'{task}: training loss of selected refit',xlabel='epoch',ylabel='training objective');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,ncol=3)
    fig.savefig(out/'train_loss.png',dpi=130);plt.close(fig)
    convergence=pd.DataFrame(convergence);write_csv(run/'convergence.csv',convergence)
    write_csv(run/'inner_search_all.csv',pd.concat(inners,ignore_index=True))
    pred=pd.read_csv(run/'predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        part=pred[pred.task==task];labels=[1,2,3] if task=='multiclass' else [0,1]
        names=['baseline','stress','amusement'] if task=='multiclass' else ['non-stress','stress']
        ConfusionMatrixDisplay(confusion_matrix(part.y_true,part.y_pred,labels=labels),display_labels=names).plot(ax=ax,cmap='Blues',colorbar=False)
        ax.set_title(task)
    fig.savefig(out/'confusion_matrices.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        c=convergence[convergence.task==task].groupby(['hidden_sizes','alpha']).size()
        c.plot.bar(ax=ax,rot=25);ax.set(title=f'{task}: selected by inner CV',ylabel='outer folds',xlabel='hidden sizes / alpha')
    fig.savefig(out/'selected_candidates.png',dpi=130);plt.close(fig)
    return convergence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs=3,required=True)
    args=p.parse_args();runs=[r.resolve() for r in args.runs]
    assert len(set(runs))==3 and not PACKAGE.exists(),'không ghi đè EXP-020 đã chốt'
    repeatability(runs)
    metrics,paired,summary=compare(runs[0])
    train_test=refit_train_test(runs[0])
    convergence=figures(runs[0],metrics,paired,train_test)
    write_json(runs[0]/'report_environment.json',dict(core=environment(),python_executable=sys.executable,
        matplotlib=matplotlib.__version__,matplotlib_location=str(Path(matplotlib.__file__).parent),
        note='plot dependencies loaded from existing D drive environment after primary train site-packages'))
    results=PACKAGE/'results';PACKAGE.mkdir(parents=True)
    shutil.copytree(runs[0],results)
    for i,run in enumerate(runs[1:],2):shutil.copytree(run,results/'repeatability'/f'R{i:02}')
    shutil.copy2(runs[0]/'config.yaml',PACKAGE/'config.yaml')
    shutil.copy2(runs[0]/'metrics_by_subject.csv',PACKAGE/'results.csv')
    shutil.copytree(results/'figures',PACKAGE/'figures')
    with zipfile.ZipFile(results/'report_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(__file__,'scripts/exp020_report.py')
    manifest={p.relative_to(results).as_posix():file_hash(p) for p in results.rglob('*') if p.is_file()}
    write_json(results/'artifact_checksums.json',manifest)
    print('package:',PACKAGE,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
