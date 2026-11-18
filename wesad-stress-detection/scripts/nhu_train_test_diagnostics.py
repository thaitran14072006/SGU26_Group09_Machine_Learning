"""Refit cấu hình đã chốt để đo train/test cùng mô hình; không chọn tham số mới."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import sys
import warnings
import zipfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
    from threadpoolctl import threadpool_limits
    from wesad.data import SUBJECTS,load_feature_table
    from wesad.evaluation import task_target
    from wesad.models import build_pipeline
    from wesad.utils import file_hash,environment,write_json,write_csv,read_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',choices=['006','010','015'],default=['006','010','015'])
    args=parser.parse_args()
    now=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    out=ROOT/'outputs'/now.strftime('nhu_train_test_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    data=ROOT/'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
    schema=data.with_name('feature_schema.json')
    frame,columns=load_feature_table(data,schema,SUBJECTS)
    assert len(frame)==6189 and len(columns)==68
    digest=file_hash(data)
    assert digest=='2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba'
    bases={
        '006':ROOT/'experiments/01_baselines/EXP-006/results',
        '010':ROOT/'experiments/02_tuning/EXP-010/results',
        '015':ROOT/'experiments/03_balancing/EXP-015/results',
    }
    metadata=dict(date=now.strftime('%d/%m/%Y'),scope='supplementary_refit_diagnostics',
        member='như',experiments=args.experiments,data_sha256=digest,feature_columns=columns,
        environment=environment(),seed=42,source_sha256=file_hash(__file__),
        interpretation='train and outer test metrics from the same refitted model; no tuning or model selection',
        train_scoring='original outer train before ROS duplication; predict skips sampler',
        test_scoring='held-out subject; not used to select hyperparameters or stopping epoch')
    write_json(out/'run_info.json',metadata)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in ['scripts/nhu_train_test_diagnostics.py','src/wesad/data.py',
                     'src/wesad/evaluation.py','src/wesad/models.py','src/wesad/utils.py']:
            z.write(ROOT/path,path)
    X=frame[columns].astype(float)
    print('output:',out,flush=True)
    total=0
    for exp in args.experiments:
        reference=bases[exp]
        destination=out/f'EXP-{exp}';destination.mkdir()
        (destination/'folds').mkdir()
        if exp=='006':
            info=read_json(reference/'run_info.json')
            assert info['data_sha256']==digest
            config=dict(seed=42,balance=False,model_params={'mlp':info['pipeline']['mlp']})
            fixed={}
            reference_info={'path':str(reference.relative_to(ROOT)),
                'run_info_sha256':file_hash(reference/'run_info.json'),
                'metrics_sha256':file_hash(reference/'metrics_by_subject.csv'),
                'original_pandas':info['pandas']}
        else:
            manifest=read_json(reference/'config.json')
            assert manifest['data_sha256']==digest
            config=manifest['settings']
            fixed=read_json(reference/'best_params.json')
            reference_info={'path':str(reference.relative_to(ROOT)),
                'config_sha256':file_hash(reference/'config.json'),
                'predictions_sha256':file_hash(reference/'predictions.csv'),
                'best_params_sha256':file_hash(reference/'best_params.json')}
        assert config['seed']==42 and bool(config['balance'])==(exp=='015')
        write_json(destination/'reference.json',reference_info)
        write_json(destination/'config.json',config)
        metrics=[];predictions=[]
        ref_metrics=pd.read_csv(reference/'metrics_by_subject.csv').rename(columns={'test_subject':'subject_id'})
        if exp!='006':all_reference=pd.read_csv(reference/'predictions.csv',dtype={'window_id':str,'subject_id':str})
        for task in ['multiclass','binary']:
            y,labels=task_target(frame,task)
            for subject in SUBJECTS:
                train=frame.subject_id.ne(subject);test=~train
                assert frame.loc[train,'subject_id'].nunique()==14
                assert frame.loc[test,'subject_id'].nunique()==1
                key=f'{task}/mlp/{subject}'
                params=fixed[key] if fixed else {}
                estimator=build_pipeline('mlp',config)
                if params:estimator.set_params(**params)
                start=perf_counter()
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                    warnings.simplefilter('always')
                    estimator.fit(X.loc[train],y.loc[train])
                    train_pred=estimator.predict(X.loc[train])
                    test_pred=estimator.predict(X.loc[test])
                pred=frame.loc[test,['subject_id','window_id']].copy()
                pred['y_true']=y.loc[test].to_numpy();pred['y_pred']=test_pred
                pred['task']=task;pred['model']='mlp'
                if exp=='006':
                    ref=pd.read_csv(reference/f'predictions_{task}_{subject}.csv')
                    assert set(ref.row_index)==set(frame.index[test])
                    ref['window_id']=frame.loc[ref.row_index,'window_id'].to_numpy()
                    reference_pred=ref[['subject_id','window_id','y_true','y_pred']]
                else:
                    reference_pred=all_reference[(all_reference.task==task)&(all_reference.subject_id==subject)][['subject_id','window_id','y_true','y_pred']]
                joined=reference_pred.merge(pred,on=['subject_id','window_id'],how='outer',
                    suffixes=('_reference','_refit'),validate='one_to_one',indicator=True)
                assert joined['_merge'].eq('both').all() and joined.y_true_reference.eq(joined.y_true_refit).all()
                differences=int(joined.y_pred_reference.ne(joined.y_pred_refit).sum())
                ref_row=ref_metrics[(ref_metrics.task==task)&(ref_metrics.subject_id==subject)].iloc[0]
                train_f1=float(f1_score(y.loc[train],train_pred,labels=labels,average='macro',zero_division=0))
                test_f1=float(f1_score(y.loc[test],test_pred,labels=labels,average='macro',zero_division=0))
                train_ba=float(balanced_accuracy_score(y.loc[train],train_pred))
                test_ba=float(balanced_accuracy_score(y.loc[test],test_pred))
                row=dict(experiment=f'EXP-{exp}',task=task,subject_id=subject,n_train=int(train.sum()),n_test=int(test.sum()),
                    train_macro_f1=train_f1,test_macro_f1=test_f1,
                    train_balanced_accuracy=train_ba,test_balanced_accuracy=test_ba,
                    macro_f1_gap=train_f1-test_f1,balanced_accuracy_gap=train_ba-test_ba,
                    reference_test_macro_f1=float(ref_row.macro_f1),
                    reference_test_balanced_accuracy=float(ref_row.balanced_accuracy),
                    reference_prediction_differences=differences,
                    reference_metrics_match=bool(np.isclose(test_f1,ref_row.macro_f1,atol=1e-12,rtol=0)
                        and np.isclose(test_ba,ref_row.balanced_accuracy,atol=1e-12,rtol=0)),
                    n_iter=int(estimator['model'].n_iter_),seconds=round(perf_counter()-start,4))
                fold=dict(**row,labels=labels,best_params=params,
                    outer_train_subjects=sorted(set(frame.loc[train,'subject_id'])),outer_test_subjects=[subject],
                    train_confusion_matrix=confusion_matrix(y.loc[train],train_pred,labels=labels).tolist(),
                    test_confusion_matrix=confusion_matrix(y.loc[test],test_pred,labels=labels).tolist(),
                    warnings=[str(w.message) for w in caught])
                write_json(destination/'folds'/f'{task}_{subject}.json',fold)
                write_csv(destination/'folds'/f'{task}_{subject}_predictions.csv',pred)
                metrics.append(row);predictions.append(pred)
                write_csv(destination/'metrics.csv',pd.DataFrame(metrics))
                total+=1
                print(f'{total}/{len(args.experiments)*30} EXP-{exp} {task}/{subject}: train={train_f1:.4f}, test={test_f1:.4f}, old_prediction_diff={differences}',flush=True)
        table=pd.DataFrame(metrics)
        write_csv(destination/'predictions_test.csv',pd.concat(predictions,ignore_index=True))
        summary=table.groupby('task').agg(n_subjects=('subject_id','nunique'),
            train_macro_f1_mean=('train_macro_f1','mean'),test_macro_f1_mean=('test_macro_f1','mean'),
            train_balanced_accuracy_mean=('train_balanced_accuracy','mean'),test_balanced_accuracy_mean=('test_balanced_accuracy','mean'),
            macro_f1_gap_mean=('macro_f1_gap','mean'),balanced_accuracy_gap_mean=('balanced_accuracy_gap','mean')).reset_index()
        write_csv(destination/'summary.csv',summary)
        assert len(table)==30
        verification=dict(status='passed',outer_folds=30,reference_prediction_differences=int(table.reference_prediction_differences.sum()),
            all_reference_metrics_match=bool(table.reference_metrics_match.all()),
            all_reference_predictions_match=bool(table.reference_prediction_differences.eq(0).all()),
            training_scoring='original outer train, before oversampling duplicates',test_used_for_selection=False)
        write_json(destination/'verification.json',verification)
    print('DIAGNOSTICS COMPLETE:',out,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
