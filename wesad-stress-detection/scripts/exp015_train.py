"""EXP-015: RandomOverSampler train-only cho MLP, giữ best_params từ EXP-010."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import argparse
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'experiments/02_tuning/EXP-010/results'
sys.path.insert(0,str(ROOT/'src'))


def check_reference():
    from wesad.utils import read_json,file_hash,environment
    info=read_json(REFERENCE/'verification.json')
    assert info['status']=='passed' and info['outer_folds']==30
    for name,digest in read_json(REFERENCE/'artifact_checksums.json').items():
        path=(REFERENCE/name).resolve()
        assert path.is_relative_to(REFERENCE.resolve()) and file_hash(path)==digest,name
    assert read_json(REFERENCE/'environment.json')==environment(), 'môi trường khác exp010'
    return read_json(REFERENCE/'best_params.json')


def prepare():
    import yaml
    from wesad.data import SUBJECTS,load_feature_table
    from wesad.models import build_pipeline
    from wesad.utils import read_config,file_hash,write_json
    params=check_reference()
    config=read_config(REFERENCE/'config.yaml')
    now=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    out=ROOT/'outputs'/now.strftime('exp015_mlp_balanced_%Y%m%d_%H%M%S_%f')
    assert len(params)==30
    assert set(params)=={f'{task}/mlp/{subject}' for task in config['tasks'] for subject in SUBJECTS}
    config.pop('search_spaces')
    config.update(protocol='exp015_mlp_balanced_fixed_exp010_v1',nested=False,balance=True,
        fixed_fold_params=params,output_dir=out.relative_to(ROOT).as_posix(),
        reference_params_sha256=file_hash(REFERENCE/'best_params.json'),
        reference_predictions_sha256=file_hash(REFERENCE/'predictions.csv'),
        reference_run='experiments/02_tuning/EXP-010/results')
    config['source_paths']=['src/wesad/data.py','src/wesad/evaluation.py',
        'src/wesad/models.py','src/wesad/utils.py','scripts/exp015_train.py']
    assert file_hash(ROOT/config['features'])==config['feature_sha256']
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    assert len(features)==68 and len(frame)==6189
    pipeline=build_pipeline('mlp',config)
    assert list(pipeline.named_steps)==['imputer','scaler','sampler','model']
    assert pipeline['sampler'].random_state==42
    out.mkdir(parents=True)
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    write_json(out/'notebook_provenance.json',dict(experiment='EXP-015',member='như',
        date=now.strftime('%d/%m/%Y'),run_type='new',reference_experiment='EXP-010',
        method='RandomOverSampler chỉ trên outer train; không tuning lại',
        outer_folds=30,model_fits=30,feature_sha256=config['feature_sha256']))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in config['source_paths']:archive.write(ROOT/path,path)
    return out,config


def sampler_audit(config,out):
    """Dựng lại preprocessing/ROS cùng train+seed; không fit thêm MLP."""
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from wesad.data import load_feature_table,SUBJECTS
    from wesad.evaluation import task_target
    from wesad.models import build_pipeline
    from wesad.utils import write_csv
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    rows=[]
    for task in config['tasks']:
        y,_=task_target(frame,task)
        for subject in SUBJECTS:
            train=(frame.subject_id!=subject).to_numpy()
            test=~train
            fit=build_pipeline('mlp',config).set_params(**config['fixed_fold_params'][f'{task}/mlp/{subject}'])
            prep=clone(fit[:-1])
            xr,yr=prep.fit_resample(frame.loc[train,features],y[train])
            indices=prep['sampler'].sample_indices_
            assert indices.min()>=0 and indices.max()<train.sum()
            assert np.array_equal(np.asarray(yr),y[train].to_numpy()[indices])
            before=Counter(y[train]);after=Counter(yr)
            assert set(after)==set(before) and set(after.values())=={max(before.values())}
            transformed_test=prep[:-1].transform(frame.loc[test,features])
            assert len(transformed_test)==test.sum()
            subjects=set(frame.loc[train,'subject_id'])
            assert len(subjects)==14 and subject not in subjects
            rows.append(dict(task=task,subject_id=subject,train_subjects=','.join(sorted(subjects)),
                n_train_before=int(train.sum()),n_train_after=len(yr),n_test_before=int(test.sum()),
                n_test_after=len(transformed_test),class_counts_before=json.dumps(dict(before),sort_keys=True),
                class_counts_after=json.dumps(dict(after),sort_keys=True),
                audit_scope='recreated_preprocessing_and_sampler_same_train_and_seed',passed=True))
    write_csv(out/'sampler_audit.csv',pd.DataFrame(rows))


def verify(out):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
    from wesad.data import SUBJECTS,load_feature_table
    from wesad.evaluation import task_target
    from wesad.utils import read_config,read_json,file_hash,write_json,write_csv
    config=read_config(out/'config.yaml')
    params=check_reference()
    assert config['balance'] is True and config['nested'] is False
    assert config['models']==['mlp'] and config['tasks']==['multiclass','binary']
    assert config['fixed_fold_params']==params and 'search_spaces' not in config
    assert config['reference_params_sha256']==file_hash(REFERENCE/'best_params.json')
    assert config['reference_predictions_sha256']==file_hash(REFERENCE/'predictions.csv')
    assert file_hash(ROOT/config['features'])==config['feature_sha256']
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    assert len(features)==68
    manifest=read_json(out/'config.json');summary=read_json(out/'summary.json')
    assert summary['status']=='complete' and read_json(out/'best_params.json')==params
    all_pred=pd.read_csv(out/'predictions.csv',dtype={'subject_id':str,'window_id':str})
    plain=pd.read_csv(REFERENCE/'predictions.csv',dtype={'subject_id':str,'window_id':str})
    key=['task','model','subject_id','window_id','sensor_set','normalization']
    paired=plain.merge(all_pred,on=key,how='outer',validate='one_to_one',indicator=True,suffixes=('_plain','_balanced'))
    assert len(paired)==12378 and paired['_merge'].eq('both').all()
    assert paired.y_true_plain.eq(paired.y_true_balanced).all()
    records=[]
    for task in config['tasks']:
        y,labels=task_target(frame,task)
        for subject in SUBJECTS:
            folder=out/'folds'/task/'mlp'/subject
            complete=read_json(folder/'complete.json')
            assert complete['signature']==manifest['signature']
            for name,digest in complete['files'].items():
                path=(folder/name).resolve()
                assert path.is_relative_to(folder.resolve()) and file_hash(path)==digest
            detail=read_json(folder/'details.json');metric=read_json(folder/'metrics.json')
            assert detail['best_params']==params[f'{task}/mlp/{subject}']
            assert detail['inner_splits']==[]
            assert set(detail['outer_train_subjects'])==set(SUBJECTS)-{subject}
            assert detail['outer_test_subjects']==[subject]
            pred=pd.read_csv(folder/'predictions.csv',dtype={'subject_id':str,'window_id':str})
            reference=frame.loc[frame.subject_id==subject,['subject_id','window_id']].copy()
            reference['expected']=y[frame.subject_id==subject].to_numpy()
            joined=reference.merge(pred,on=['subject_id','window_id'],how='outer',validate='one_to_one',indicator=True)
            assert joined['_merge'].eq('both').all() and joined.expected.eq(joined.y_true).all()
            assert pred.y_pred.isin(labels).all()
            assert np.isclose(metric['macro_f1'],f1_score(pred.y_true,pred.y_pred,labels=labels,average='macro',zero_division=0))
            assert np.isclose(metric['balanced_accuracy'],balanced_accuracy_score(pred.y_true,pred.y_pred))
            assert detail['confusion_matrix']==confusion_matrix(pred.y_true,pred.y_pred,labels=labels).tolist()
            combined=all_pred[(all_pred.task==task)&(all_pred.subject_id==subject)]
            pd.testing.assert_frame_equal(pred.sort_values('window_id').reset_index(drop=True),
                combined[pred.columns].sort_values('window_id').reset_index(drop=True))
            loss=detail['mlp_optimization']
            assert len(loss['loss_curve'])==loss['n_iter'] and np.isfinite(loss['loss_curve']).all()
            assert np.isclose(loss['loss_curve'][-1],loss['loss'])
            records.append(dict(**metric,n_iter=loss['n_iter'],train_loss=loss['loss'],
                reached_max_iter=loss['reached_max_iter'],warning_count=len(detail['warnings']),
                warnings=' | '.join(detail['warnings'])))
    verified=pd.DataFrame(records)
    for score in summary['results']:
        part=verified[verified.task==score['task']]
        assert len(part)==15 and score['complete_15_subjects']
        for metric in ['macro_f1','balanced_accuracy']:
            assert np.isclose(part[metric].mean(),score[metric+'_mean'])
            assert np.isclose(part[metric].std(ddof=1),score[metric+'_std'])
    audit=pd.read_csv(out/'sampler_audit.csv')
    assert len(audit)==30 and audit.passed.all()
    assert audit.n_test_before.eq(audit.n_test_after).all()
    assert audit.n_train_after.ge(audit.n_train_before).all()
    write_csv(out/'verified_metrics_by_subject.csv',verified)
    result=dict(status='passed',experiment='EXP-015',outer_folds=30,predictions=12378,
        reference='EXP-010',fixed_params_match=True,sampler_audit_rows=30,test_rows_unchanged=True,
        audit_scope='recreated_preprocessing_and_sampler_same_train_and_seed')
    write_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    from wesad.evaluation import run_experiment
    from wesad.utils import read_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-run',type=Path)
    parser.add_argument('--verify-only',type=Path)
    args=parser.parse_args()
    if args.verify_only:verify(args.verify_only.resolve());return
    if args.resume_run:
        out=args.resume_run.resolve();config=read_config(out/'config.yaml')
    else:out,config=prepare()
    print('output:',out,flush=True)
    run_experiment(config,resume=bool(args.resume_run),outer_subjects=config['outer_subjects'])
    sampler_audit(config,out)
    verify(out)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
