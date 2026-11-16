"""EXP-010: MLP nested LOSO bằng core runner hiện có, không dùng outer test chọn grid."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def prepare():
    import yaml
    from sklearn.model_selection import ParameterGrid
    from wesad.data import SUBJECTS, load_feature_table
    from wesad.utils import read_config, file_hash, write_json
    config=read_config('configs/g4_nhu_tuning.yaml')
    now=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    out=ROOT/'outputs'/now.strftime('exp010_mlp_nested_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    config.update(protocol='exp010_mlp_nested_physiological68_v1',models=['mlp'],
                  outer_subjects=list(SUBJECTS),output_dir=out.relative_to(ROOT).as_posix())
    config['search_spaces']={'mlp':config['search_spaces']['mlp']}
    config['source_paths']=['src/wesad/data.py','src/wesad/evaluation.py',
                           'src/wesad/models.py','src/wesad/utils.py','scripts/exp010_train.py']
    assert config['nested'] and config['inner_splits']==3
    assert not config['balance'] and config['seed']==42
    assert file_hash(ROOT/config['features'])==config['feature_sha256']
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    assert len(features)==68 and len(frame)==6189
    assert len(list(ParameterGrid(config['search_spaces']['mlp'])))==4
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    write_json(out/'notebook_provenance.json',dict(experiment='EXP-010',member='như',
        date=now.strftime('%d/%m/%Y'),run_type='rerun',baseline_experiment='EXP-006',
        note='chạy mới; không nhập điểm lịch sử G4 làm kết quả của lượt này',
        feature_sha256=config['feature_sha256'],outer_folds=30,inner_fits=360,outer_refits=30))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in config['source_paths']:archive.write(ROOT/path,path)
    return out,config


def verify(out):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
    from sklearn.model_selection import ParameterGrid
    from wesad.data import SUBJECTS,load_feature_table
    from wesad.evaluation import task_target
    from wesad.g4_nhu import check_splits
    from wesad.utils import read_config,read_json,file_hash,write_json,write_csv
    config=read_config(out/'config.yaml')
    assert config['models']==['mlp'] and config['tasks']==['multiclass','binary']
    assert config['nested'] and config['inner_splits']==3 and not config['balance']
    assert config['feature_sha256']==file_hash(ROOT/config['features'])
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    assert len(features)==68
    manifest=read_json(out/'config.json')
    summary=read_json(out/'summary.json')
    assert summary['status']=='complete'
    records=[]
    aggregate=pd.read_csv(out/'predictions.csv',dtype={'window_id':str,'subject_id':str})
    assert len(aggregate)==12378 and not aggregate.duplicated(['task','model','subject_id','window_id']).any()
    candidates=list(ParameterGrid(config['search_spaces']['mlp']))
    for task in config['tasks']:
        y,labels=task_target(frame,task)
        for subject in SUBJECTS:
            folder=out/'folds'/task/'mlp'/subject
            completion=read_json(folder/'complete.json')
            assert completion['signature']==manifest['signature']
            assert {'predictions.csv','metrics.json','details.json','inner_cv.csv'}<=set(completion['files'])
            for name,digest in completion['files'].items():
                path=(folder/name).resolve()
                assert path.is_relative_to(folder.resolve()) and file_hash(path)==digest
            details=read_json(folder/'details.json')
            check_splits(details,subject)
            inner=pd.read_csv(folder/'inner_cv.csv')
            assert len(inner)==4 and np.isfinite(inner.mean_test_score).all()
            split_columns=[f'split{i}_test_score' for i in range(3)]
            assert np.allclose(inner[split_columns].mean(axis=1),inner.mean_test_score)
            winner=int(np.argmax(inner.mean_test_score.to_numpy()))
            assert json.dumps(details['best_params'],sort_keys=True)==json.dumps(candidates[winner],sort_keys=True)
            pred=pd.read_csv(folder/'predictions.csv',dtype={'window_id':str,'subject_id':str})
            reference=frame.loc[frame.subject_id==subject,['subject_id','window_id']].copy()
            reference['expected']=y[frame.subject_id==subject].to_numpy()
            joined=reference.merge(pred,on=['subject_id','window_id'],how='outer',validate='one_to_one',indicator=True)
            assert joined['_merge'].eq('both').all() and joined.expected.eq(joined.y_true).all()
            assert pred.y_pred.isin(labels).all()
            combined=aggregate[(aggregate.task==task)&(aggregate.subject_id==subject)]
            pd.testing.assert_frame_equal(pred.sort_values('window_id').reset_index(drop=True),
                                          combined[pred.columns].sort_values('window_id').reset_index(drop=True))
            metric=read_json(folder/'metrics.json')
            assert np.isclose(metric['macro_f1'],f1_score(pred.y_true,pred.y_pred,labels=labels,average='macro',zero_division=0))
            assert np.isclose(metric['balanced_accuracy'],balanced_accuracy_score(pred.y_true,pred.y_pred))
            assert details['confusion_matrix']==confusion_matrix(pred.y_true,pred.y_pred,labels=labels).tolist()
            loss=details['mlp_optimization']
            assert len(loss['loss_curve'])==loss['n_iter'] and np.isfinite(loss['loss_curve']).all()
            assert np.isclose(loss['loss_curve'][-1],loss['loss'])
            records.append(dict(**metric,best_inner_macro_f1=float(inner.mean_test_score.iloc[winner]),
                best_params=json.dumps(details['best_params'],sort_keys=True),
                warning_count=len(details['warnings']),n_iter=loss['n_iter'],train_loss=loss['loss'],
                reached_max_iter=loss['reached_max_iter']))
    verified=pd.DataFrame(records)
    for s in summary['results']:
        part=verified[verified.task==s['task']]
        assert len(part)==15 and s['complete_15_subjects']
        for metric in ['macro_f1','balanced_accuracy']:
            assert np.isclose(part[metric].mean(),s[metric+'_mean'])
            assert np.isclose(part[metric].std(ddof=1),s[metric+'_std'])
    write_csv(out/'verified_metrics_by_subject.csv',verified)
    result=dict(status='passed',experiment='EXP-010',outer_folds=30,predictions=len(aggregate),
        candidates_per_outer=4,inner_folds=3,checks=['checkpoint hashes','subject leakage',
        'inner winner','predictions and labels','metrics','loss traces','sample mean and sd'])
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
    if args.verify_only:
        verify(args.verify_only.resolve());return
    if args.resume_run:
        out=args.resume_run.resolve();config=read_config(out/'config.yaml')
    else:out,config=prepare()
    print('output:',out,flush=True)
    run_experiment(config,resume=bool(args.resume_run),outer_subjects=config['outer_subjects'])
    verify(out)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
