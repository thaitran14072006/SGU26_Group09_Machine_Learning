"""Chạy thêm EXP006/010/015, ghi nhật ký ngắn theo seed."""
from pathlib import Path
import argparse,json,sys,zipfile,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
FOLDERS={'006':'01_baselines/EXP-006_MLP','010':'02_tuning/EXP-010','015':'03_balancing/EXP-015'}
def verify(out,config):
    import pandas as pd,numpy as np
    from sklearn.metrics import f1_score,balanced_accuracy_score
    from wesad.utils import read_json,file_hash,write_json
    from wesad.data import load_feature_table,SUBJECTS
    from wesad.evaluation import task_target
    from wesad.g4_nhu import check_splits
    from sklearn.model_selection import ParameterGrid
    frame,features=load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    summary=read_json(out/'summary.json');assert summary['status']=='complete'
    pred=pd.read_csv(out/'predictions.csv');assert len(pred)==12378 and not pred.duplicated(['task','subject_id','window_id']).any()
    rows=[]
    for task in config['tasks']:
        y,labels=task_target(frame,task)
        for subject in SUBJECTS:
            d=out/'folds'/task/'mlp'/subject;c=read_json(d/'complete.json')
            assert all(file_hash(d/n)==h for n,h in c['files'].items())
            detail=read_json(d/'details.json');m=read_json(d/'metrics.json')
            assert set(detail['outer_train_subjects'])==set(SUBJECTS)-{subject} and detail['outer_test_subjects']==[subject]
            p=pred[(pred.task==task)&(pred.subject_id==subject)]
            expected=frame[frame.subject_id==subject][['subject_id','window_id']].copy();expected['expected']=y[frame.subject_id==subject].to_numpy()
            paired=expected.merge(p,on=['subject_id','window_id'],validate='one_to_one',how='outer',indicator=True)
            assert paired['_merge'].eq('both').all() and paired.expected.eq(paired.y_true).all()
            assert np.isclose(m['macro_f1'],f1_score(p.y_true,p.y_pred,labels=labels,average='macro',zero_division=0))
            assert np.isclose(m['balanced_accuracy'],balanced_accuracy_score(p.y_true,p.y_pred))
            if config['nested']:
                check_splits(detail,subject)
                inner=pd.read_csv(d/'inner_cv.csv');candidates=list(ParameterGrid(config['search_spaces']['mlp']))
                assert len(inner)==len(candidates)==4
                assert np.allclose(inner[[f'split{i}_test_score' for i in range(3)]].mean(axis=1),inner.mean_test_score)
                assert detail['best_params']==candidates[int(np.argmax(inner.mean_test_score.to_numpy()))]
            else:assert not detail['inner_splits']
            if config.get('fixed_fold_params'):assert detail['best_params']==config['fixed_fold_params'][f'{task}/mlp/{subject}']
            opt=detail['mlp_optimization'];assert len(opt['loss_curve'])==opt['n_iter'] and np.isfinite(opt['loss_curve']).all()
            t=detail['train_metrics'];assert t['n_train']==6189-len(p) and 0<=t['macro_f1']<=1
            rows.append(dict(**m,train_macro_f1=t['macro_f1'],train_balanced_accuracy=t['balanced_accuracy'],n_iter=opt['n_iter'],warning_count=len(detail['warnings']),max_iter_reached=opt['reached_max_iter']))
    scores=pd.DataFrame(rows)
    for record in summary['results']:
        g=scores[scores.task==record['task']];assert len(g)==15
        for metric in ['macro_f1','balanced_accuracy']:
            assert np.isclose(g[metric].mean(),record[metric+'_mean']) and np.isclose(g[metric].std(ddof=1),record[metric+'_std'])
    scores.to_csv(out/'verified_metrics_by_subject.csv',index=False)
    write_json(out/'verification.json',{'status':'passed','outer_folds':30,'predictions':len(pred),'same_model_train_test':True,'inner_selection_checked':config['nested']})
def run(exp,seed,resume):
    import yaml
    from wesad.utils import read_config,read_json,write_json,file_hash,environment
    from wesad.evaluation import run_experiment
    from wesad.models import build_pipeline
    config=read_config('experiments/02_tuning/EXP-010/results/config.yaml')
    assert environment()==read_json(ROOT/'experiments/02_tuning/EXP-010/results/environment.json')
    assert file_hash(ROOT/config['features'])==config['feature_sha256']
    config.update(seed=seed,record_train_metrics=True,record_mlp_loss=True)
    if exp=='006':
        base=read_config('experiments/01_baselines/EXP-006_MLP/config.yaml')
        config['model_params']=base['model_params'];config['model_params']['mlp']['random_state']=seed
        config.update(nested=False,balance=False);config.pop('search_spaces',None)
    if exp=='015':
        ref=ROOT/'experiments'/FOLDERS['010']/'seed_runs/20261122'/f'seed_{seed}'
        assert read_json(ref/'verification.json')['status']=='passed'
        config.update(nested=False,balance=True,fixed_fold_params=read_json(ref/'best_params.json'),reference_run=ref.relative_to(ROOT).as_posix(),reference_params_sha256=file_hash(ref/'best_params.json'))
        config.pop('search_spaces',None)
    config['protocol']=f'exp{exp}_seed_repeat_20261122'
    config['output_dir']=f'experiments/{FOLDERS[exp]}/seed_runs/20261122/seed_{seed}'
    config['source_paths']=['src/wesad/data.py','src/wesad/evaluation.py','src/wesad/models.py','src/wesad/utils.py','scripts/mlp_other_seeds.py']
    model=build_pipeline('mlp',config);assert model['model'].random_state==seed
    if exp=='015':assert model['sampler'].random_state==seed
    out=ROOT/config['output_dir'];journal=out.parent/f'seed_{seed}_journal.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
    def event(status,note):
        with journal.open('a',encoding='utf8') as f:f.write(json.dumps(dict(date='22/11/2026',member='như',experiment='EXP-'+exp,seed=seed,status=status,note=note),ensure_ascii=False)+'\n')
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    (out/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
    write_json(out/'run_note.json',dict(date='22/11/2026',member='như',experiment='EXP-'+exp,seed=seed,kind='new training',baseline_environment_note='EXP006 old pandas 3.1.1; new seeds use locked EXP010 environment',train_test='same fitted model, train evaluated before ROS duplication',command=sys.argv,environment=environment()))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in config['source_paths']:z.write(ROOT/p,p)
        z.write(out/'config.yaml','config.yaml')
    event('bắt đầu','chạy đủ 15 người, 2 bài toán')
    try:
        run_experiment(config,resume=resume,outer_subjects=config['outer_subjects']);verify(out,config)
        if exp=='015':
            from exp015_train import sampler_audit
            sampler_audit(config,out)
    except Exception as e:
        event('lỗi',str(e));(out/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    event('xong','đủ 30 fold, kiểm tra đạt');print('VERIFIED',exp,seed,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--kind',choices=['baseline','tuned_balanced'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    for exp in (['006'] if a.kind=='baseline' else ['010','015']):run(exp,a.seed,a.resume)
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
