"""Refit the locked LR selection procedure using all WESAD training subjects."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from threadpoolctl import threadpool_limits
from wesad.data import SUBJECTS, load_feature_table
from wesad.lr_workflow import build, freeze, plan
from wesad.inference import DEFAULT_BUNDLE, predict_text
from wesad.utils import environment, file_hash, read_config, write_json


def main():
    cfg = plan()
    for stage in cfg['order']: freeze(stage,cfg)
    default = read_config(ROOT/'configs/defaults.yaml')
    assert file_hash(ROOT/default['features']) == cfg['feature_sha256']
    frame,columns = load_feature_table(ROOT/default['features'],ROOT/default['schema'],SUBJECTS)
    X,groups = frame[columns].astype(float),frame.subject_id
    out = DEFAULT_BUNDLE
    if (out/'metadata.json').exists():
        raise FileExistsError('Candidate bundle already exists; do not overwrite an exported model.')
    out.mkdir(parents=True,exist_ok=True)
    selections,evidence = {},{}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always',ConvergenceWarning)
        for task in ['multiclass','binary']:
            y = frame.label_original if task == 'multiclass' else (frame.label_original==2).astype(int)
            labels = [1,2,3] if task=='multiclass' else [0,1]
            splits = list(GroupKFold(3).split(X,y,groups))
            recipe = None
            stages = {}
            for stage in cfg['order']:
                candidates = []
                for spec in [s for s in cfg['experiments'] if s['stage']==stage]:
                    pipe,grid,method = build(spec,cfg,recipe)
                    search = GridSearchCV(pipe,grid,scoring=make_scorer(f1_score,labels=labels,average='macro',zero_division=0),
                        cv=splits,n_jobs=1,error_score='raise',refit=True).fit(X,y)
                    params = {k:v for k,v in search.best_estimator_.get_params().items() if k in ['selection__k','selection__threshold']}
                    candidate = {'run_id':spec['run_id'],'method':method,'selection_params':params,
                        'weight_policy':search.best_estimator_['model'].weight_policy,
                        'C':search.best_estimator_['model'].C,'inner_macro_f1':float(search.best_score_)}
                    candidates.append((candidate,search.best_estimator_))
                selected,model = max(candidates,key=lambda c:c[0]['inner_macro_f1'])
                recipe = dict(selected)
                stages[stage] = {'candidates':[c[0] for c in candidates],'chosen':selected}
                print(task,stage,selected,flush=True)
            joblib.dump(model,out/f'{task}.joblib')
            reloaded = joblib.load(out/f'{task}.joblib')
            np.testing.assert_array_equal(model.predict(X),reloaded.predict(X))
            np.testing.assert_allclose(model.predict_proba(X),reloaded.predict_proba(X),rtol=0,atol=0)
            selections[task] = recipe
            evidence[task] = {'stages':stages,'selected_features':np.array(columns)[model['selection'].get_support()].tolist(),
                'inner_splits':[{'train':sorted(set(groups.iloc[a])),'validation':sorted(set(groups.iloc[b]))} for a,b in splits],
                'save_load_predictions_equal':True,'save_load_probabilities_equal':True}
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise ValueError('Deployment refit did not converge.')
    write_json(out/'selection_evidence.json',evidence)
    write_json(out/'feature_schema.json',{'feature_columns':columns,'count':68,'version':'physiological_68_v1'})
    write_json(out/'metadata.json',{'status':'LR candidate, not final group winner','created_at':datetime.now(timezone.utc).isoformat(),
        'training_rows':len(frame),'training_subjects':list(SUBJECTS),'data_sha256':cfg['feature_sha256'],
        'feature_columns':columns,'environment':environment(),'selected':selections,
        'model_sha256':{f'{t}.joblib':file_hash(out/f'{t}.joblib') for t in selections},
        'label_mapping':{'binary':{'0':'Non-stress','1':'Stress'},'multiclass':{'1':'Baseline','2':'Stress','3':'Amusement'}},
        'evaluation':'LOSO results in reports/lr_workflow_20261121; full-data inner scores are selection criteria, not test scores',
        'limits':'Exploratory repeated WESAD benchmark; no external holdout; demo examples are training windows, not independent tests.'})
    examples = ROOT/'app/examples'
    examples.mkdir(parents=True,exist_ok=True)
    sample = frame.groupby('subject_id',sort=False).head(2)[['subject_id','window_id',*columns]]
    sample.to_csv(examples/'features.csv',index=False)
    for task in selections:
        predict_text(sample.to_csv(index=False),task).to_csv(examples/f'expected_{task}.csv',index=False)
    print('EXPORTED',out,flush=True)


if __name__=='__main__': main()
