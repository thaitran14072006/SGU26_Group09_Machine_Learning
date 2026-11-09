"""Mai G4: balanced LR, train-only sampler audit, and nested-OOF receipt checks."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, ParameterGrid

from wesad.data import SUBJECTS, load_feature_table
from wesad.evaluation import run_experiment, task_target
from wesad.models import build_pipeline
from wesad.utils import (read_json,write_json,write_csv,resolve_path,file_hash,
                         fingerprint,environment,PROJECT_ROOT)

MODELS={'knn','svm','rf','mlp'}
KEY=['task','model','sensor_set','normalization','subject_id','window_id']
SOURCE_PATHS={'src/wesad/data.py','src/wesad/evaluation.py','src/wesad/models.py',
 'src/wesad/utils.py','src/wesad/g4.py','src/wesad/g3.py','scripts/run_g4_baselines.py','scripts/merge_g4_results.py'}


def require(ok,message):
    if not ok:raise ValueError(message)


def inputs(config):
    require(file_hash(resolve_path(config['features']))==config['feature_sha256'],'Locked feature checksum differs')
    schema=read_json(resolve_path(config['schema']))
    require(schema['feature_version']==config['feature_version']=='physiological_68_v1' and len(schema['feature_columns'])==68,'Use physiological68 schema')
    frame,columns=load_feature_table(resolve_path(config['features']),resolve_path(config['schema']),SUBJECTS)
    require(list(frame.columns)==schema['output_column_order'],'Output column order differs')
    require(set(config['expected_subjects'])==set(SUBJECTS) and len(config['expected_subjects'])==15,'Expected subject list differs')
    require(config['tasks']==['multiclass','binary'] and config['normalization']=='global_train'
            and config['sensor_set']=='combined_physiological68' and config['seed']==42,'G4 main task/sensor/normalization policy differs')
    return frame,columns


def balanced_config(config):
    inputs(config)
    require(config['protocol']=='g4_balanced_lr_physiological68_v1' and config['models']==['lr']
            and config['balance'] is True and config['nested'] is False,'Expected fixed balanced LR, not tuning')
    require(not config.get('model_factories') and not config.get('search_spaces')
            and config.get('variance_threshold') is None,'No factory/search/selection for LR comparison')
    require(config.get('model_params')=={'lr':{'random_state':42}} and config.get('zero_division')==0,'LR parameters/metric policy differ')
    require(config.get('missing_train_class')=='error','Explicit missing-class policy required')
    require(set(config.get('source_paths',[]))==SOURCE_PATHS and len(config['source_paths'])==len(SOURCE_PATHS)
            and config.get('source_hash_normalization')=='LF','Balanced source scope differs')
    require(build_pipeline('lr',config)['model'].class_weight=='balanced','Missing class weights')


def audit_resampling(config,outer_subjects=None):
    """Fit only preprocessing+ROS, never classifiers, in each inner train/refit train."""
    frame,columns=inputs(config);X=frame[columns];groups=frame.subject_id
    selected=set(outer_subjects or SUBJECTS);rows=[]
    for task in config['tasks']:
      y,_=task_target(frame,task)
      for train,test in LeaveOneGroupOut().split(X,y,groups):
        outer=str(groups.iloc[test].iloc[0])
        if outer not in selected:continue
        splitsets=[('refit',train,test)]
        for i,(a,b) in enumerate(GroupKFold(n_splits=3).split(X.iloc[train],y.iloc[train],groups.iloc[train])):
            splitsets.append((f'inner_{i}',train[a],train[b]))
        for kind,fit,validation in splitsets:
          fit_groups=set(groups.iloc[fit]);val_groups=set(groups.iloc[validation])
          require(not fit_groups&val_groups and outer not in fit_groups,'Sampler subject leakage')
          for model in ('knn','mlp'):
            pipeline=build_pipeline(model,{'seed':42,'balance':True})
            prep=clone(pipeline[:-1])
            # Last step is ROS; no fit/predict of KNN/MLP is performed here.
            resampled,labels=prep.fit_resample(X.iloc[fit],y.iloc[fit])
            indices=prep['sampler'].sample_indices_
            require(indices.min()>=0 and indices.max()<len(fit),'Sampler used non-train indices')
            require(np.array_equal(np.asarray(labels),y.iloc[fit].to_numpy()[indices]),'Resampled labels changed')
            before=Counter(y.iloc[fit]);after=Counter(labels)
            require(set(before)==set(after) and set(after.values())=={max(before.values())},'ROS balance differs')
            valid_transformed=prep[:-1].transform(X.iloc[validation])
            test_transformed=prep[:-1].transform(X.iloc[test])
            require(len(valid_transformed)==len(validation) and len(test_transformed)==len(test),'Validation/test resampled')
            rows.append({'task':task,'model':model,'outer_subject':outer,'fit_scope':kind,
              'train_subjects':','.join(sorted(fit_groups)),'validation_subjects':','.join(sorted(val_groups)),
              'train_rows_before':len(fit),'train_rows_after':len(resampled),
              'validation_rows_before':len(validation),'validation_rows_after':len(valid_transformed),
              'test_rows_before':len(test),'test_rows_after':len(test_transformed),
              'class_counts_before':json.dumps({str(int(k)):int(v) for k,v in before.items()},sort_keys=True),
              'class_counts_after':json.dumps({str(int(k)):int(v) for k,v in after.items()},sort_keys=True),'passed':True})
    for model in ('lr','svm','rf'):
        require(build_pipeline(model,{'balance':True,'seed':42})['model'].class_weight=='balanced','class_weight missing')
    return pd.DataFrame(rows)


def _canonical(params):return json.dumps(params,sort_keys=True)


def candidates(settings,model):
    space=settings.get('search_spaces',{}).get(model)
    require(bool(space),'Missing tuning candidates')
    grid=list(ParameterGrid(space))
    require(len(grid)==4 and len({_canonical(p) for p in grid})==4,'Require four distinct locked candidates per model')
    allowed={'knn':{'model__n_neighbors':{3,5,9},'model__metric':{'euclidean','manhattan'},'model__weights':{'uniform','distance'}},
      'svm':{'model__C':{.1,1,10},'model__gamma':{'scale',.01,.1}},
      'rf':{'model__n_estimators':{100,200},'model__max_depth':{10,None},'model__max_features':{'sqrt',.5}},
      'mlp':{'model__hidden_layer_sizes':{(32,),(64,32)},'model__alpha':{1e-4,1e-3},'model__learning_rate_init':{1e-3,1e-2}}}[model]
    for p in grid:
      require(set(p)==set(allowed),'Candidate parameter names differ')
      for k,v in p.items():
        value=tuple(v) if k.endswith('hidden_layer_sizes') else v
        require(value in allowed[k],'Candidate outside proposal domain')
    for k in allowed:require(len({_canonical({k:p[k]}) for p in grid})>1,'Candidates must vary each proposed parameter')
    return grid


def verify_run(directory,config,kind='balanced',allow_partial=False,quick_check=False):
    directory=Path(directory);frame,columns=inputs(config)
    if kind=='balanced':balanced_config(config)
    manifest=read_json(directory/'config.json');settings=manifest['settings']
    require(manifest['data_sha256']==config['feature_sha256'] and manifest['feature_columns']==columns,'Run data/schema differs')
    require(fingerprint({k:manifest[k] for k in ('settings','data_sha256','feature_columns','code_sha256','environment')})==manifest['signature'],'Invalid run signature')
    if kind=='balanced':
        expected={k:v for k,v in config.items() if k not in {'features','schema','output_dir'}}
        require(settings==expected,'Balanced run settings differ')
    else:
        require(settings.get('nested') is True and settings.get('balance') is False and settings.get('inner_splits')==3,'Expected unbalanced main nested tuning, not a fixed/balanced variant')
        require(set(settings['models'])<=MODELS and settings['models'],'Invalid tuned models')
        require(settings['tasks']==config['tasks'] and settings['normalization']==config['normalization']
                and settings['sensor_set']==config['sensor_set'] and settings['seed']==42,'Tuned main protocol differs')
        require(not settings.get('model_factories') and settings.get('variance_threshold') is None,'Unreviewed factory/selection')
        require(settings.get('scorer','macro_f1')=='macro_f1' and settings.get('scoring','macro_f1')=='macro_f1','Inner scorer must be Macro-F1')
        for model in settings['models']:
            overrides=settings.get('model_params',{}).get(model,{})
            allowed={'random_state':42,'n_jobs':1}
            if model=='mlp':allowed.update(early_stopping=False)
            if model=='svm':allowed.update(kernel='rbf',class_weight=None)
            for key,value in overrides.items():
                if model=='mlp' and key=='max_iter':require(isinstance(value,int) and value>0,'Invalid declared MLP iteration limit')
                else:require(key in allowed and value==allowed[key],'Unreviewed main model override')
    if not allow_partial:require(read_json(directory/'summary.json')['status']=='complete','Quick/partial run not accepted')
    elif read_json(directory/'summary.json')['status']!='complete':
        if quick_check:
            require(kind=='balanced' and settings.get('run_kind')=='quick','Expected explicit quick baseline')
        elif settings.get('run_kind')!='main_shard':
            proof=read_json(directory/'verification.json') if (directory/'verification.json').exists() else {}
            assigned=settings.get('outer_subjects',[])
            require(kind=='nested' and proof.get('status')=='passed' and proof.get('scope') in {'assigned_outer4','assigned_outer_subjects'}
                    and assigned and len(set(assigned))==len(assigned) and set(assigned)<=set(SUBJECTS),
                    'Partial input must be explicitly main_shard/assigned subjects, not quick')
    archive=directory/'source_snapshot.zip'
    training_core=None
    if kind=='nested':require(archive.is_file(),'Tuning package must include its source snapshot')
    if archive.exists():
        require(settings.get('source_hash_normalization')=='LF','Declared LF source scope required')
        with zipfile.ZipFile(archive) as z:
            require(set(z.namelist())==set(settings['source_paths']),'Source snapshot inventory differs')
            hashes={n:hashlib.sha256(z.read(n).replace(b'\r\n',b'\n')).hexdigest() for n in z.namelist()}
            require(fingerprint(hashes)==manifest['code_sha256'],'Runtime source snapshot differs')
            core=['src/wesad/data.py','src/wesad/evaluation.py','src/wesad/models.py','src/wesad/utils.py']
            require(set(core)<=set(hashes),'Snapshot missing training core')
            training_core=fingerprint({n:hashes[n] for n in core})
    env=manifest['environment']
    require(env['python'].split()[0]=='3.12.14' and env['packages']==environment()['packages'],'Run environment differs')
    pred=pd.read_csv(directory/'predictions.csv',dtype={'subject_id':str,'window_id':str})
    require(not pred.duplicated(KEY).any(),'Duplicate OOF keys')
    require(set(pred.model)==set(settings['models']) and set(pred.task)==set(config['tasks']),'OOF models/tasks differ')
    require(set(pred.sensor_set)=={config['sensor_set']} and set(pred.normalization)=={config['normalization']},'OOF sensor/normalization differs')
    records=[];params={};warnings=[]
    for task in settings['tasks']:
      y,labels=task_target(frame,task);truth=frame[['subject_id','window_id']].copy();truth['y_true']=y
      for model in settings['models']:
        part=pred.loc[(pred.task==task)&(pred.model==model)];subjects=set(part.subject_id)
        require(subjects<=set(SUBJECTS),'Unknown subject')
        if kind=='nested' and settings.get('outer_subjects'):
            require(subjects==set(settings['outer_subjects']),'Assigned outer subjects differ from predictions')
        if not allow_partial:require(subjects==set(SUBJECTS),'Missing OOF subject')
        reference=truth.loc[truth.subject_id.isin(subjects)]
        joined=reference.merge(part,on=['subject_id','window_id'],how='outer',validate='one_to_one',indicator=True,suffixes=('_expected',''))
        require((joined['_merge']=='both').all() and (joined.y_true_expected==joined.y_true).all(),'Wrong OOF keys/labels')
        require(part.y_pred.isin(labels).all(),'Invalid predicted label')
        grid=candidates(settings,model) if kind=='nested' else None
        for subject in sorted(subjects):
          folder=directory/'folds'/task/model/subject;done=read_json(folder/'complete.json')
          require(done['signature']==manifest['signature'] and {'predictions.csv','metrics.json','details.json'}<=set(done['files']),'Fold identity/inventory differs')
          for n,digest in done['files'].items():
            path=(folder/n).resolve();require(path.is_relative_to(folder.resolve()) and file_hash(path)==digest,'Fold checksum differs')
          detail=read_json(folder/'details.json');metric=read_json(folder/'metrics.json')
          train=set(SUBJECTS)-{subject}
          require(set(detail['outer_train_subjects'])==train and len(detail['outer_train_subjects'])==14 and detail['outer_test_subjects']==[subject],'Outer subject leakage')
          require(set(y[frame.subject_id!=subject])==set(labels),'Outer train missing class')
          if kind=='nested':
            splits=detail['inner_splits'];require(len(splits)==3,'Require three inner folds')
            validations=[]
            for split in splits:
              a,b=set(split['train_subjects']),set(split['validation_subjects'])
              require(a and b and not a&b and a|b==train and subject not in a|b,'Inner/test subject leakage')
              validations.extend(b)
            require(len(validations)==14 and set(validations)==train,'Inner validation subject coverage differs')
            cv=pd.read_csv(folder/'inner_cv.csv')
            require(len(cv)==4,'Missing candidate CV scores')
            cvparams=[ast.literal_eval(s) for s in cv['params']]
            require({_canonical(p) for p in cvparams}=={_canonical(p) for p in grid},'CV candidate grid differs')
            values=cv.mean_test_score.to_numpy(dtype=float);require(np.isfinite(values).all(),'Nonfinite inner CV scores')
            split_scores=cv[[f'split{i}_test_score' for i in range(3)]].to_numpy(dtype=float)
            require(np.isfinite(split_scores).all() and np.allclose(split_scores.mean(axis=1),values,atol=1e-12,rtol=0),'Inner mean scores differ from three splits')
            best=cvparams[int(values.argmax())]
            require(_canonical(detail['best_params'])==_canonical(best),'Best params not selected by inner CV')
            params[f'{task}/{model}/{subject}']=detail['best_params']
          else:require(not detail['best_params'] and not detail['inner_splits'],'Tuning in balanced baseline')
          g=part.loc[part.subject_id==subject]
          pd.testing.assert_frame_equal(g.reset_index(drop=True),pd.read_csv(folder/'predictions.csv',dtype={'subject_id':str,'window_id':str}).reset_index(drop=True))
          scores={'macro_f1':float(f1_score(g.y_true,g.y_pred,labels=labels,average='macro',zero_division=0)),
                  'balanced_accuracy':float(balanced_accuracy_score(g.y_true,g.y_pred))}
          require(metric['subject_id']==subject and metric['task']==task and metric['model']==model and metric['n_test']==len(g),'Fold metric identity differs')
          for k,v in scores.items():require(np.isclose(v,metric[k],atol=1e-12,rtol=0),'Stored metric differs')
          require(detail['confusion_matrix']==confusion_matrix(g.y_true,g.y_pred,labels=labels).tolist(),'Confusion matrix differs')
          records.append({**metric,'sensor_set':config['sensor_set'],'normalization':config['normalization'],'balance':settings['balance'],'nested':settings['nested']})
          if detail['warnings']:warnings.append({'task':task,'model':model,'subject_id':subject,'warnings':detail['warnings']})
    if kind=='nested':
        exported=read_json(directory/'best_params.json')
        require(exported==params,'best_params export differs from fold details')
    return pred,pd.DataFrame(records),params,{'status':'passed','kind':kind,'n_folds':len(records),'n_predictions':len(pred),
        'training_core_sha256':training_core,'producer':manifest,'warnings_by_fold':warnings}


def run_balanced(config,resume=False,quick=None):
    balanced_config(config)
    result=run_experiment(config,resume=resume,outer_subjects=[quick] if quick else None)
    # A quick run verifies its requested fold only; never admitted as main.
    out=resolve_path(config['output_dir'])
    pred,metrics,_,verified=verify_run(out,config,allow_partial=bool(quick),quick_check=bool(quick))
    audit=audit_resampling(config,[quick] if quick else None)
    write_csv(out/'resampling_audit.csv',audit);write_csv(out/'verified_metrics_by_subject.csv',metrics)
    write_json(out/'verification.json',{**verified,'resampling_audits':len(audit),'resampling_passed':bool(audit.passed.all())})
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (out/'environment.txt').write_text(json.dumps(environment(),indent=2),encoding='utf-8')
    write_json(out/'model_parameters.json',build_pipeline('lr',config)['model'].get_params())
    frame,_=inputs(config);weight_rows=[]
    for task in config['tasks']:
      y,labels=task_target(frame,task)
      for subject in ([quick] if quick else SUBJECTS):
        train=y[frame.subject_id!=subject];counts=Counter(train)
        weight_rows.append({'task':task,'subject_id':subject,'n_train':len(train),
          'class_weights':json.dumps({str(label):len(train)/(len(labels)*counts[label]) for label in labels},sort_keys=True)})
    write_csv(out/'class_weights_by_fold.csv',pd.DataFrame(weight_rows))
    if not quick:compare_g3(out,config,metrics)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
      for name in config['source_paths']:z.write(resolve_path(name),name)
    return result


def compare_g3(out,config,balanced_metrics):
    from wesad.g3 import verify_run as verify_g3
    reference=resolve_path(config['reference_g3'])
    c=yaml.safe_load((reference/'config.yaml').read_text(encoding='utf-8'))
    require(c['feature_sha256']==config['feature_sha256'],'G3 reference uses different data')
    for key in ('features','schema'):c[key]=config[key]
    c['output_dir']=str(reference)
    proof,metrics=verify_g3(reference,c)
    plain=metrics.loc[metrics.model=='lr']
    compared=plain.merge(balanced_metrics,on=['task','subject_id'],validate='one_to_one',suffixes=('_g3','_balanced'))
    require(len(compared)==30,'Need paired baseline scores for all 15 subjects and both tasks')
    for score in ('macro_f1','balanced_accuracy'):
        compared[score+'_delta']=compared[score+'_balanced']-compared[score+'_g3']
    write_csv(Path(out)/'comparison_by_subject.csv',compared)
    rows=[]
    for label,table in [('dummy_G3',metrics.loc[metrics.model=='dummy']),('lr_G3',plain),('lr_class_weight_balanced',balanced_metrics)]:
      for task,g in table.groupby('task'):
        r={'variant':label,'task':task,'n_subjects':len(g)}
        for score in ('macro_f1','balanced_accuracy'):
          r[score+'_mean']=float(g[score].mean());r[score+'_std']=float(g[score].std(ddof=1))
        rows.append(r)
    write_json(Path(out)/'comparison_summary.json',{'status':'complete','reference_G3_verification':proof,
      'results':rows,'meaning':'Paired 15-subject comparison; no significance test or tuning performed.'})


def merge_tuned(directories,output,config):
    output=Path(output);require(not output.exists(),'Use a new merged output')
    predictions=[];metrics=[];params={};sources=[];common_code=None;model_designs={}
    for directory in map(Path,directories):
        p,m,best,proof=verify_run(directory,config,kind='nested',allow_partial=True)
        require(proof['producer']['settings'].get('run_kind')!='quick','Cannot merge quick into main')
        code=proof['training_core_sha256']
        if common_code is None:common_code=code
        require(code==common_code,'Different numerical training core; inspect before merging')
        settings=proof['producer']['settings']
        for model in settings['models']:
            design=fingerprint({'candidates':sorted(_canonical(p) for p in candidates(settings,model)),
                                'base_params':settings.get('model_params',{}).get(model,{})})
            require(model not in model_designs or model_designs[model]==design,'Different candidate grids/base parameters between outer shards')
            model_designs[model]=design
        require(not set(params)&set(best),'Duplicate best_params outer fold')
        params.update(best);predictions.append(p);metrics.append(m);sources.append(proof)
    expected={(task,model,s) for task in config['tasks'] for model in MODELS for s in SUBJECTS}
    present=set()
    if predictions:
        joined=pd.concat(predictions,ignore_index=True);joined_metrics=pd.concat(metrics,ignore_index=True)
        require(not joined.duplicated(KEY).any(),'Duplicate tuned predictions')
        present=set(zip(joined_metrics.task,joined_metrics.model,joined_metrics.subject_id))
    missing=sorted(expected-present)
    output.mkdir(parents=True)
    if predictions:
        write_csv(output/'predictions.csv',joined);write_csv(output/'metrics_by_subject.csv',joined_metrics)
    summaries=[]
    if predictions:
      for (task,model),g in joined_metrics.groupby(['task','model']):
        part=joined.loc[(joined.task==task)&(joined.model==model)];labels=[1,2,3] if task=='multiclass' else [0,1]
        row={'task':task,'model':model,'n_subjects':len(g),'complete_15_subjects':len(g)==15,
             'labels':labels,'confusion_matrix':confusion_matrix(part.y_true,part.y_pred,labels=labels).tolist()}
        for score in ('macro_f1','balanced_accuracy'):
            row[score+'_mean']=float(g[score].mean());row[score+'_std']=float(g[score].std(ddof=1)) if len(g)>1 else None
        summaries.append(row)
    write_json(output/'best_params.json',params)
    result={'status':'complete' if not missing else 'waiting_for_tuned_packages','n_completed_outer_folds':len(present),
      'expected_outer_folds':120,'missing_folds':[{'task':t,'model':m,'subject_id':s} for t,m,s in missing],
      'missing_models':sorted({m for _,m,_ in missing}),'source_packages':sources,'results':summaries,
      'note':'Balanced LR is a separate fixed baseline, never substituted for one of the four tuned main models.'}
    write_json(output/'summary.json',result)
    return result
