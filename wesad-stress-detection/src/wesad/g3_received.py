"""Explicit review of the supplied older KNN/SVM runner; never rewrite producer identities."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import yaml
from wesad.data import SUBJECTS,load_feature_table
from wesad.evaluation import task_target
from wesad.g3 import require,verify_run,KEY
from wesad.utils import read_json,write_json,write_csv,file_hash,fingerprint,environment,PROJECT_ROOT

FACTORY_HASH='2794d76159e9cd051d935b45f18c7ff7e3468ed1a1cc0a53d3ed800a354a00f1'


def check_legacy_settings(settings):
    expected={'expected_subjects':list(SUBJECTS),'models':['knn','svm'],'tasks':['multiclass','binary'],
      'seed':42,'sensor_set':'combined_physiological68','normalization':'global_train',
      'balance':False,'nested':False,'variance_threshold':None,
      'model_factories':{'knn':'wesad.models_neighbors_svm:build_pipeline','svm':'wesad.models_neighbors_svm:build_pipeline'},'model_params':{}}
    require(settings==expected,'Unreviewed legacy KNN/SVM settings/factory/parameters')


def review_source(source):
    source=Path(source)
    factory=source/'src/wesad/models_neighbors_svm.py'
    digest=hashlib.sha256(factory.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    require(digest==FACTORY_HASH,'Unreviewed factory source; manual review required')
    parse=lambda p:ast.parse(Path(p).read_text(encoding='utf-8-sig'))
    dump=lambda node:ast.dump(node,include_attributes=False)
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    common=functions(parse(PROJECT_ROOT/'src/wesad/evaluation.py'))
    legacy=functions(parse(source/'src/wesad/evaluation.py'))
    for name in ('task_target','grouped_inner_splits','summarize'):
        require(dump(common[name])==dump(legacy[name]),'Legacy numerical evaluation differs: '+name)
    def training_loop(fn):return next(n for n in fn.body if isinstance(n,ast.Try))
    require(dump(training_loop(common['run_experiment']))==dump(training_loop(legacy['run_experiment'])), 'Legacy fit/predict loop differs')
    def prefix(fn):
        nodes=[]
        for node in fn.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='effective' for t in node.targets):break
            nodes.append(dump(node))
        return nodes
    require(prefix(common['run_experiment'])==prefix(legacy['run_experiment']),'Legacy input/split preflight differs')
    for name in ('data.py','models.py'):
        require(dump(parse(source/'src/wesad'/name))==dump(parse(PROJECT_ROOT/'src/wesad'/name)), 'Legacy base module differs: '+name)
    return {'factory_sha256_LF':digest,'numerical_evaluation_AST_matches':True,
      'review_scope':'Supplied source bundle; runner differences are source-identity metadata. Factory reviewed as median imputer/keep_empty_features, StandardScaler, default KNN/SVC.',
      'runtime_source_snapshot_bound_to_result_hash':False,
      'limit':'Result producer is dirty and has no runtime source snapshot. Review of supplied source does not establish exact identity of all files at training time.'}


def verify_legacy(directory,source,config):
    directory=Path(directory);review=review_source(source)
    manifest=read_json(directory/'config.json');check_legacy_settings(manifest['settings'])
    frame,columns=load_feature_table(config['features'],config['schema'],SUBJECTS)
    require(manifest['data_sha256']==file_hash(Path(config['features']))==config['feature_sha256'], 'Legacy feature checksum differs')
    require(manifest['feature_columns']==columns and len(columns)==68,'Legacy feature schema differs')
    env=manifest['environment']
    require(env['python'].split()[0]=='3.12.14' and env['packages']==environment()['packages'],'Legacy environment differs')
    require(fingerprint({k:manifest[k] for k in ('settings','data_sha256','feature_columns','code_sha256','environment')})==manifest['signature'],'Invalid legacy signature')
    summary,status=[read_json(directory/n) for n in ('summary.json','run_status.json')]
    require(summary['status']==status['status']=='complete' and status['n_completed_folds']==60,'Legacy quick/partial result')
    predictions=pd.read_csv(directory/'predictions.csv',dtype={'subject_id':str,'window_id':str})
    require(not predictions.duplicated(KEY).any(),'Duplicate legacy predictions')
    require(set(predictions.model)=={'knn','svm'} and set(predictions.task)=={'multiclass','binary'},'Legacy models/tasks')
    require(set(predictions.sensor_set)=={'combined_physiological68'} and set(predictions.normalization)=={'global_train'},'Legacy normalization')
    splits=read_json(directory/'splits.json');records=[];warnings=[]
    for task in ('multiclass','binary'):
      y,labels=task_target(frame,task)
      expected=frame[['subject_id','window_id']].copy();expected['y_true']=y
      for model in ('knn','svm'):
        part=predictions.loc[(predictions.task==task)&(predictions.model==model)]
        joined=expected.merge(part,on=['subject_id','window_id'],how='outer',validate='one_to_one',indicator=True,suffixes=('_expected',''))
        require((joined['_merge']=='both').all() and (joined.y_true_expected==joined.y_true).all(),'Legacy OOF keys/truth differ')
        require(part.y_pred.isin(labels).all(),'Legacy predicted labels')
        group=[]
        for subject in SUBJECTS:
          folder=directory/'folds'/task/model/subject;done=read_json(folder/'complete.json')
          require(done['signature']==manifest['signature'] and {'details.json','metrics.json','predictions.csv'}<=set(done['files']),'Legacy fold identity')
          for name,digest in done['files'].items():
            path=(folder/name).resolve()
            require(path.is_relative_to(folder.resolve()) and file_hash(path)==digest,'Legacy fold checksum')
          detail=read_json(folder/'details.json');metric=read_json(folder/'metrics.json')
          require(set(detail['outer_train_subjects'])==set(SUBJECTS)-{subject} and len(detail['outer_train_subjects'])==14 and detail['outer_test_subjects']==[subject],'Legacy subject leakage')
          require(not detail['inner_splits'] and not detail['best_params'],'Tuning in legacy default run')
          key=f'{task}/{model}/{subject}'
          require(splits[key]=={k:detail[k] for k in ('outer_train_subjects','outer_test_subjects','inner_splits')},'Legacy split export differs')
          g=part.loc[part.subject_id==subject]
          pd.testing.assert_frame_equal(g.reset_index(drop=True),pd.read_csv(folder/'predictions.csv',dtype={'subject_id':str,'window_id':str}).reset_index(drop=True))
          scores={'macro_f1':float(f1_score(g.y_true,g.y_pred,average='macro',labels=labels,zero_division=0)),
                  'balanced_accuracy':float(balanced_accuracy_score(g.y_true,g.y_pred))}
          require(metric['subject_id']==subject and metric['model']==model and metric['task']==task and metric['n_test']==len(g),'Legacy metric identity')
          for name,value in scores.items():require(np.isclose(value,metric[name],atol=1e-12,rtol=0),'Legacy metric differs')
          require(detail['confusion_matrix']==confusion_matrix(g.y_true,g.y_pred,labels=labels).tolist(),'Legacy fold confusion matrix differs')
          row={**metric,'sensor_set':'combined_physiological68','normalization':'global_train'}
          records.append(row);group.append(row)
          if detail['warnings']:warnings.append({'task':task,'model':model,'subject_id':subject,'warnings':detail['warnings']})
        stored=[r for r in summary['results'] if r['task']==task and r['model']==model]
        require(len(stored)==1 and stored[0]['complete_15_subjects'] and stored[0]['n_subjects']==15,'Legacy summary completeness')
        for score in ('macro_f1','balanced_accuracy'):
          values=[r[score] for r in group]
          require(np.isclose(stored[0][score+'_mean'],np.mean(values),atol=1e-12,rtol=0) and np.isclose(stored[0][score+'_std'],np.std(values,ddof=1),atol=1e-12,rtol=0),'Legacy summary scores differ')
        require(stored[0]['confusion_matrix']==confusion_matrix(part.y_true,part.y_pred,labels=labels).tolist(),'Legacy summary confusion matrix differs')
    require(len(records)==60 and len(predictions)==len(frame)*4,'Legacy result counts')
    return predictions,pd.DataFrame(records),{'status':'passed_OOF_and_reviewed_declared_protocol','source_review':review,'producer':manifest,'warnings_by_fold':warnings}


def merge_received(standard_runs,legacy_run,legacy_source,output,config):
    output=Path(output);require(not output.exists(),'Use a new output directory')
    predictions=[];metrics=[];sources=[];code=None
    for directory in map(Path,standard_runs):
        require(read_json(directory/'scope.json')['scope']=='main','Quick result')
        require((directory/'source_snapshot.zip').is_file(),'Missing runtime source evidence')
        c=yaml.safe_load((directory/'config.yaml').read_text(encoding='utf-8-sig'))
        for key in ('features','schema','output_dir'):c[key]=config[key]
        expected={**config,'models':c['models']}
        require(c==expected,'Standard protocol differs')
        verification,m=verify_run(directory,c)
        manifest=read_json(directory/'config.json');env=manifest['environment']
        require(env['python'].split()[0]=='3.12.14' and env['packages']==environment()['packages'],'Standard environment differs')
        if code is None:code=verification['code_sha256']
        require(code==verification['code_sha256'],'Standard source differs')
        predictions.append(pd.read_csv(directory/'predictions.csv',dtype={'subject_id':str,'window_id':str}));metrics.append(m)
        sources.append({'status':'passed_standard_protocol','producer':manifest,'verification':verification})
    p,m,review=verify_legacy(legacy_run,legacy_source,config)
    predictions.append(p);metrics.append(m);sources.append(review)
    joined=pd.concat(predictions,ignore_index=True);scores=pd.concat(metrics,ignore_index=True)
    require(not joined.duplicated(KEY).any(),'Duplicate group predictions')
    require(set(joined.model)=={'dummy','lr','knn','svm','rf','mlp'},'Missing/extra group model')
    results=[]
    for (task,model),g in scores.groupby(['task','model']):
        require(len(g)==15 and set(g.subject_id)==set(SUBJECTS),'Incomplete group folds')
        pred=joined.loc[(joined.task==task)&(joined.model==model)];labels=[1,2,3] if task=='multiclass' else [0,1]
        row={'task':task,'model':model,'n_subjects':15,'n_predictions':len(pred),'labels':labels,
             'confusion_matrix':confusion_matrix(pred.y_true,pred.y_pred,labels=labels).tolist()}
        for score in ('macro_f1','balanced_accuracy'):
            row[score+'_mean']=float(g[score].mean());row[score+'_std']=float(g[score].std(ddof=1))
        results.append(row)
    require(len(results)==12 and len(scores)==180 and len(joined)==74268,'Incomplete six-model output')
    output.mkdir(parents=True)
    write_csv(output/'predictions.csv',joined);write_csv(output/'metrics_by_subject.csv',scores)
    report={'status':'complete_OOF_aggregation','scope':'six_default_models_two_tasks_combined_physiological68',
      'n_subjects':15,'n_folds':180,'n_predictions':74268,'missing_models':[],
      'data_sha256':config['feature_sha256'],'results':results,'source_packages':sources,
      'all_producer_code_hashes_identical':False,'strict_same_runtime_source_acceptance':False,
      'limits':['KNN/SVM compatibility reviewed against supplied source; producer dirty and exact runtime source snapshot not available.',
                'Original producer identities retained; no signature/config rewritten, no new fitting or tuning.']}
    write_json(output/'summary.json',report)
    return report
