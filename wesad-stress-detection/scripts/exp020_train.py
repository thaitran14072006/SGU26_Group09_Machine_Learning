"""EXP-020: grid kiến trúc/alpha mới, nested LOSO bằng core runner."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import sys
import subprocess
import zipfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def prepare():
    import yaml
    from sklearn.model_selection import ParameterGrid
    from wesad.data import SUBJECTS, load_feature_table
    from wesad.utils import read_config, file_hash, write_json, environment
    plan = read_config('configs/exp020_mlp_architecture.yaml')
    config = read_config(plan['reference_config'])
    reference_grid = list(ParameterGrid(config['search_spaces']['mlp']))
    config.update({k:v for k,v in plan.items() if k != 'reference_config'})
    config['model_params'] = plan['model_params']
    config['outer_subjects'] = list(SUBJECTS)
    grid = list(ParameterGrid(config['search_spaces']['mlp']))
    assert len(grid) == 4
    assert all(candidate not in reference_grid for candidate in grid)
    assert config['nested'] and config['inner_splits'] == 3 and not config['balance']
    assert config['seed'] == 42 and config['tasks'] == ['multiclass','binary']
    assert file_hash(ROOT / config['features']) == config['feature_sha256']
    frame, features = load_feature_table(ROOT/config['features'], ROOT/config['schema'], SUBJECTS)
    assert len(frame) == 6189 and len(features) == 68
    assert environment() == json.loads((ROOT/'experiments/02_tuning/EXP-010/results/environment.json').read_text())
    now = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    out = ROOT / 'outputs' / (now.strftime('exp020_mlp_nested_%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    config['output_dir'] = out.relative_to(ROOT).as_posix()
    config['source_paths'] = ['src/wesad/data.py','src/wesad/evaluation.py',
        'src/wesad/models.py','src/wesad/utils.py','scripts/exp020_train.py']
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    git = ['git','-c',f'safe.directory={ROOT.parent.as_posix()}']
    head = subprocess.run(git+['rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dirty = subprocess.run(git+['status','--porcelain'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    write_json(out/'notebook_provenance.json',dict(experiment='EXP-020',member='như',
        date=now.strftime('%d/%m/%Y'),run_type='new',git_commit=head,git_dirty=bool(dirty),
        source_status='uncommitted changes captured in source_snapshot.zip',
        python_executable=sys.executable,command=subprocess.list2cmdline([sys.executable,*sys.argv]),
        outer_folds=30,inner_fits=360,outer_refits=30,seed=42,
        reference_experiment='EXP-010',feature_sha256=config['feature_sha256'],
        interpretation='exploratory comparison on reused LOSO benchmark; no independent final holdout'))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in config['source_paths']+['configs/exp020_mlp_architecture.yaml']:
            z.write(ROOT/p,p)
    return out, config


def verify(out):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
    from wesad.data import SUBJECTS, load_feature_table
    from wesad.evaluation import task_target
    from wesad.g4_nhu import check_splits
    from wesad.utils import read_config, read_json, file_hash, write_json, write_csv
    config = read_config(out/'config.yaml')
    assert config['protocol'] == 'exp020_mlp_architecture_regularization_v1'
    assert config['nested'] and config['inner_splits'] == 3 and not config['balance']
    assert config['models'] == ['mlp'] and config['tasks'] == ['multiclass','binary']
    assert config['feature_sha256'] == file_hash(ROOT/config['features'])
    frame, features = load_feature_table(ROOT/config['features'],ROOT/config['schema'],SUBJECTS)
    manifest, summary = read_json(out/'config.json'), read_json(out/'summary.json')
    assert summary['status'] == 'complete' and len(features) == 68
    candidates = list(ParameterGrid(config['search_spaces']['mlp']))
    reference = list(ParameterGrid(read_config('experiments/02_tuning/EXP-010/results/config.yaml')['search_spaces']['mlp']))
    assert len(candidates) == 4 and all(p not in reference for p in candidates)
    aggregate = pd.read_csv(out/'predictions.csv',dtype={'subject_id':str,'window_id':str})
    assert len(aggregate) == 12378 and not aggregate.duplicated(['task','subject_id','window_id']).any()
    records = []
    for task in config['tasks']:
        y, labels = task_target(frame, task)
        for subject in SUBJECTS:
            folder = out/'folds'/task/'mlp'/subject
            complete = read_json(folder/'complete.json')
            assert complete['signature'] == manifest['signature']
            for name,digest in complete['files'].items():
                path = (folder/name).resolve()
                assert path.is_relative_to(folder.resolve()) and file_hash(path) == digest
            detail = read_json(folder/'details.json')
            check_splits(detail, subject)
            inner = pd.read_csv(folder/'inner_cv.csv')
            assert len(inner) == 4 and np.isfinite(inner.mean_test_score).all()
            assert np.allclose(inner[[f'split{i}_test_score' for i in range(3)]].mean(axis=1),inner.mean_test_score)
            winner = int(np.argmax(inner.mean_test_score.to_numpy()))
            assert detail['best_params'] == candidates[winner]
            for i,candidate in enumerate(candidates):
                for key,value in candidate.items():
                    import ast
                    actual = inner.iloc[i]['param_'+key]
                    actual = ast.literal_eval(actual) if isinstance(actual,str) and key.endswith('hidden_layer_sizes') else actual
                    assert actual == value, (key,actual,value)
            pred = pd.read_csv(folder/'predictions.csv',dtype={'subject_id':str,'window_id':str})
            expected = frame.loc[frame.subject_id==subject,['subject_id','window_id']].copy()
            expected['expected'] = y[frame.subject_id==subject].to_numpy()
            pair = expected.merge(pred,on=['subject_id','window_id'],how='outer',validate='one_to_one',indicator=True)
            assert pair['_merge'].eq('both').all() and pair.expected.eq(pair.y_true).all()
            assert pred.y_pred.isin(labels).all()
            saved = aggregate[(aggregate.task==task)&(aggregate.subject_id==subject)]
            pd.testing.assert_frame_equal(pred.sort_values('window_id').reset_index(drop=True),saved[pred.columns].sort_values('window_id').reset_index(drop=True))
            m = read_json(folder/'metrics.json')
            assert np.isclose(m['macro_f1'],f1_score(pred.y_true,pred.y_pred,labels=labels,average='macro',zero_division=0))
            assert np.isclose(m['balanced_accuracy'],balanced_accuracy_score(pred.y_true,pred.y_pred))
            assert detail['confusion_matrix'] == confusion_matrix(pred.y_true,pred.y_pred,labels=labels).tolist()
            loss = detail['mlp_optimization']
            assert len(loss['loss_curve']) == loss['n_iter'] and np.isfinite(loss['loss_curve']).all()
            assert np.isclose(loss['loss_curve'][-1],loss['loss'])
            records.append(dict(**m,n_iter=loss['n_iter'],train_loss=loss['loss'],
                reached_max_iter=loss['reached_max_iter'],warning_count=len(detail['warnings']),
                best_inner_macro_f1=float(inner.mean_test_score.iloc[winner])))
    verified = pd.DataFrame(records)
    for s in summary['results']:
        part = verified[verified.task==s['task']]
        assert len(part) == 15 and s['complete_15_subjects']
        for key in ['macro_f1','balanced_accuracy']:
            assert np.isclose(part[key].mean(),s[key+'_mean'])
            assert np.isclose(part[key].std(ddof=1),s[key+'_std'])
    write_csv(out/'verified_metrics_by_subject.csv',verified)
    result = dict(status='passed',experiment='EXP-020',outer_folds=30,predictions=len(aggregate),
        candidates_per_outer=4,inner_folds=3,checks=['data hash','checkpoint hashes','subject leakage',
        'new grid','candidate identity and inner winner','predictions and metrics','loss','mean and sample sd'])
    write_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    from wesad.evaluation import run_experiment
    from wesad.utils import read_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-run',type=Path)
    p.add_argument('--verify-only',type=Path)
    args = p.parse_args()
    if args.verify_only:
        verify(args.verify_only.resolve()); return
    if args.resume_run:
        out=args.resume_run.resolve(); config=read_config(out/'config.yaml')
    else:
        out, config = prepare()
    print('output:',out,flush=True)
    run_experiment(config,resume=bool(args.resume_run),outer_subjects=config['outer_subjects'])
    verify(out)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
