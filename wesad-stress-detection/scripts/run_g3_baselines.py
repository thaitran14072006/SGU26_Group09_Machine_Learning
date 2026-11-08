"""Run G3 default model batches on the locked physiological68 dataset."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g3 import run_defaults
from wesad.models import MODEL_NAMES
from wesad.utils import read_config

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/defaults.yaml')
    p.add_argument('--models',nargs='+',choices=MODEL_NAMES)
    p.add_argument('--output')
    p.add_argument('--quick-subject')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=read_config(a.config)
    if a.models:config['models']=a.models
    if a.output:config['output_dir']=a.output
    elif a.quick_subject:config['output_dir']+='_quick'
    config['run_kind']='quick' if a.quick_subject else 'main'
    result,verified=run_defaults(config,a.resume,a.quick_subject)
    print(f"{verified['scope']}: {result['status']}, {verified['n_folds']} folds, {verified['n_predictions']} predictions")
