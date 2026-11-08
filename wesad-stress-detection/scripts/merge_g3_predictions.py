"""Verify and merge G3 model batches; missing models remain explicitly pending."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g3 import merge_runs
from wesad.utils import read_config,resolve_path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/defaults.yaml')
    a=p.parse_args();config=read_config(a.config);config['run_kind']='main'
    r=merge_runs([resolve_path(d) for d in a.runs],resolve_path(a.output),config)
    print(r['status']+'; missing models: '+', '.join(r['missing_models']))
