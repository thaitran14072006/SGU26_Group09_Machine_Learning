"""Mai G4 balanced LR and train-only resampling audit."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g4 import run_balanced
from wesad.utils import read_config

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',default='configs/g4_balanced_baseline.yaml')
 p.add_argument('--output');p.add_argument('--quick-subject');p.add_argument('--resume',action='store_true')
 a=p.parse_args();config=read_config(a.config)
 if a.output:config['output_dir']=a.output
 elif a.quick_subject:config['output_dir']+='_quick'
 config['run_kind']='quick' if a.quick_subject else 'main'
 if a.quick_subject and 'quick' not in Path(config['output_dir']).name.lower():p.error('Quick output name must contain quick')
 print(run_balanced(config,a.resume,a.quick_subject)['status'])
