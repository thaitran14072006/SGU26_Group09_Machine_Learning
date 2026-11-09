"""Validate/merge nested main G4 packages; never fabricate missing results."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g4 import merge_tuned
from wesad.utils import read_config,resolve_path

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--output',required=True)
 p.add_argument('--config',default='configs/g4_balanced_baseline.yaml')
 a=p.parse_args();r=merge_tuned([resolve_path(x) for x in a.runs],resolve_path(a.output),read_config(a.config))
 print(r['status']+f"; {r['n_completed_outer_folds']}/120 outer folds")
