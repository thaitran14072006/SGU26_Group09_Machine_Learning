"""RF/MLP G4: four fixed candidates, nested CV, separate quick/main outputs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g4_nhu import run
from wesad.utils import read_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/g4_nhu_tuning.yaml')
    p.add_argument('--output')
    p.add_argument('--test-subject',help='Quick check only; requires output with quick in its name')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    config=read_config(args.config)
    if args.output: config['output_dir']=args.output
    print(run(config,resume=args.resume,subject=args.test_subject,verify_only=args.verify_only))

if __name__=='__main__': main()
