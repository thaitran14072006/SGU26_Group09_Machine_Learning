"""Merge standard G3 results with the explicitly reviewed older KNN/SVM package."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wesad.g3_received import merge_received
from wesad.utils import read_config,resolve_path

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--standard-runs',nargs='+',required=True)
 p.add_argument('--knn-svm-run',required=True)
 p.add_argument('--reviewed-source',required=True)
 p.add_argument('--output',required=True)
 p.add_argument('--config',default='configs/defaults.yaml')
 a=p.parse_args();config=read_config(a.config);config['run_kind']='main'
 for key in ('features','schema','output_dir'):config[key]=str(resolve_path(config[key]))
 r=merge_received([resolve_path(x) for x in a.standard_runs],resolve_path(a.knn_svm_run),resolve_path(a.reviewed_source),resolve_path(a.output),config)
 print(r['status']+f"; {r['n_folds']} folds, {r['n_predictions']} predictions")
