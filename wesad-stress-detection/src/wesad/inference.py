"""CSV inference for the trusted, locally exported LR candidate bundle."""
import argparse
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from wesad.utils import PROJECT_ROOT, file_hash

DEFAULT_BUNDLE = PROJECT_ROOT / 'artifacts/models/lr_candidate_20261122'


def read_input(text, columns):
    header = next(csv.reader(io.StringIO(text)), [])
    if not header or len(header) != len(set(header)):
        raise ValueError('CSV header is empty or contains duplicate columns.')
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise ValueError('CSV contains no data rows.')
    missing = sorted(set(columns)-set(frame.columns))
    if missing:
        raise ValueError('Missing required feature columns: '+', '.join(missing))
    X = frame[columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values are not supported.')
    if X.isna().all(axis=1).any():
        raise ValueError('A row cannot have all 68 feature values missing.')
    return frame, X


def predict_text(text, task='multiclass', bundle=DEFAULT_BUNDLE):
    if task not in {'binary','multiclass'}:
        raise ValueError('Task must be binary or multiclass.')
    bundle = Path(bundle)
    meta = json.loads((bundle/'metadata.json').read_text(encoding='utf8'))
    for name, digest in meta['model_sha256'].items():
        if file_hash(bundle/name) != digest:
            raise ValueError('Model checksum mismatch: '+name)
    frame, X = read_input(text, meta['feature_columns'])
    model = joblib.load(bundle/f'{task}.joblib')
    prediction = model.predict(X)
    result = frame[[c for c in ['subject_id','window_id'] if c in frame]].copy()
    result['predicted_label'] = prediction
    result['predicted_name'] = [meta['label_mapping'][task][str(int(y))] for y in prediction]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--task', choices=['binary','multiclass'], default='multiclass')
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = predict_text(Path(args.csv).read_text(encoding='utf-8-sig'),args.task,args.bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
