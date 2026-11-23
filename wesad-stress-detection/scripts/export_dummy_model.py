"""Xuất một bài toán Dummy khi cần nộp baseline."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from model_handoff_common import ROOT, GROUP, DATA, sha256, diem_theo_nguoi, doc_csv, doc_json


def export(task):
    sys.path.insert(0, str(ROOT / 'src'))
    import joblib
    import numpy as np
    from wesad.data import load_feature_table
    from wesad.evaluation import task_target
    from wesad.models import build_pipeline
    from wesad.utils import read_config

    config_path = ROOT / 'experiments/01_baselines/EXP-001_Dummy/config.yaml'
    cfg = read_config(config_path)
    folder = GROUP / 'dummy'
    model_path = folder / f'{task}.joblib'
    meta_path = folder / f'metadata_{task}.json'
    for path in [model_path, meta_path]:
        if path.exists():
            raise FileExistsError(f'đã có {path.name} trong dummy/, không ghi đè')
    data_path, schema_path = ROOT / cfg['features'], ROOT / cfg['schema']
    if sha256(data_path) != cfg['feature_sha256']:
        raise ValueError('hash dữ liệu khác EXP-001')
    frame, columns = load_feature_table(data_path, schema_path, cfg['expected_subjects'])
    X = frame[columns].astype(float)
    y, labels = task_target(frame, task)
    pipe = build_pipeline('dummy', cfg)
    strategy = pipe.named_steps['model'].strategy
    if strategy != 'most_frequent':
        raise ValueError('strategy khác most_frequent của EXP-001, cần đối chiếu lại')
    source = 'experiments/01_baselines/EXP-001_Dummy/results.csv'
    scores = diem_theo_nguoi([r for r in doc_csv(ROOT / source) if r['task'] == task])
    schema = {'feature_columns': columns, 'feature_version': cfg['feature_version'],
              'source_schema_sha256': sha256(schema_path)}
    output_schema = folder / 'feature_schema.json'
    if output_schema.exists() and doc_json(output_schema) != schema:
        raise ValueError('schema dummy/ khác bài toán đã xuất')
    pipe.fit(X, y)
    folder.mkdir(parents=True, exist_ok=True)
    with model_path.open('xb') as stream:
        joblib.dump(pipe, stream, compress=3)
    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(pipe.predict(X.head(10)), loaded.predict(X.head(10)))
    if set(loaded.classes_) != set(labels):
        raise ValueError('classes_ của Dummy không khớp nhãn')
    if not output_schema.exists():
        with output_schema.open('x', encoding='utf8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
    packages = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'scipy', 'joblib', 'PyYAML']}
    subjects = sorted(frame.subject_id.unique().tolist(), key=lambda s: int(s[1:]))
    meta = {'owner': 'Mai', 'model': 'dummy', 'task': task, 'experiment_id': 'EXP-001',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'strategy': strategy,
            'final_params': pipe.named_steps['model'].get_params(deep=False),
            'reference_config': str(config_path.relative_to(ROOT)).replace('\\', '/'),
            'data_path': cfg['features'], 'data_sha256': sha256(data_path),
            'training_rows': len(frame), 'training_subjects': subjects,
            'fit_scope': 'all_15_subjects; bản refit không có test riêng',
            'feature_columns': columns, 'feature_version': cfg['feature_version'],
            'environment': {'python': platform.python_version(), 'packages': packages},
            'loso_source': source, 'loso_scores': scores,
            'evaluation_scope': 'LOSO của Dummy most_frequent EXP-001',
            'model_sha256': sha256(model_path), 'labels': labels,
            'label_mapping': {'0': 'Non-stress', '1': 'Stress'} if task == 'binary'
                             else {'1': 'Baseline', '2': 'Stress', '3': 'Amusement'},
            'reload_first_10_matches': True}
    with meta_path.open('x', encoding='utf8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'dummy/{task}: đã xuất, nạp lại khớp 10 dòng đầu, classes_ đúng')
    from build_model_summary import main as build_summary
    return build_summary([])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['binary', 'multiclass'], required=True)
    args = parser.parse_args()
    return export(args.task)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
