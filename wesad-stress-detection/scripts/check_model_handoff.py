"""Nạp model trong phiên Python mới và lập danh mục bàn giao."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {'binary': {0, 1}, 'multiclass': {1, 2, 3}}
SUBJECTS = {f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]}
OUT = ROOT / 'artifacts/models/group_submission'
LR = ROOT / 'artifacts/models/lr_candidate_20261122'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def local_path(value):
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Đường dẫn phải nằm trong project.')
    return p


def loso_metrics(path, task, filters=None):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('task') == task]
    for column, value in (filters or {}).items():
        rows = [r for r in rows if r.get(column) == str(value)]
    ids = [r['subject_id'] for r in rows]
    if len(ids) != 15 or set(ids) != SUBJECTS:
        raise ValueError(f'{path.name}: cần đúng một dòng mỗi người, đủ 15 người cho {task}.')
    result = {}
    for column in ['macro_f1', 'balanced_accuracy']:
        values = [float(r[column]) for r in rows]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError(f'{path.name}: {column} không hợp lệ.')
        result[column] = statistics.mean(values)
        result[column + '_sd'] = statistics.stdev(values)
    return result


def bundle_info(algorithm, task):
    folder = LR if algorithm == 'lr' else OUT / algorithm
    model = folder / f'{task}.joblib'
    meta_path = folder / f'metadata_{task}.json'
    if algorithm == 'lr':
        meta_path = folder / 'metadata.json'
    meta = read_json(meta_path)
    schema = folder / meta.get('schema_file', 'feature_schema.json')
    if schema.resolve().parent != folder.resolve():
        raise ValueError('Schema phải nằm cùng thư mục model.')
    if algorithm == 'lr':
        evaluation = {
            'experiment_id': 'EXP-016 + EXP-011 + LR_Regularization',
            'loso_source': 'reports/lr_workflow_20261121/tuning_results.csv',
            'loso_filters': {},
            'evaluation_scope': 'LOSO của quy trình LR chọn cấu hình tuần tự bằng inner CV',
        }
        final_config = meta['selected'][task]
    else:
        required = ['experiment_id', 'loso_source', 'evaluation_scope', 'final_params',
                    'training_rows', 'training_subjects', 'data_sha256', 'environment']
        for field in required:
            if field not in meta:
                raise ValueError(f'{meta_path.name}: thiếu {field}.')
        evaluation = meta
        final_config = meta['final_params']
    return folder, model, meta_path, schema, meta, evaluation, final_config


def check_one(algorithm, task):
    # Tiến trình này chỉ nạp và dự đoán, không gọi fit.
    sys.path.insert(0, str(ROOT / 'src'))
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted

    folder, model_path, meta_path, schema_path, meta, evaluation, final_config = bundle_info(algorithm, task)
    columns = read_json(schema_path)['feature_columns']
    if len(columns) != 68 or len(set(columns)) != 68:
        raise ValueError('Schema cần đủ 68 tên cột riêng biệt.')
    if 'feature_columns' in meta and columns != meta['feature_columns']:
        raise ValueError('Thứ tự feature trong metadata và schema không khớp.')
    if set(meta['training_subjects']) != SUBJECTS or meta['training_rows'] != 6189:
        raise ValueError('Metadata không khớp bộ G2 gồm 15 người và 6.189 cửa sổ.')
    data = ROOT / 'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
    if digest(data) != meta['data_sha256']:
        raise ValueError('Hash dữ liệu không khớp metadata.')
    env = meta['environment']
    if sys.version.split()[0] != env['python'].split()[0]:
        raise ValueError('Phiên bản Python không khớp môi trường model.')
    for name, version in env['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Phiên bản {name} không khớp môi trường model.')
    checksum = digest(model_path)
    expected = meta.get('model_sha256', {})
    if isinstance(expected, str):
        expected = {model_path.name: expected}
    if model_path.name in expected and checksum != expected[model_path.name]:
        raise ValueError('Hash model không khớp metadata.')
    model = joblib.load(model_path)
    check_is_fitted(model)
    if int(model.n_features_in_) != len(columns):
        raise ValueError('Số cột đầu vào model và schema không khớp.')
    if set(map(int, model.classes_)) != TASKS[task]:
        raise ValueError('Tập nhãn của model không khớp bài toán.')
    sample = pd.read_csv(ROOT / 'app/examples/features.csv')
    X = sample[columns].astype(float)
    prediction = np.asarray(model.predict(X))
    if len(prediction) != len(X) or not set(prediction.tolist()).issubset(TASKS[task]):
        raise ValueError('Nhãn dự đoán không hợp lệ.')
    checks = ['fresh_process_load', 'environment', 'data_hash', 'schema_order',
              'input_feature_count', 'classes', 'sample_predict']
    if algorithm == 'lr':
        expected_prediction = pd.read_csv(ROOT / f'app/examples/expected_{task}.csv')
        np.testing.assert_array_equal(prediction, expected_prediction.predicted_label.to_numpy())
        evidence = read_json(folder / 'selection_evidence.json')[task]
        chosen = evidence['stages']['tuning']['chosen']
        if chosen != final_config:
            raise ValueError('Cấu hình cuối không khớp selection_evidence.')
        for name, value in final_config['selection_params'].items():
            if model.get_params()[name] != value:
                raise ValueError(f'{name} trong model khác metadata.')
        if model['model'].C != final_config['C'] or model['model'].weight_policy != final_config['weight_policy']:
            raise ValueError('C hoặc trọng số lớp khác metadata.')
        checks += ['model_hash', 'expected_predictions', 'final_config']
    source = local_path(evaluation['loso_source'])
    filters = evaluation.get('loso_filters', {})
    metrics = loso_metrics(source, task, filters)
    if algorithm == 'lr':
        with (ROOT / 'reports/lr_workflow_20261121/comparison.csv').open(encoding='utf-8-sig', newline='') as f:
            summary = [r for r in csv.DictReader(f) if r['variant'] == 'tuning' and r['task'] == task]
        if len(summary) != 1:
            raise ValueError('Không tìm thấy một dòng tuning của LR.')
        for name in ['macro_f1', 'balanced_accuracy']:
            if not math.isclose(metrics[name], float(summary[0][name + '_mean']), abs_tol=1e-12, rel_tol=0):
                raise ValueError('Điểm theo người không khớp bảng tổng hợp LR.')
    record = {
        'algorithm': algorithm, 'task': task, 'model_path': relative(model_path),
        'experiment_id': evaluation['experiment_id'], **metrics,
        'loso_source': relative(source), 'loso_filters': json.dumps(filters, ensure_ascii=False),
        'evaluation_scope': evaluation['evaluation_scope'],
        'final_config': json.dumps(final_config, ensure_ascii=False, sort_keys=True),
        'training_rows': meta['training_rows'],
        'training_subjects': ','.join(sorted(SUBJECTS, key=lambda x: int(x[1:]))),
        'feature_version': read_json(schema_path).get('version', read_json(schema_path).get('feature_version', '')),
        'metadata_path': relative(meta_path), 'schema_path': relative(schema_path),
        'model_sha256': checksum, 'check_status': 'passed',
    }
    return {'record': record, 'checks': checks, 'sample_rows': len(X),
            'predictions': prediction.tolist(), 'environment': env,
            'data_sha256': meta['data_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-one', nargs=2, metavar=('ALGORITHM', 'TASK'))
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    if args.check_one:
        print(json.dumps(check_one(*args.check_one), ensure_ascii=False, allow_nan=False))
        return
    completed, absent, failed = [], [], []
    for algorithm in ['lr', 'knn', 'svm', 'rf', 'mlp']:
        for task in TASKS:
            folder = LR if algorithm == 'lr' else OUT / algorithm
            path = folder / f'{task}.joblib'
            if not path.is_file():
                absent.append(relative(path))
                continue
            run = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                                  '--check-one', algorithm, task], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf8', timeout=180)
            if run.returncode:
                failed.append((relative(path), run.stderr.strip()))
            else:
                completed.append(json.loads(run.stdout))
                print(f'Đã kiểm tra {algorithm} {task}', flush=True)
    if failed:
        for path, error in failed:
            print(path + '\n' + error, file=sys.stderr)
        raise SystemExit('Chưa cập nhật danh mục vì có model kiểm tra không đạt.')
    if not completed:
        raise SystemExit('Chưa có model nạp được để lập danh mục.')
    OUT.mkdir(parents=True, exist_ok=True)
    summary_dir = OUT / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / 'model_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(completed[0]['record']))
        writer.writeheader()
        writer.writerows(row['record'] for row in completed)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'models': completed, 'summary_sha256': digest(summary_dir / 'model_summary.csv')}
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + '\n', encoding='utf8')
    print(f'Đã ghi {len(completed)} model vào artifacts/models/group_submission/summary/model_summary.csv')
    for path in absent:
        print('Chưa nhận: ' + path)
    if args.require_all and absent:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
