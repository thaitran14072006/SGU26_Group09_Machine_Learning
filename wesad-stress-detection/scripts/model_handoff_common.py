"""Đọc các kiểu metadata đang dùng trong nhóm."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / 'artifacts/models/group_submission'
LR = ROOT / 'artifacts/models/lr_candidate_20261122'
TASKS = {'binary': {0, 1}, 'multiclass': {1, 2, 3}}
SUBJECTS = {f'S{i}' for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]}
DATA = 'data/features/g2_physiological68_group15_20261107/all_subjects.csv'


def doc_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def doc_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def duong_dan(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('đường dẫn nằm ngoài project')
    return path


def tuong_doi(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def cac_goi():
    yield 'lr', LR
    for name in ['knn', 'svm', 'rf', 'mlp', 'dummy']:
        yield name, GROUP / name


def cac_model():
    for name, folder in cac_goi():
        for task in TASKS:
            if (folder / f'{task}.joblib').is_file():
                yield name, task, folder


def doc_metadata(name, task, folder):
    path = folder / ('metadata.json' if name == 'lr' else f'metadata_{task}.json')
    meta = doc_json(path)
    env = meta.get('environment', {})
    python = env.get('python') or meta.get('python', '')
    packages = env.get('packages') or {k: v for k, v in env.items() if k != 'python'}
    packages = packages or meta.get('package_versions', {})
    checksum = meta.get('model_sha256', meta.get('joblib_sha256', ''))
    if isinstance(checksum, dict):
        checksum = checksum.get(f'{task}.joblib', '')
    config = meta.get('final_params', meta.get('estimator_params', meta.get('model_params')))
    scores = meta.get('reference_loso_scores', meta.get('loso_mean_by_subject', meta.get('loso_scores', {})))
    source = meta.get('reference_loso_source', meta.get('loso_source', ''))
    exp = meta.get('experiment_id', '')
    scope = meta.get('metric_scope', meta.get('evaluation_scope', 'LOSO tham chiếu của EXP'))
    data = meta.get('data_path', meta.get('features', ''))
    fit_scope = meta.get('fit_scope', '')
    if name == 'lr':
        # LR đã xuất theo quy trình tuần tự, không lấy điểm run C cao nhất.
        config = meta['selected'][task]
        source = 'reports/lr_workflow_20261121/tuning_results.csv'
        exp = 'EXP-016 + EXP-011 + LR_Regularization'
        scope = 'LOSO của quy trình LR tuần tự, chọn cấu hình bằng inner CV'
        data = DATA
        fit_scope = 'full_data; training_subjects ghi đủ 15 người'
        rows = [r for r in doc_csv(ROOT / 'reports/lr_workflow_20261121/comparison.csv')
                if r['task'] == task and r['variant'] == 'tuning']
        if len(rows) != 1:
            raise ValueError('thiếu dòng tuning trong bảng LR')
        scores = rows[0]
    schema_name = meta.get('schema_file', meta.get('feature_schema', 'feature_schema.json'))
    schema = (folder / schema_name).resolve()
    if not schema.is_relative_to(folder.resolve()):
        raise ValueError('schema không nằm trong thư mục model')
    return dict(raw=meta, path=path, experiment_id=exp, config=config, source=source,
                scores=scores, scope=scope, variant=meta.get('variant', ''),
                filters=meta.get('loso_filters', {}), schema=schema, data=data,
                data_sha256=meta.get('data_sha256', meta.get('feature_sha256', '')),
                rows=meta.get('training_rows', meta.get('n_windows_fit')),
                subjects=meta.get('training_subjects', meta.get('subjects_fit', [])),
                fit_scope=fit_scope, python=python, packages=packages, checksum=checksum)


def diem_metadata(scores, name):
    keys = [name, name + '_mean']
    if name.endswith('_sd'):
        keys = [name, name[:-3] + '_std']
    for key in keys:
        if key in scores:
            value = float(scores[key])
            if not math.isfinite(value):
                raise ValueError('điểm metadata không hữu hạn')
            return value
    raise ValueError('thiếu ' + name + ' trong metadata')


def loc_dong(rows, task, info):
    rows = [r for r in rows if r.get('task') == task]
    filters = dict(info.get('filters', {}))
    if info.get('variant') and rows and 'variant' in rows[0]:
        filters.setdefault('variant', info['variant'])
    for key, value in filters.items():
        rows = [r for r in rows if r.get(key) == str(value)]
    return rows


def diem_theo_nguoi(rows):
    ids = [r['subject_id'] for r in rows]
    if len(ids) != 15 or set(ids) != SUBJECTS:
        raise ValueError('cần đúng một dòng mỗi người, đủ 15 người')
    result = {}
    for name in ['macro_f1', 'balanced_accuracy']:
        values = [float(r[name]) for r in rows]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('điểm LOSO không hợp lệ: ' + name)
        result[name] = statistics.mean(values)
        result[name + '_sd'] = statistics.stdev(values)
    return result


def diem_goc(info, task):
    if not info['source']:
        raise ValueError('TODO: thiếu nguồn LOSO')
    source = duong_dan(info['source'])
    rows = doc_csv(source)
    if not rows:
        raise ValueError('nguồn LOSO rỗng')
    summary = None
    raw = source
    if 'subject_id' not in rows[0]:
        selected = loc_dong(rows, task, info)
        if len(selected) != 1:
            raise ValueError('nguồn tổng hợp có nhiều phiên bản, cần ghi rõ variant')
        summary = selected[0]
        raw = source.with_name('results.csv')
        rows = doc_csv(raw)
    result = diem_theo_nguoi(loc_dong(rows, task, info))
    issues = []
    for origin, declared in [('metadata', info['scores']), ('summary gốc', summary)]:
        if declared is None:
            continue
        for name in result:
            try:
                value = diem_metadata(declared, name)
                if not math.isclose(value, result[name], abs_tol=1e-10, rel_tol=0):
                    issues.append(f'{origin} {name}={value:.12g}, theo người={result[name]:.12g}')
            except (ValueError, TypeError) as exc:
                issues.append(str(exc))
    return result, tuong_doi(raw), issues
