"""Kiểm tra file, metadata và khả năng nạp model trong phiên Python mới."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from model_handoff_common import (ROOT, GROUP, LR, TASKS, SUBJECTS, cac_goi,
                                  doc_json, doc_metadata, sha256, duong_dan, tuong_doi)


def bang_nhau(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def check_one_model(name, task):
    sys.path.insert(0, str(ROOT / 'src'))
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted

    folder = LR if name == 'lr' else GROUP / name
    messages = []
    def log(level, message):
        messages.append({'level': level, 'message': message})
    try:
        info = doc_metadata(name, task, folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'algorithm': name, 'task': task, 'messages': [{'level': 'error', 'message': str(exc)}]}
    required = {'mã EXP': info['experiment_id'], 'cấu hình cuối': info['config'],
                'nguồn dữ liệu': info['data'], 'số cửa sổ fit': info['rows'],
                'subject fit': info['subjects'], 'Python': info['python'],
                'phiên bản thư viện': info['packages'], 'phạm vi fit': info['fit_scope']}
    for label, value in required.items():
        if value is None or value == '' or value == [] or value == {}:
            log('error', 'metadata thiếu ' + label)
    raw = info['raw']
    if raw.get('task', task) != task or raw.get('model', name) != name:
        log('error', 'model hoặc task trong metadata không khớp tên file')
    subjects = info['subjects']
    if not subjects or len(subjects) != len(set(subjects)) or set(subjects) - SUBJECTS:
        log('error', 'danh sách subject fit không hợp lệ')
    elif set(subjects) == SUBJECTS:
        log('ok', 'metadata ghi fit đủ 15 người; đây là bản refit toàn bộ dữ liệu')
        if info['rows'] != 6189:
            log('error', 'số cửa sổ fit khác 6.189 của bộ G2')
    if info['python'] and sys.version.split()[0] != info['python'].split()[0]:
        log('warning', f"Python đang chạy {sys.version.split()[0]}, lúc xuất {info['python'].split()[0]}")
    for package, expected in info['packages'].items():
        if expected is None:
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            log('warning', 'chưa cài thư viện đã ghi trong metadata: ' + package)
            continue
        if actual != expected:
            log('warning', f'{package} đang dùng {actual}, lúc xuất {expected}')
    if info['data'] and info['data_sha256']:
        try:
            if sha256(duong_dan(info['data'])) != info['data_sha256']:
                log('error', 'hash dữ liệu khác metadata')
            else:
                log('ok', 'hash dữ liệu khớp metadata')
        except OSError:
            log('warning', 'không có bảng dữ liệu để kiểm tra hash')
    else:
        log('error', 'thiếu nguồn hoặc hash dữ liệu')
    path = folder / f'{task}.joblib'
    if not info['checksum']:
        log('error', 'metadata thiếu hash model')
    elif sha256(path) != info['checksum']:
        log('error', 'hash model khác metadata, dừng nạp file này')
        return {'algorithm': name, 'task': task, 'messages': messages}
    else:
        log('ok', 'hash model khớp metadata')
    try:
        columns = doc_json(info['schema'])['feature_columns']
        if not columns or len(columns) != len(set(columns)):
            raise ValueError('schema rỗng hoặc trùng tên cột')
        if raw.get('feature_columns') is not None and raw['feature_columns'] != columns:
            raise ValueError('thứ tự cột metadata khác schema')
        sample = pd.read_csv(ROOT / 'app/examples/features.csv')
        X = sample[columns].astype(float)
        if any(c in columns for c in ['subject_id', 'window_id', 'label_original', 'label_binary']):
            raise ValueError('schema có cột metadata hoặc nhãn')
        model = joblib.load(path)
        check_is_fitted(model)
        if model.n_features_in_ != len(columns):
            raise ValueError('số cột model khác schema')
        if hasattr(model, 'feature_names_in_') and list(model.feature_names_in_) != columns:
            raise ValueError('thứ tự cột model khác schema')
        if set(map(int, model.classes_)) != TASKS[task]:
            raise ValueError('classes_ không khớp nhãn bài toán')
        y = np.asarray(model.predict(X))
        if y.ndim != 1 or len(y) != len(X) or not set(y.tolist()).issubset(TASKS[task]):
            raise ValueError('predict ra nhãn hoặc số dòng không hợp lệ')
        log('ok', f'phiên Python mới nạp được model, predict đúng tập nhãn trên {len(X)} dòng')
        params = model.named_steps['model'].get_params()
        for key, value in (info['config'] or {}).items():
            if name == 'lr':
                continue
            # Các metadata có thể ghi params của estimator hoặc cả pipeline.
            actual = model.get_params().get(key) if '__' in key else params.get(key)
            if not bang_nhau(actual, value):
                log('error', f'tham số {key} trong model khác metadata')
        if name == 'lr':
            for key, value in info['config']['selection_params'].items():
                if model.get_params()[key] != value:
                    log('error', f'{key} trong LR khác metadata')
            for key in ['C', 'weight_policy']:
                if params[key] != info['config'][key]:
                    log('error', f'{key} trong LR khác metadata')
            expected = pd.read_csv(ROOT / f'app/examples/expected_{task}.csv')['predicted_label']
            np.testing.assert_array_equal(y, expected.to_numpy())
            log('ok', 'dự đoán LR khớp file mẫu đã lưu')
        expected_path = folder / f'expected_{task}.csv'
        example_path = folder / 'example_features.csv'
        if expected_path.exists() and example_path.exists():
            example = pd.read_csv(example_path)[columns].astype(float)
            expected = pd.read_csv(expected_path).iloc[:, 0].to_numpy()
            np.testing.assert_array_equal(model.predict(example), expected)
            log('ok', 'dự đoán khớp mẫu riêng của người xuất model')
    except Exception as exc:
        log('error', f'nạp hoặc predict chưa đạt: {exc}')
    return {'algorithm': name, 'task': task, 'model_path': tuong_doi(path), 'messages': messages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('MODEL', 'TASK'))
    parser.add_argument('--include-dummy', action='store_true', help='yêu cầu đủ cả hai file Dummy')
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(check_one_model(*args.worker), ensure_ascii=False))
        return 0
    results, errors = [], []
    for name, folder in cac_goi():
        if name == 'dummy' and not args.include_dummy and not folder.exists():
            print('dummy/: chưa có, bỏ qua')
            continue
        filenames = ['binary.joblib', 'multiclass.joblib', 'feature_schema.json']
        filenames += ['metadata.json'] if name == 'lr' else ['metadata_binary.json', 'metadata_multiclass.json']
        missing = [f for f in filenames if not (folder / f).is_file()]
        for filename in missing:
            message = f'thiếu {filename} trong {name}/'
            print(message, flush=True)
            errors.append(message)
        if not missing:
            print(f'{name}/: đủ file', flush=True)
        for task in TASKS:
            metadata = 'metadata.json' if name == 'lr' else f'metadata_{task}.json'
            if any(not (folder / f).is_file() for f in [f'{task}.joblib', metadata, 'feature_schema.json']):
                continue
            try:
                run = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                                      '--worker', name, task], cwd=ROOT, capture_output=True,
                                     text=True, encoding='utf8', timeout=180)
            except (OSError, subprocess.TimeoutExpired) as exc:
                message = f'{name}/{task}: không chạy được phiên kiểm tra: {exc}'
                errors.append(message)
                print(message, flush=True)
                continue
            if run.returncode:
                message = f'{name}/{task}: {run.stderr.strip()}'
                errors.append(message)
                print(message, flush=True)
                continue
            result = json.loads(run.stdout)
            results.append(result)
            for item in result['messages']:
                label = {'ok': 'đạt', 'warning': 'khác môi trường', 'error': 'lỗi'}[item['level']]
                print(f"{name}/{task}: {label} - {item['message']}", flush=True)
                if item['level'] == 'error':
                    errors.append(f"{name}/{task}: {item['message']}")
    GROUP.mkdir(parents=True, exist_ok=True)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version.split()[0],
              'models': results, 'errors': errors, 'status': 'passed' if not errors else 'failed'}
    (GROUP / 'submission_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'Đã kiểm tra {len(results)} model, {len(errors)} lỗi.')
    return int(bool(errors))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
