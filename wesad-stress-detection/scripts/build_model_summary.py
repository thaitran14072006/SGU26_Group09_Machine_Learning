"""Lập danh mục model từ metadata và điểm LOSO gốc."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

from model_handoff_common import (ROOT, GROUP, cac_model, doc_metadata, diem_goc,
                                  doc_csv, tuong_doi, diem_theo_nguoi)

FIELDS = ['algorithm', 'task', 'model_path', 'experiment_id', 'macro_f1',
          'balanced_accuracy', 'loso_source', 'macro_f1_sd', 'balanced_accuracy_sd',
          'metadata_path', 'metadata_loso_source', 'variant', 'evaluation_scope',
          'fit_scope', 'score_check']
SUMMARY = GROUP / 'summary'


def write_overview(rows, issues):
    lines = ['TỔNG HỢP BÀN GIAO MODEL WESAD', '',
             f'Danh mục có {len(rows)} file model. Số mục cần đối chiếu lại: {len(issues)}.', '',
             'Thuật toán / bài toán / Macro-F1 / Balanced Accuracy / mã EXP']
    for row in rows:
        f1 = f"{row['macro_f1']:.4f}" if isinstance(row['macro_f1'], (int, float)) else 'TODO'
        ba = f"{row['balanced_accuracy']:.4f}" if isinstance(row['balanced_accuracy'], (int, float)) else 'TODO'
        lines.append(f"{row['algorithm'].upper()} / {row['task']} / {f1} / {ba} / {row['experiment_id']}")
    report_path = GROUP / 'submission_check.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf8'))
        checked = datetime.fromisoformat(report['checked_at']).astimezone(timezone(timedelta(hours=7)))
        lines += ['', f"Lần kiểm tra nạp model: {checked:%d/%m/%Y %H:%M}.",
                  f"Đã kiểm tra {len(report['models'])} model, {len(report['errors'])} lỗi trong lần này."]
        warnings = [m for m in report['models'] if any(x['level'] == 'warning' for x in m['messages'])]
        if warnings:
            lines.append('KNN/SVM có khác biệt môi trường xuất và môi trường kiểm tra. Chi tiết nằm trong submission_check.json.')
    lines += ['', 'Dummy dùng most_frequent của EXP-001, là baseline so sánh.',
              'Điểm LOSO lấy theo người từ EXP gốc. CSV mẫu chỉ kiểm tra nạp và dự đoán.',
              'LR dùng điểm quy trình tuần tự. Các bản refit không có thêm điểm test độc lập.',
              '', 'model_summary.csv: danh mục chi tiết và nguồn điểm.',
              'model_summary_nhu.csv: bảng bàn giao riêng của Như.',
              'score_mismatches.csv ở thư mục cha: các chỗ lệch cần hỏi lại.']
    (SUMMARY / 'Tong_hop_model.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')


def build_rows():
    rows, issues = [], []
    for name, task, folder in cac_model():
        row = dict.fromkeys(FIELDS, '')
        row.update(algorithm=name, task=task, model_path=tuong_doi(folder / f'{task}.joblib'))
        try:
            info = doc_metadata(name, task, folder)
            row.update(experiment_id=info['experiment_id'] or 'TODO',
                       metadata_path=tuong_doi(info['path']), metadata_loso_source=info['source'],
                       variant=info['variant'], evaluation_scope=info['scope'], fit_scope=info['fit_scope'])
            metrics, source, differences = diem_goc(info, task)
            row.update(metrics, loso_source=source, score_check='matched' if not differences else 'mismatch')
            for message in differences:
                issues.append((name, task, 'metadata_vs_loso', message))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(score_check='TODO')
            issues.append((name, task, 'missing_source', str(exc)))
        rows.append(row)
    return rows, issues


def compare_dummy(rows):
    source = ROOT / 'experiments/01_baselines/EXP-001_Dummy/results.csv'
    issues = []
    for task in ['binary', 'multiclass']:
        baseline = diem_theo_nguoi([r for r in doc_csv(source) if r['task'] == task])
        for row in rows:
            if row['task'] != task or row['algorithm'] == 'dummy' or row['score_check'] == 'TODO':
                continue
            for metric in ['macro_f1', 'balanced_accuracy']:
                if float(row[metric]) <= baseline[metric]:
                    issues.append((row['algorithm'], task, 'dummy_comparison',
                                   f'{metric}={row[metric]} không cao hơn Dummy={baseline[metric]}'))
    return issues


def compare_summary(path, rows):
    issues = []
    saved = doc_csv(path)
    expected = {r['model_path']: r for r in rows}
    seen = set()
    for row in saved:
        model = row.get('model_path', '')
        if model in seen or model not in expected:
            issues.append((row.get('algorithm', ''), row.get('task', ''), 'summary', 'trùng hoặc dư model: ' + model))
            continue
        seen.add(model)
        original = expected[model]
        for key in ['algorithm', 'task', 'experiment_id', 'loso_source', 'variant']:
            if row.get(key, '') != str(original[key]):
                issues.append((original['algorithm'], original['task'], 'summary', 'khác ' + key))
        for metric in ['macro_f1', 'balanced_accuracy', 'macro_f1_sd', 'balanced_accuracy_sd']:
            try:
                if not math.isclose(float(row[metric]), float(original[metric]), abs_tol=1e-10, rel_tol=0):
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                issues.append((original['algorithm'], original['task'], 'summary', f'{metric} lệch với EXP gốc'))
    for model in expected.keys() - seen:
        r = expected[model]
        issues.append((r['algorithm'], r['task'], 'summary', 'thiếu dòng model: ' + model))
    return issues


def write_summary():
    rows, issues = build_rows()
    if not rows:
        raise ValueError('chưa có joblib để lập danh mục')
    issues.extend(compare_dummy(rows))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    with (SUMMARY / 'model_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows, issues


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='đối chiếu CSV hiện có, không ghi lại danh mục')
    args = p.parse_args(argv)
    if args.check:
        rows, issues = build_rows()
        issues += compare_dummy(rows)
        issues += compare_summary(SUMMARY / 'model_summary.csv', rows)
    else:
        rows, issues = write_summary()
        write_overview(rows, issues)
    with (GROUP / 'score_mismatches.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'task', 'check', 'message'])
        w.writerows(issues)
    for row in rows:
        status = {'matched': 'khớp', 'mismatch': 'lệch', 'TODO': 'cần kiểm tra'}[row['score_check']]
        print(f"{row['algorithm']}/{row['task']}: {status}")
    for name, task, _, message in issues:
        print(f'{name}/{task}: {message}')
    print(f'Danh mục có {len(rows)} model, {len(issues)} mục cần hỏi lại.')
    return int(bool(issues))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
