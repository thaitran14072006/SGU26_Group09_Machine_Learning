"""Independently verify the completed selection stage and write its handoff."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wesad.data import SUBJECTS, load_feature_table
from wesad.lr_workflow import (audit_baselines, folder, freeze, identity, plan,
                               validate_completed)
from wesad.utils import file_hash, fingerprint, read_config


def main():
    cfg = plan()
    defaults = read_config(ROOT / 'configs/defaults.yaml')
    assert file_hash(ROOT / defaults['features']) == cfg['feature_sha256']
    frame, features = load_feature_table(ROOT / defaults['features'],
                                        ROOT / defaults['schema'], SUBJECTS)
    assert len(frame) == 6189 and len(features) == 68
    baselines = audit_baselines(cfg, frame)
    base = pd.read_csv(ROOT / 'experiments/01_baselines/EXP-002_LR/results.csv')
    rows = []
    checks = []
    for spec in cfg['experiments']:
        if spec['stage'] != 'feature_selection':
            continue
        receipt = validate_completed(spec)
        assert receipt['signature'] == fingerprint(identity(spec, cfg, None))
        out = folder(spec)
        metrics = pd.read_csv(out / 'results.csv')
        pred = pd.read_csv(out / 'predictions.csv', dtype={'window_id': str})
        details = json.loads((out / 'fold_details.json').read_text(encoding='utf8'))
        assert len(metrics) == 30 and len(pred) == 12378
        assert not metrics.duplicated(['task', 'subject_id']).any()
        assert not pred.duplicated(['task', 'subject_id', 'window_id']).any()
        assert set(metrics.task) == {'multiclass', 'binary'}
        for task in ('multiclass', 'binary'):
            labels = [1, 2, 3] if task == 'multiclass' else [0, 1]
            part = pred[pred.task == task]
            expected = frame[['subject_id', 'window_id', 'label_original']].copy()
            expected['window_id'] = expected.window_id.astype(str)
            merged = expected.merge(part, on=['subject_id', 'window_id'],
                                    how='outer', validate='one_to_one', indicator=True)
            assert len(merged) == 6189 and merged['_merge'].eq('both').all()
            truth = (merged.label_original == 2).astype(int) if task == 'binary' else merged.label_original
            assert np.array_equal(truth, merged.y_true)
            assert merged.y_pred.isin(labels).all()
            assert set(metrics.loc[metrics.task == task, 'subject_id']) == set(SUBJECTS)
            for subject in SUBJECTS:
                p = part[part.subject_id == subject]
                m = metrics[(metrics.task == task) & (metrics.subject_id == subject)].iloc[0]
                d = details[f'{task}/{subject}']
                assert len(p) == m.n_test
                assert np.isclose(m.macro_f1, f1_score(p.y_true, p.y_pred, labels=labels,
                                                      average='macro', zero_division=0))
                assert np.isclose(m.balanced_accuracy, balanced_accuracy_score(p.y_true, p.y_pred))
                assert d['confusion_matrix'] == confusion_matrix(p.y_true, p.y_pred, labels=labels).tolist()
                assert d['outer_test_subjects'] == [subject]
                assert set(d['outer_train_subjects']) == set(SUBJECTS) - {subject}
                assert not d['fatal_warnings'] and len(d['inner_splits']) == 3
                seen = []
                for split in d['inner_splits']:
                    a, b = set(split['train_subjects']), set(split['validation_subjects'])
                    assert not a & b and subject not in a | b
                    assert a | b == set(SUBJECTS) - {subject}
                    seen.extend(b)
                assert len(seen) == 14 and set(seen) == set(SUBJECTS) - {subject}
                assert len(d['selected_features']) == m.selected_k
                assert set(d['selected_features']) <= set(features)
                candidates = d['inner_candidates']
                assert len(candidates) == len(spec['candidates'])
                assert all(np.isfinite(c['mean_macro_f1']) for c in candidates)
                best = max(candidates, key=lambda c: c['mean_macro_f1'])
                assert best['params'] == d['selection_params']
                assert np.isclose(best['mean_macro_f1'], d['inner_macro_f1'])
            m = metrics[metrics.task == task]
            b = base[base.task == task]
            pairs = m.merge(b, on=['task', 'subject_id'], validate='one_to_one', suffixes=('', '_baseline'))
            assert len(pairs) == 15
            saved = pd.read_csv(out / 'summary.csv').set_index('task').loc[task]
            for metric in ('macro_f1', 'balanced_accuracy'):
                assert np.isclose(saved[metric + '_mean'], m[metric].mean())
                assert np.isclose(saved[metric + '_std'], m[metric].std(ddof=1))
            rows.append(dict(run_id=spec['run_id'], task=task,
                macro_f1_mean=m.macro_f1.mean(), macro_f1_std=m.macro_f1.std(ddof=1),
                balanced_accuracy_mean=m.balanced_accuracy.mean(),
                balanced_accuracy_std=m.balanced_accuracy.std(ddof=1),
                delta_macro_f1_vs_lr=(pairs.macro_f1 - pairs.macro_f1_baseline).mean(),
                source=out.relative_to(ROOT).as_posix()))
        checks.append({'run_id': spec['run_id'], 'status': 'passed',
                       'folds': 30, 'oof_predictions': 12378,
                       'acceptance_sha256': file_hash(out / 'acceptance.json')})
    gate = freeze('feature_selection', cfg)
    report = ROOT / 'reports/lr_selection_20261120'
    report.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(report / 'comparison.csv', index=False)
    sources = ['src/wesad/lr_workflow.py', 'src/wesad/data.py', 'src/wesad/utils.py',
               'scripts/run_lr_workflow.py', 'configs/lr_sequential_workflow.yaml',
               'scripts/continue_lr_experiments.py', 'scripts/verify_lr_selection.py']
    with zipfile.ZipFile(report / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(ROOT / source, source)
    now = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
    receipt = dict(status='passed',verified_at=now,baseline_audit=baselines,runs=checks,
                   gate_sha256=file_hash(ROOT / 'experiments/lr_feature_selection_gate.json'),
                   source_snapshot_sha256=file_hash(report / 'source_snapshot.zip'),
                   recipes=len(gate['recipes']),
                   checks=['receipt hashes', 'data checksum', 'OOF keys and labels',
                           'recomputed F1/BA/confusion matrices', 'subject-separated splits',
                           'inner candidate winner', 'summary sample SD'])
    (report / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    lines = ['# Kết quả giai đoạn chọn đặc trưng LR', '', f'Kiểm tra lúc {now}.', '',
             'So sánh ANOVA (FS-01), mutual information (FS-02) và loại tương quan cao (FS-03). '
             'Thời gian chạy và cấu hình được lưu trong hồ sơ từng run.', '',
             '| Run | Task | Macro-F1 ± SD | Balanced Accuracy ± SD | Δ F1 so LR đối chứng |',
             '|---|---|---|---|---|']
    for r in scores.itertuples():
        lines.append(f'| {r.run_id} | {r.task} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | '
                     f'{r.balanced_accuracy_mean:.4f} ± {r.balanced_accuracy_std:.4f} | '
                     f'{r.delta_macro_f1_vs_lr:+.4f} |')
    lines += ['', 'Đủ 3 run × 30 fold; mỗi run có 12.378 dự đoán OOF. Kiểm tra độc lập đạt. '
              'Gate selection khóa 30 công thức theo task/outer subject bằng inner Macro-F1. '
              'Không dùng bảng điểm outer ở trên để chọn phương pháp cho bước sau.', '',
              'Kết quả tăng/giảm phụ thuộc phương pháp và người. Không kết luận khác biệt có ý nghĩa '
              'thống kê từ chênh lệch trung bình này. Điểm inner là tiêu chí lựa chọn và có thể lạc quan; '
              'benchmark đã được sử dụng nhiều lần, chưa có external holdout.', '',
              'Mỗi bài toán được đánh giá riêng trên cùng 15 subject. Các chênh lệch trong bảng '
              'được tính so với LR đối chứng trên các cặp subject tương ứng.', '',
              'Nguồn: comparison.csv, verification.json, và các thư mục run ghi trong cột source.']
    (report / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps(receipt, indent=2))
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
