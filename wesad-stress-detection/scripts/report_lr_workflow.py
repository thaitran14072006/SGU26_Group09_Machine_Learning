"""Verify the full sequential protocol and export subject-level diagnostics."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/lr_workflow/matplotlib_cache'))
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from verify_lr_run import verify
from wesad.data import SUBJECTS, load_feature_table
from wesad.lr_workflow import audit_baselines, folder, freeze, plan, write_state
from wesad.utils import file_hash, read_config


def main():
    cfg = plan()
    checks = [verify(s['run_id']) for s in cfg['experiments']]
    defaults = read_config(ROOT / 'configs/defaults.yaml')
    frame, _ = load_feature_table(ROOT / defaults['features'], ROOT / defaults['schema'], SUBJECTS)
    baselines = audit_baselines(cfg, frame)
    gates = {stage: freeze(stage, cfg) for stage in cfg['order']}
    write_state(cfg)
    out = ROOT / 'reports/lr_workflow_20261121'
    out.mkdir(parents=True, exist_ok=True)
    tables, predictions = {}, {}
    for name, path in [('Dummy', ROOT / 'experiments/01_baselines/EXP-001_Dummy/runs/20261114'),
                       ('LR_baseline', ROOT / 'experiments/01_baselines/EXP-002_LR')]:
        tables[name] = pd.read_csv(path / 'results.csv')
        predictions[name] = pd.read_csv(path / 'predictions.csv')
    for spec in cfg['experiments']:
        path = folder(spec)
        tables[spec['run_id']] = pd.read_csv(path / 'results.csv')
        predictions[spec['run_id']] = pd.read_csv(path / 'predictions.csv')
    decisions = []
    for stage, gate in gates.items():
        ms, ps = [], []
        for key, recipe in gate['recipes'].items():
            task, subject = key.split('/')
            run = recipe['source_run']
            for source, target in [(tables[run], ms), (predictions[run], ps)]:
                target.append(source[(source.task == task) & (source.subject_id == subject)].copy())
            decisions.append({'stage': stage, 'task': task, 'subject_id': subject, **recipe})
        tables[stage] = pd.concat(ms, ignore_index=True)
        predictions[stage] = pd.concat(ps, ignore_index=True)
        assert len(tables[stage]) == 30 and len(predictions[stage]) == 12378
        tables[stage].to_csv(out / f'{stage}_results.csv', index=False)
        predictions[stage].to_csv(out / f'{stage}_predictions.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'inner_decisions.csv', index=False)
    log = load_workbook(ROOT / 'experiments/experiments_logs.xlsx', read_only=True, data_only=True)
    log_rows = list(log['Experiment Log'].iter_rows(min_row=2, values_only=True))
    for spec in cfg['experiments']:
        saved = pd.read_csv(folder(spec) / 'summary.csv').set_index('task')
        for task in ('binary', 'multiclass'):
            key = f'{spec["experiment_id"]}/{spec["run_id"]}/{task}'
            entries = [r for r in log_rows if r[0] == key]
            assert len(entries) == 1, f'Missing/duplicate log: {key}'
            assert np.isclose(entries[0][8], saved.loc[task, 'macro_f1_mean'])
            assert np.isclose(entries[0][9], saved.loc[task, 'balanced_accuracy_mean'])
    log.close()
    all_scores, paired = [], []
    for name, metrics in tables.items():
        for task, part in metrics.groupby('task'):
            base = tables['LR_baseline'].query('task == @task')
            pairs = part.merge(base, on=['task', 'subject_id'], validate='one_to_one', suffixes=('', '_baseline'))
            delta = pairs.macro_f1 - pairs.macro_f1_baseline
            all_scores.append({'variant': name, 'task': task,
                'macro_f1_mean': part.macro_f1.mean(), 'macro_f1_std': part.macro_f1.std(ddof=1),
                'balanced_accuracy_mean': part.balanced_accuracy.mean(),
                'balanced_accuracy_std': part.balanced_accuracy.std(ddof=1),
                'delta_f1_vs_lr': delta.mean(), 'subjects_improved': int((delta > 1e-12).sum()),
                'subjects_worse': int((delta < -1e-12).sum())})
            for row in pairs.itertuples():
                paired.append({'variant': name, 'task': task, 'subject_id': row.subject_id,
                               'macro_f1': row.macro_f1, 'baseline_f1': row.macro_f1_baseline,
                               'delta_f1': row.macro_f1 - row.macro_f1_baseline})
    scores = pd.DataFrame(all_scores)
    scores.to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'paired_subject_deltas.csv', index=False)
    step_rows = []
    references = {'feature_selection': 'LR_baseline', 'balancing': 'feature_selection',
                  'tuning': 'balancing', 'BAL-01': 'feature_selection',
                  'BAL-02': 'BAL-01', 'BAL-03': 'BAL-01'}
    references.update({f'TUNE-{i:02d}': 'balancing' for i in range(1, 6)})
    for variant, reference in references.items():
        pairs = tables[variant].merge(tables[reference], on=['task', 'subject_id'],
                                     validate='one_to_one', suffixes=('', '_reference'))
        for r in pairs.itertuples():
            step_rows.append({'variant': variant, 'reference': reference, 'task': r.task,
                              'subject_id': r.subject_id,
                              'delta_f1': r.macro_f1 - r.macro_f1_reference,
                              'delta_ba': r.balanced_accuracy - r.balanced_accuracy_reference})
    pd.DataFrame(step_rows).to_csv(out / 'paired_step_deltas.csv', index=False)
    final = tables['tuning'].copy()
    final['train_test_gap'] = final.train_macro_f1 - final.macro_f1
    final.to_csv(out / 'final_subject_diagnostics.csv', index=False)
    per_class = []
    for task, part in predictions['tuning'].groupby('task'):
        labels = [0, 1] if task == 'binary' else [1, 2, 3]
        names = ['Non-stress', 'Stress'] if task == 'binary' else ['Baseline', 'Stress', 'Amusement']
        report = classification_report(part.y_true, part.y_pred, labels=labels,
                                        target_names=names, output_dict=True, zero_division=0)
        for name in names:
            per_class.append({'task': task, 'class': name, **report[name]})
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(part.y_true, part.y_pred, labels=labels,
            display_labels=names, normalize='true', values_format='.2f', cmap='Blues', ax=ax)
        ax.set_title(f'Nested LR: {task} (pooled OOF)')
        fig.tight_layout()
        fig.savefig(out / f'confusion_{task}.png', dpi=160)
        plt.close(fig)
    pd.DataFrame(per_class).to_csv(out / 'pooled_class_metrics.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    variants = ['Dummy', 'LR_baseline', 'feature_selection', 'balancing', 'tuning']
    for i, task in enumerate(['multiclass', 'binary']):
        s = scores[scores.task == task].set_index('variant').loc[variants]
        axes[i, 0].bar(variants, s.macro_f1_mean, yerr=s.macro_f1_std, capsize=4)
        axes[i, 0].tick_params(axis='x', rotation=20)
        axes[i, 0].set(ylim=(0, 1.05), ylabel='Subject Macro-F1 mean +/- SD', title=task)
        f = final[final.task == task].set_index('subject_id').loc[list(SUBJECTS)]
        axes[i, 1].plot(SUBJECTS, f.train_macro_f1, marker='o', label='Train')
        axes[i, 1].plot(SUBJECTS, f.macro_f1, marker='o', label='Outer test')
        axes[i, 1].tick_params(axis='x', rotation=45)
        axes[i, 1].set(ylim=(0, 1.05), ylabel='Macro-F1', title=f'{task}: final train/test gap')
        axes[i, 1].legend()
    fig.tight_layout()
    fig.savefig(out / 'workflow_and_gap.png', dpi=160)
    plt.close(fig)
    lines = ['# LR: kết quả chuỗi thực nghiệm ngày 21/11/2026', '',
        'Phạm vi: phần Dummy/LR của Thái. Hoàn tất 3 selection + 3 balancing + 5 tuning; '
        'selection đã hoàn tất trước ngày 21/11. Tám run BAL/TUNE được chạy và commit riêng ngày 21/11.', '',
        '## Thiết kế và nghiệm thu', '',
        '6.189 cửa sổ, 68 đặc trưng, 15 subject; hai bài toán. Mỗi run có 30 outer LOSO fold '
        'và 12.378 dự đoán OOF. Inner GroupKFold(3) chỉ dùng 14 subject train. '
        'Median imputation, scaling, selection và trọng số lớp được fit trong pipeline. '
        'Mọi run được kiểm tra checksum, khóa OOF, nhãn, điểm F1/BA tính lại, confusion matrix, '
        'split theo người, gate phụ thuộc, cấu hình và SD mẫu.', '',
        'Các dòng feature_selection/balancing/tuning ghép kết quả theo quyết định inner CV '
        'riêng từng task/outer subject; không chọn run theo điểm outer test. '
        'Chi tiết từng run nằm trong comparison.csv; quyết định nằm trong inner_decisions.csv.', '',
        '| Giai đoạn | Task | Macro-F1 ± SD | Balanced Accuracy ± SD | Δ F1 so LR | Subject tăng/giảm |',
        '|---|---|---|---|---|---|']
    for name in variants:
        for r in scores[scores.variant == name].itertuples():
            lines.append(f'| {name} | {r.task} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | '
                f'{r.balanced_accuracy_mean:.4f} ± {r.balanced_accuracy_std:.4f} | '
                f'{r.delta_f1_vs_lr:+.4f} | {r.subjects_improved}/{r.subjects_worse} |')
    lines += ['', '## Từng phương án cân bằng và tuning', '',
        'Các số dưới đây dùng để mô tả kết quả; quyết định chuyển bước vẫn lấy inner CV. '
        'BAL-02/03 được đối chiếu với BAL-01 để tách tác động trọng số lớp. '
        'Mỗi TUNE được đối chiếu với gate balancing. paired_step_deltas.csv chứa đủ cặp subject.', '',
        '| Run | Task | Macro-F1 ± SD | Δ F1 so bước đối chiếu | Subject tăng/giảm |',
        '|---|---|---|---|---|']
    steps = pd.DataFrame(step_rows)
    for r in scores[scores.variant.str.startswith(('BAL-', 'TUNE-'))].itertuples():
        changes = steps[(steps.variant == r.variant) & (steps.task == r.task)].delta_f1
        lines.append(f'| {r.variant} | {r.task} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | '
                     f'{changes.mean():+.4f} | {int((changes > 1e-12).sum())}/{int((changes < -1e-12).sum())} |')
    lines += ['', '## Phân tích lỗi', '']
    for task, part in final.groupby('task'):
        worst = part.nsmallest(3, 'macro_f1')
        lines.append(f'- {task}: subject khó nhất: ' + ', '.join(
            f'{r.subject_id} (F1={r.macro_f1:.4f})' for r in worst.itertuples()) +
            f'. Train-test gap trung bình {part.train_test_gap.mean():.4f}.')
    for r in per_class:
        lines.append(f'- {r["task"]}/{r["class"]}: precision={r["precision"]:.4f}, '
                     f'recall={r["recall"]:.4f}, F1={r["f1-score"]:.4f} (gộp cửa sổ OOF).')
    binary = predictions['tuning'].query("task == 'binary'")
    fn = int(((binary.y_true == 1) & (binary.y_pred == 0)).sum())
    fp = int(((binary.y_true == 0) & (binary.y_pred == 1)).sum())
    lines += [f'- Binary: bỏ sót Stress {fn} cửa sổ; báo động sai {fp} cửa sổ.', '',
        '![Quy trình và train-test gap](workflow_and_gap.png)', '',
        '![Nhầm lẫn multiclass](confusion_multiclass.png)', '',
        '![Nhầm lẫn binary](confusion_binary.png)', '',
        '## Kết luận và giới hạn', '',
        'Báo cáo đầy đủ mức tăng/giảm theo subject trong paired_subject_deltas.csv. '
        'Điểm từng lớp và confusion matrix gộp cửa sổ dùng để phân tích lỗi, '
        'không thay thế mean ± SD theo 15 subject. Train score dùng chẩn đoán, không phải test score. '
        'Không suy ra ý nghĩa thống kê từ chênh lệch trung bình.', '',
        'Việc tái sử dụng inner splits ở các bước là quy trình lựa chọn tuần tự đã định trước; '
        'inner score có thể lạc quan. WESAD đã được nhóm xem nhiều lần, chỉ có 15 người trong '
        'phòng thí nghiệm và chưa có external holdout. Kết quả mang tính thăm dò; '
        'không tuyên bố LR là mô hình tốt nhất nhóm khi chưa so sánh KNN/SVM/RF/MLP.', '',
        'Bàn giao tiếp theo theo lịch 22/11: xuất model LR ứng viên, schema/metadata, '
        'kiểm tra save/load và ví dụ dự đoán CSV; ghép kết quả các thành viên trước 10:00.', '',
        '## Tái kiểm tra', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/report_lr_workflow.py', '```', '',
        'Nguồn ảnh và số liệu: các runs trong experiments, các gate lr_*_gate.json; '
        'verification.json ghi hash hồ sơ nguồn và mọi file báo cáo.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    receipt = {'status': 'passed', 'verified_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
        'runs': checks, 'baseline_audit': baselines,
        'experiment_log_entries_verified': 22,
        'sources': {name: file_hash(ROOT / name) for name in
                    ['scripts/verify_lr_run.py', 'scripts/report_lr_workflow.py',
                     'src/wesad/lr_workflow.py', 'configs/lr_sequential_workflow.yaml']},
        'gates': {stage: file_hash(ROOT / f'experiments/lr_{stage}_gate.json') for stage in gates},
        'report_files': {p.name: file_hash(p) for p in sorted(out.iterdir())
                         if p.is_file() and p.name != 'verification.json'}}
    (out / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(scores[scores.variant.isin(variants)].to_string(index=False))
    print('VERIFIED: 11 runs, 330 folds, 136158 OOF predictions; report:', out)


if __name__ == '__main__':
    main()
