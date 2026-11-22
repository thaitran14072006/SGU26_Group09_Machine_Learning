"""Build the EXP-009 report notebook; execution is a separate verification step."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
EXP = 'experiments/02_tuning/EXP-009_RF'
cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n'})


def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n',
                  'execution_count': None, 'outputs': []})


md('''
# EXP-009 — Tinh chỉnh Random Forest

**Người thực hiện:** Mai. **Ngày thực hiện:** 20/11/2026.

**Trạng thái:** phân tích kết quả G4 đã lưu và chạy lại nested LOSO ngày 20/11/2026.
Lượt chạy mới có 30 outer fold, gồm 15 người cho binary và 15 người cho multiclass.
Kết quả lịch sử và kết quả chạy lại được lưu riêng, không cộng thành 60 quan sát độc lập.

Notebook trình bày câu hỏi, cấu hình, cách chạy, kết quả và kết luận của RF tuning.
Đối chứng chính là RF mặc định 100 cây của EXP-005, không phải khảo sát RF 20 cây bổ sung.
''')
md('''
## 02 — Câu hỏi nghiên cứu

Thay đổi `n_estimators`, `max_depth`, `max_features` qua 4 ứng viên G4 có cải thiện
Macro-F1 và Balanced Accuracy ngoài người học so với RF mặc định không?

Giả thuyết: chọn cấu hình bằng inner GroupKFold có thể cải thiện điểm test, nhưng mức tăng
có thể khác giữa binary và multiclass, và giữa các người tham gia. Chỉ thay tham số RF;
giữ dữ liệu physiological68, seed, nhãn, quy trình train-only và không cân bằng lớp.
Vì 4 ứng viên thay nhiều tham số cùng lúc, thí nghiệm này không tách riêng tác động nhân quả của từng tham số.
''')
code('''
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

here = Path.cwd()
repo = next(p for p in (here, *here.parents) if (p / 'src/wesad/evaluation.py').is_file())
experiment = repo / 'experiments/02_tuning/EXP-009_RF'
config = yaml.safe_load((experiment / 'config.yaml').read_text(encoding='utf-8'))
metrics = ['macro_f1', 'balanced_accuracy']
tasks = ['binary', 'multiclass']
task_names = {'binary': 'Nhị phân', 'multiclass': 'Ba lớp'}
figures = experiment / 'figures'
figures.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
print('Thư mục hồ sơ:', experiment)
print('Môi trường phân tích:', sys.version.split()[0])
''')
md('''
## 03 — Dữ liệu

Sử dụng bảng physiological68 đã khóa SHA256, cùng 15 subject như EXP-005.
Binary: 0 = không stress (baseline và amusement), 1 = stress.
Multiclass: 1 = baseline, 2 = stress, 3 = amusement.
Mã người và mã cửa sổ chỉ dùng để chia fold/đối chiếu; không đưa vào 68 biến đầu vào.
''')
code('''
data_path = repo / config['features']
schema_path = repo / config['schema']
digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
assert digest == config['feature_sha256'], 'Dữ liệu không còn khớp lượt chạy'
schema = json.loads(schema_path.read_text(encoding='utf-8'))
data = pd.read_csv(data_path, dtype={'subject_id': str, 'window_id': str})
assert len(schema['feature_columns']) == 68
assert set(data.subject_id) == set(config['expected_subjects'])
assert not data.duplicated(['subject_id', 'window_id']).any()
print('CSV:', config['features'])
print('SHA256:', digest)
print('Số dòng:', len(data), '| Số đặc trưng:', len(schema['feature_columns']), '| Số người:', data.subject_id.nunique())
display(data.groupby('subject_id').size().rename('Số cửa sổ').to_frame())
display(data.label_original.value_counts().sort_index().rename('Số cửa sổ theo nhãn').to_frame())
''')
md('''
## 04 — Protocol đánh giá

- Outer LOSO: giữ riêng 1 người test, 14 người train, đủ 15 lượt cho từng bài toán.
- Inner GroupKFold: chia 14 người train thành 3 fold theo người. Chọn ứng viên bằng trung bình Macro-F1 của 3 inner fold.
- Mỗi lần fit dùng median imputation học trên train của fold đó. RF không cần scaling trong pipeline dùng chung.
- Sau lựa chọn, refit trên đủ outer train rồi dự đoán người outer test; outer test không tham gia chọn cấu hình.
- Seed 42; RF và GridSearchCV dùng `n_jobs=1`. Không feature selection, resampling hoặc class weighting.
- Macro-F1 dùng nhãn cố định và `zero_division=0`; Balanced Accuracy dùng các lớp thực sự xuất hiện trong test.
- Báo cáo mean và sample SD (`ddof=1`) theo 15 người. Confusion matrix tổng hợp dự đoán OOF có trọng số theo số cửa sổ, không thay thế mean theo người.

4 ứng viên × 3 inner fold × 30 outer fold = 360 inner fit, cộng 30 outer refit.
Mã huấn luyện dùng `src/wesad/evaluation.py`; notebook không viết lại LOSO.
''')
md('''
## 05 — Cấu hình

Baseline EXP-005: `n_estimators=100`, `max_depth=None`, `max_features='sqrt'`, `class_weight=None`.
4 cấu hình dưới giữ nguyên grid G4. C1–C4 chỉ là tên để đọc biểu đồ, không phải thứ hạng.
`max_features=0.5` là xét khoảng một nửa số biến tại mỗi lần chia nhánh; cả 68 biến vẫn được đưa vào RF.
''')
code('''
grid = [dict(zip(item.keys(), [v[0] for v in item.values()])) for item in config['search_spaces']['rf']]
candidate_names = {json.dumps(params, sort_keys=True): f'C{i + 1}' for i, params in enumerate(grid)}
display(pd.DataFrame(grid, index=[f'C{i + 1}' for i in range(len(grid))]))
print((experiment / 'config.yaml').read_text(encoding='utf-8'))
''')
md('''
## 06 — Code thực thi

`RUN_TRAIN=False` mặc định để mở notebook đọc kết quả mà không huấn luyện lại.
Bật khi chủ động chạy/tiếp tục đúng cấu hình của hồ sơ. Runner chỉ bỏ qua các fold có chữ ký
và checksum hợp lệ. Khi thay dữ liệu, mã train hoặc cấu hình, phải dùng thư mục output mới.

Kết quả máy lưu tại `outputs/exp009_rf_mai_20261120/`; bản chốt và bằng chứng nằm trong thư mục experiment.
Môi trường training là `.venv-g1` Python 3.12.14; môi trường vẽ notebook có thể khác và được ghi riêng.
''')
code('''
RUN_TRAIN = False
train_python = repo / '.venv-g1/Scripts/python.exe'
train_command = [str(train_python), '-X', 'utf8', 'scripts/run_exp009_rf.py', '--config',
                 str(experiment / 'config.yaml'), '--resume']
report_command = [str(train_python), '-X', 'utf8', 'scripts/report_exp009_rf.py']
print('Lệnh chạy:', subprocess.list2cmdline(train_command))
if RUN_TRAIN:
    subprocess.run(train_command, cwd=repo, check=True)
    subprocess.run(report_command, cwd=repo, check=True)
else:
    print('Chỉ đọc kết quả đã lưu; không fit mô hình trong lần mở notebook này.')
''')
md('''
## 07 — Kết quả và kiểm tra bằng chứng

`results.csv` là lượt chạy lại ngày 20/11. `results_historical.csv` là nguồn G4 đã kiểm chứng
từ 2 gói RF/MLP được ghép đủ 15 người; chỉ lấy RF. `results_baseline.csv` là EXP-005 đã xác minh.
Các bảng được ghép theo task và subject, không ghép theo thứ tự dòng.
''')
code('''
proof = json.loads((experiment / 'verification.json').read_text(encoding='utf-8'))
provenance = json.loads((experiment / 'provenance.json').read_text(encoding='utf-8'))
assert proof['status'] == 'passed' and proof['n_folds'] == 30
assert proof['n_predictions'] == len(data) * 2
assert proof['producer']['data_sha256'] == digest
results = pd.read_csv(experiment / 'results.csv')
predictions = pd.read_csv(experiment / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
summary = pd.read_csv(experiment / 'comparison_summary.csv')
paired = pd.read_csv(experiment / 'comparison_exp005.csv')
history_comparison = pd.read_csv(experiment / 'comparison_historical.csv')
assert len(results) == len(paired) == len(history_comparison) == 30
assert not results.duplicated(['task', 'subject_id']).any()
assert not predictions.duplicated(['task', 'subject_id', 'window_id']).any()
assert all(set(g.subject_id) == set(config['expected_subjects']) for _, g in results.groupby('task'))
# Verify all saved fold files against their signed completion manifest.
for task in config['tasks']:
    for subject in config['expected_subjects']:
        folder = experiment / 'folds' / task / 'rf' / subject
        done = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        assert done['signature'] == proof['producer']['signature']
        for name, expected_hash in done['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected_hash
display(summary.round(4))
display(results[['task', 'subject_id'] + metrics].round(4))
display(Markdown('**Mean ± SD theo người**'))
for row in summary.itertuples():
    print(f'{row.variant}, {row.task}: Macro-F1 {row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f}; '
          f'BA {row.balanced_accuracy_mean:.4f} ± {row.balanced_accuracy_std:.4f}')
print('Đối chiếu G4 lịch sử:', provenance['historical_comparison'])
''')
md('''
## 08 — Hình minh họa

Hình lấy trực tiếp từ CSV của EXP-005 và lượt chạy lại EXP-009.
Thanh sai số là SD giữa người, không phải khoảng tin cậy.
''')
code('''
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for i, task in enumerate(tasks):
    for j, metric in enumerate(metrics):
        part = summary.loc[summary.task.eq(task)].set_index('variant').loc[['baseline_EXP005', 'rerun_EXP009']]
        ax = axes[i, j]
        ax.bar(['RF mặc định', 'RF tuning'], part[metric + '_mean'], yerr=part[metric + '_std'],
               color=['#879CAF', '#245C8C'], capsize=5)
        ax.set_title(f'{task_names[task]} — {metric}')
        ax.set_ylim(0, 1.05); ax.set_ylabel('Điểm mean ± SD theo người')
fig.tight_layout()
fig.savefig(figures / 'default_vs_tuned.png', dpi=150, bbox_inches='tight')
plt.show()
''')
code('''
order = config['expected_subjects']
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, task in zip(axes, tasks):
    part = paired.loc[paired.task.eq(task)].set_index('subject_id').loc[order]
    x = np.arange(len(order))
    ax.bar(x - .18, part.macro_f1_delta, .36, label='Δ Macro-F1', color='#245C8C')
    ax.bar(x + .18, part.balanced_accuracy_delta, .36, label='Δ Balanced Accuracy', color='#D69336')
    ax.axhline(0, color='#555555', linewidth=.8)
    ax.set_title(task_names[task]); ax.set_ylabel('Tuning − mặc định'); ax.legend()
axes[-1].set_xticks(np.arange(len(order)), order); axes[-1].set_xlabel('Người outer test')
fig.tight_layout(); fig.savefig(figures / 'delta_by_subject.png', dpi=150, bbox_inches='tight')
plt.show()
''')
code('''
frequency = pd.read_csv(experiment / 'candidate_frequency.csv')
frequency['candidate'] = frequency.configuration.map(candidate_names)
assert frequency.candidate.notna().all()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, task in zip(axes, tasks):
    values = frequency.loc[frequency.task.eq(task)].set_index('candidate').n_subjects.reindex(['C1', 'C2', 'C3', 'C4'], fill_value=0)
    assert values.sum() == 15
    ax.bar(values.index, values, color='#245C8C')
    ax.set_title(task_names[task]); ax.set_ylim(0, 16); ax.set_ylabel('Số outer fold chọn cấu hình')
    for i, v in enumerate(values): ax.text(i, v + .2, str(v), ha='center')
fig.tight_layout(); fig.savefig(figures / 'candidate_frequency.png', dpi=150, bbox_inches='tight')
plt.show()
display(pd.read_csv(experiment / 'best_params_by_subject.csv'))
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, task in zip(axes, tasks):
    saved = json.loads((experiment / f'confusion_matrix_{task}.json').read_text(encoding='utf-8'))
    matrix = np.asarray(saved['matrix'])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    labels = ['Không stress', 'Stress'] if task == 'binary' else ['Baseline', 'Stress', 'Amusement']
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{matrix[i,j]}\\n{normalized[i,j]:.1%}', ha='center', va='center',
                    color='white' if normalized[i,j] > .55 else '#172B3A')
    ax.set_xticks(range(len(labels)), labels, rotation=15)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Dự đoán'); ax.set_ylabel('Nhãn thực tế'); ax.set_title(task_names[task])
fig.subplots_adjust(left=.10, right=.84, bottom=.20, top=.86, wspace=.45)
cax = fig.add_axes([.90, .22, .018, .62])
fig.colorbar(im, cax=cax, label='Tỷ lệ theo hàng')
fig.savefig(figures / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
''')
md('''
## 09 — Phân tích

Đánh giá đồng thời điểm trung bình, biến thiên theo người, số người được cải thiện và lỗi theo lớp.
Recall bên dưới được tính trên OOF gộp mọi cửa sổ; mỗi người có thể đóng góp số cửa sổ khác nhau.
Thời gian tìm kiếm/fit không phải độ trễ suy luận, và chỉ mô tả tài nguyên của lượt chạy này.
''')
code('''
lines = []
for task in tasks:
    part = paired.loc[paired.task.eq(task)]
    tuned = results.loc[results.task.eq(task)]
    weak = tuned.nsmallest(2, 'macro_f1')
    df1, dba = part.macro_f1_delta.mean(), part.balanced_accuracy_delta.mean()
    wins = int(part.macro_f1_delta.gt(1e-10).sum())
    losses = int(part.macro_f1_delta.lt(-1e-10).sum())
    ties = len(part) - wins - losses
    worst = part.loc[part.macro_f1_delta.idxmin()]
    best_row = part.loc[part.macro_f1_delta.idxmax()]
    score = tuned.macro_f1.mean()
    lines.append(f"### {task_names[task]}\\n"
        f"RF tuning đạt Macro-F1 {score:.4f} ± {tuned.macro_f1.std(ddof=1):.4f}. "
        f"So với EXP-005, ΔMacro-F1 {df1:+.4f}, ΔBA {dba:+.4f}. "
        f"Macro-F1 tăng ở {wins}/15 người, giảm ở {losses}/15 và không đổi ở {ties}/15. "
        f"Mức tăng lớn nhất ở {best_row.subject_id} ({best_row.macro_f1_delta:+.4f}); "
        f"thay đổi thấp nhất ở {worst.subject_id} ({worst.macro_f1_delta:+.4f}). "
        f"Hai fold có điểm tuning thấp nhất: " + ', '.join(f'{r.subject_id} ({r.macro_f1:.4f})' for r in weak.itertuples()) + '.')
display(Markdown('\\n\\n'.join(lines)))
display(pd.read_csv(experiment / 'classification_report.csv').round(4))
display(summary[['variant', 'task', 'fit_and_search_seconds_mean']].round(2))
display(Markdown('Không thể quy mọi thay đổi cho một tham số riêng vì grid đồng thời thay số cây, độ sâu và số biến xét tại mỗi lần chia. Các chênh lệch là kết quả quan sát, chưa có kiểm định ý nghĩa thống kê.'))
''')
md('''
## 10 — Kết luận và provenance

Kết luận dưới được tạo từ số liệu thật, riêng cho từng task. EXP-009 chỉ đánh giá quy trình tuning;
chưa chọn và fit model cuối để demo. Cấu hình được chọn có thể khác giữa các outer fold.
Không dùng outer test để chọn lại tham số hoặc tự coi cấu hình xuất hiện nhiều nhất là cấu hình triển khai.
''')
code('''
conclusions = []
for task in tasks:
    part = paired.loc[paired.task.eq(task)]
    delta = part.macro_f1_delta.mean()
    direction = 'tăng' if delta > 1e-10 else 'giảm' if delta < -1e-10 else 'không đổi'
    conclusions.append(f"- **{task_names[task]}:** Macro-F1 trung bình {direction} ({delta:+.4f}); "
                       f"Balanced Accuracy thay đổi {part.balanced_accuracy_delta.mean():+.4f}. "
                       'Cần cân nhắc cả biến thiên theo người và chi phí tìm kiếm khi chọn mô hình cuối.')
display(Markdown('\\n'.join(conclusions)))
history_info = provenance['historical_comparison']
print('Số fold khác điểm so với G4 lịch sử:', history_info['n_changed_metric_folds'])
print('Chênh lệch metric tuyệt đối lớn nhất:', history_info['max_abs_metric_difference'])
print('Số fold khác tham số:', history_info['n_changed_parameter_folds'])
print('Nguồn train mới:', provenance['training_source'])
print('Nguồn baseline:', provenance['baseline_source'])
print('Git commit train:', proof['producer'].get('git_commit'))
print('Nếu commit là None, môi trường local không cung cấp được metadata Git; không gán commit giả.')
display(proof['producer']['environment'])
print('Chữ ký lượt train:', proof['producer']['signature'])
print('Mã nguồn train đã lưu trong source_snapshot.zip; checksum từng fold trong complete.json.')
print('Nguồn lịch sử và môi trường gốc: historical_provenance.json.')
print('Lần thực thi notebook này chỉ đọc báo cáo và tạo hình khi RUN_TRAIN=False.')
''')

nb = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'exp009-{i:02d}'
path = ROOT / EXP / 'experiment.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
working = ROOT / 'notebooks/experiments/Mai_EXP-009_RF_tuning.ipynb'
working.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
print(path)
