"""Select SVM candidates by inner score per outer fold; export full diagnostics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from exp008_thai import BASE, setup, verify, score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from wesad.data import SUBJECTS
from wesad.utils import file_hash, write_json


def main():
    cfg,frame,_=setup()
    checks=[verify(s['run_id']) for s in cfg['runs']]
    out=ROOT/'reports/exp008_thai_20261122'
    out.mkdir(parents=True,exist_ok=True)
    metrics,predictions,details={}, {}, {}
    for spec in cfg['runs']:
        run=spec['run_id'];folder=BASE/'runs'/run
        metrics[run]=pd.read_csv(folder/'results.csv')
        predictions[run]=pd.read_csv(folder/'predictions.csv')
        details[run]=json.loads((folder/'fold_details.json').read_text())
    chosen,mparts,pparts=[],[],[]
    for task in ['multiclass','binary']:
        for subject in SUBJECTS:
            rows=[(s,metrics[s['run_id']].query('task == @task and subject_id == @subject').iloc[0]) for s in cfg['runs']]
            spec,row=max(rows,key=lambda item:item[1].inner_macro_f1)
            run=spec['run_id']
            assert all(details[run][f'{task}/{subject}']['inner_splits']==details[s['run_id']][f'{task}/{subject}']['inner_splits'] for s in cfg['runs'])
            chosen.append(dict(task=task,subject_id=subject,**spec,inner_macro_f1=row.inner_macro_f1))
            mparts.append(metrics[run].query('task == @task and subject_id == @subject'))
            pparts.append(predictions[run].query('task == @task and subject_id == @subject'))
    final=pd.concat(mparts,ignore_index=True);pred=pd.concat(pparts,ignore_index=True)
    assert len(final)==30 and len(pred)==12378
    final['train_test_gap']=final.train_macro_f1-final.macro_f1
    final.to_csv(out/'nested_results.csv',index=False)
    pred.to_csv(out/'nested_predictions.csv',index=False)
    pd.DataFrame(chosen).to_csv(out/'inner_decisions.csv',index=False)
    pd.DataFrame(chosen).groupby(['task','run_id']).size().rename('subjects_selected').reset_index().to_csv(out/'selection_counts.csv',index=False)
    metrics['nested_selection']=final
    rows=[]
    for run,table in metrics.items():
        params=next((s for s in cfg['runs'] if s['run_id']==run),{'C':'per_fold_inner_cv','gamma':'per_fold_inner_cv'})
        for task,part in table.groupby('task'):
            rows.append(dict(run_id=run,C=params['C'],gamma=params['gamma'],task=task,macro_f1_mean=part.macro_f1.mean(),macro_f1_std=part.macro_f1.std(ddof=1),
                balanced_accuracy_mean=part.balanced_accuracy.mean(),balanced_accuracy_std=part.balanced_accuracy.std(ddof=1)))
    summary=pd.DataFrame(rows);summary.to_csv(out/'comparison.csv',index=False)
    lr=pd.read_csv(ROOT/'reports/lr_workflow_20261121/tuning_results.csv')
    pairs=final.merge(lr,on=['task','subject_id'],validate='one_to_one',suffixes=('_svm','_lr'))
    pairs['delta_f1_svm_minus_lr']=pairs.macro_f1_svm-pairs.macro_f1_lr
    pairs.to_csv(out/'paired_lr_comparison.csv',index=False)
    classes=[]
    for task,part in pred.groupby('task'):
        labels=[0,1] if task=='binary' else [1,2,3]
        names=['Non-stress','Stress'] if task=='binary' else ['Baseline','Stress','Amusement']
        fig,ax=plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay.from_predictions(part.y_true,part.y_pred,labels=labels,display_labels=names,
            normalize='true',values_format='.2f',cmap='Blues',ax=ax)
        ax.set_title(f'EXP-008 inner-selected SVM: {task}')
        fig.tight_layout();fig.savefig(out/f'confusion_{task}.png',dpi=150);plt.close(fig)
        report=classification_report(part.y_true,part.y_pred,labels=labels,target_names=names,zero_division=0,output_dict=True)
        classes.extend({'task':task,'class':name,**report[name]} for name in names)
        for subject in SUBJECTS:
            p=part[part.subject_id==subject];r=final.query('task == @task and subject_id == @subject').iloc[0]
            for key,value in score(p.y_true,p.y_pred,labels).items():assert np.isclose(r[key],value)
    pd.DataFrame(classes).to_csv(out/'pooled_class_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,task in zip(axes,['multiclass','binary']):
        part=summary[summary.task==task]
        ax.bar(part.run_id,part.macro_f1_mean,yerr=part.macro_f1_std,capsize=4)
        ax.set(ylim=(0,1.05),ylabel='Macro-F1 mean +/- subject SD',title=task)
        ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    book=load_workbook(ROOT/'experiments/experiments_logs.xlsx',read_only=True,data_only=True)
    entries=list(book['Experiment Log'].iter_rows(min_row=2,values_only=True));book.close()
    for spec in cfg['runs']:
        for task in ['multiclass','binary']:
            matches=[r for r in entries if r[0]==f'EXP-008/{spec["run_id"]}/{task}']
            assert len(matches)==1
            r=summary[(summary.run_id==spec['run_id'])&(summary.task==task)].iloc[0]
            assert np.isclose(matches[0][8],r.macro_f1_mean) and np.isclose(matches[0][9],r.balanced_accuracy_mean)
    lines=['# EXP-008 — Sáu thực nghiệm SVM của Thái','',
        'Bổ sung theo yêu cầu ngày 22/11/2026. Giữ nguyên hồ sơ lịch sử của Nam. '
        'Sáu thực nghiệm là sáu cấu hình RBF cố định, không phải sáu thuật toán tìm kiếm độc lập. '
        'C = 0.1/1/10, gamma = 0.001/0.01, 68 features, không cân bằng lớp hoặc chọn đặc trưng.', '',
        'Mỗi run: 15 subject × hai task = 30 outer fold, 12.378 dự đoán OOF. '
        'Median imputation và scaling fit riêng trong mỗi inner/outer train. '
        'Mỗi outer subject chọn cấu hình từ sáu run bằng Macro-F1 trung bình của cùng ba inner group split. '
        'Điểm outer không được dùng để chọn cấu hình. Hòa điểm chọn run xuất hiện trước trong YAML.', '',
        '| Run | C | Gamma | Task | Macro-F1 ± SD | Balanced Accuracy ± SD |','|---|---|---|---|---|---|']
    for r in summary.itertuples():lines.append(f'| {r.run_id} | {r.C} | {r.gamma} | {r.task} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | {r.balanced_accuracy_mean:.4f} ± {r.balanced_accuracy_std:.4f} |')
    lines+=['','## Phân tích theo subject và lỗi','']
    for task,part in final.groupby('task'):
        delta=pairs.loc[pairs.task==task,'delta_f1_svm_minus_lr']
        worst=part.nsmallest(3,'macro_f1')
        lines.append(f'- {task}: Δ F1 SVM−LR={delta.mean():+.4f}; '
            f'{int((delta>1e-12).sum())} subject tăng, {int((delta < -1e-12).sum())} giảm. '
            f'Train-test gap trung bình {part.train_test_gap.mean():.4f}. Subject khó: '+
            ', '.join(f'{r.subject_id} ({r.macro_f1:.4f})' for r in worst.itertuples())+'.')
    binary=pred[pred.task=='binary']
    lines+=[f'- Binary: bỏ sót Stress {int(((binary.y_true==1)&(binary.y_pred==0)).sum())} cửa sổ; '
            f'báo động sai {int(((binary.y_true==0)&(binary.y_pred==1)).sum())} cửa sổ.', '',
        '![So sánh cấu hình](comparison.png)','','![Multiclass](confusion_multiclass.png)','','![Binary](confusion_binary.png)','',
        '## Kết luận và giới hạn','',
        'comparison.csv chứa đủ sáu cấu hình và quy trình nested_selection; paired_lr_comparison.csv '
        'chứa đủ 15 cặp subject mỗi task. Không chọn run có outer score cao nhất để báo cáo như một quy trình tuning. '
        'Các lớp trong pooled_class_metrics.csv gộp cửa sổ để phân tích lỗi, không thay mean ± SD qua subject. '
        'Không khẳng định ý nghĩa thống kê từ chênh lệch trung bình. Benchmark đã được xem nhiều lần, '
        'số người nhỏ, chưa có external holdout; chỉ kết luận thăm dò, chưa chọn mô hình thắng cả nhóm.', '',
        'Nguồn: configs/exp008_thai.yaml; experiments/02_tuning/EXP-008_SVM_Thai/runs/SVM-01…06; '
        'reports/lr_workflow_20261121/tuning_results.csv. Notebook hiển thị kết quả lưu từ runner; '
        'ô notebook không được ghi là đã thực thi qua Jupyter.', '',
        'Tái kiểm tra: `.\\.venv\\Scripts\\python.exe scripts/report_exp008_thai.py`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write_json(out/'verification.json',{'status':'passed','verified_at':datetime.now(timezone.utc).isoformat(),
        'runs':checks,'log_entries':12,'nested_folds':30,'nested_predictions':12378,
        'files':{p.name:file_hash(p) for p in out.iterdir() if p.name!='verification.json'}})
    print(summary.to_string(index=False));print('VERIFIED six runs, 180 folds, 74268 OOF predictions')


if __name__=='__main__':main()
