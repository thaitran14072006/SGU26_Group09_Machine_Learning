"""Phân tích EXP-015 so với EXP-010; chỉ đọc dự đoán, không fit MLP."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'experiments/02_tuning/EXP-010/results'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def data_check(run):
    current=read_json(run/'config.json');reference=read_json(REFERENCE/'config.json')
    assert current['data_sha256']==reference['data_sha256']
    assert current['feature_columns']==reference['feature_columns']
    assert current['environment']==reference['environment']
    assert read_json(run/'best_params.json')==read_json(REFERENCE/'best_params.json')
    data=ROOT/'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==current['data_sha256']
    frame=pd.read_csv(data,dtype={'subject_id':str,'window_id':str})
    assert frame.subject_id.nunique()==15 and len(frame)==6189
    print('ngày chạy:',read_json(run/'notebook_provenance.json')['date'])
    print('dữ liệu: 6.189 cửa sổ, 15 người, 68 feature')
    print('sha256:',current['data_sha256'])
    print('môi trường train:',current['environment'])
    print('best_params và môi trường khớp exp010; chỉ thêm RandomOverSampler trên train')
    return frame,current


def compare(run):
    assert read_json(run/'verification.json')['status']=='passed'
    plain=pd.read_csv(REFERENCE/'verified_metrics_by_subject.csv')
    balanced=pd.read_csv(run/'verified_metrics_by_subject.csv')
    paired=plain.merge(balanced,on=['task','model','subject_id'],validate='one_to_one',suffixes=('_plain','_balanced'))
    assert len(paired)==30 and paired.n_test_plain.eq(paired.n_test_balanced).all()
    rows=[]
    for task,part in paired.groupby('task'):
        for metric in ['macro_f1','balanced_accuracy']:
            delta=part[f'{metric}_balanced']-part[f'{metric}_plain']
            paired.loc[part.index,f'delta_{metric}']=delta
            rows.append(dict(task=task,metric=metric,n_subjects=len(part),
                plain_mean=float(part[f'{metric}_plain'].mean()),plain_std=float(part[f'{metric}_plain'].std(ddof=1)),
                balanced_mean=float(part[f'{metric}_balanced'].mean()),balanced_std=float(part[f'{metric}_balanced'].std(ddof=1)),
                mean_delta=float(delta.mean()),improved=int((delta>1e-12).sum()),
                worse=int((delta < -1e-12).sum()),equal=int((abs(delta)<=1e-12).sum())))
    summary=pd.DataFrame(rows)
    paired.to_csv(run/'plain_vs_balanced_by_subject.csv',index=False)
    summary.to_csv(run/'comparison_summary.csv',index=False)
    return paired,summary,balanced


def loss_traces(run):
    rows=[]
    for task in ['multiclass','binary']:
        for subject in read_json(run/'config.json')['settings']['expected_subjects']:
            detail=read_json(run/'folds'/task/'mlp'/subject/'details.json')
            rows.append(dict(task=task,subject_id=subject,**detail['mlp_optimization']))
    (run/'loss_traces.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    return rows


def figures(run,paired,audit,losses):
    folder=run/'figures';folder.mkdir(exist_ok=True)
    subjects=sorted(paired.subject_id.unique(),key=lambda s:int(s[1:]))
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        part=paired[paired.task==task].set_index('subject_id').loc[subjects]
        pd.DataFrame({'exp010 chưa cân bằng':part.macro_f1_plain,'exp015 cân bằng':part.macro_f1_balanced}).plot.bar(ax=ax,rot=0)
        ax.set(title=f'{task}: macro f1 theo người',xlabel='người làm test',ylabel='macro f1',ylim=(0,1.05))
        ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'plain_vs_balanced.png',dpi=130);plt.show()
    plain=pd.read_csv(REFERENCE/'predictions.csv');balanced=pd.read_csv(run/'predictions.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for i,task in enumerate(['multiclass','binary']):
        labels=[1,2,3] if task=='multiclass' else [0,1]
        names=['baseline','stress','amusement'] if task=='multiclass' else ['non-stress','stress']
        for ax,frame,title in zip(axes[i],[plain,balanced],['exp010 chưa cân bằng','exp015 cân bằng']):
            part=frame[frame.task==task]
            ConfusionMatrixDisplay.from_predictions(part.y_true,part.y_pred,labels=labels,
                display_labels=names,cmap='Blues',colorbar=False,ax=ax)
            ax.set_title(f'{task}: {title}');ax.tick_params(axis='x',rotation=15)
    fig.savefig(folder/'confusion_matrices.png',dpi=130);plt.show()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        row=audit[(audit.task==task)&(audit.subject_id=='S4')].iloc[0]
        before=json.loads(row.class_counts_before);after=json.loads(row.class_counts_after)
        pd.DataFrame({'trước ros':before,'sau ros':after}).plot.bar(ax=ax,rot=0)
        ax.set(title=f'{task}: train khi test s4',xlabel='nhãn',ylabel='số mẫu train')
    fig.savefig(folder/'sampler_class_counts.png',dpi=130);plt.show()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        for item in losses:
            if item['task']==task:ax.plot(range(1,len(item['loss_curve'])+1),item['loss_curve'],alpha=.6,label=item['subject_id'])
        ax.set(title=f'{task}: train loss sau ros',xlabel='epoch',ylabel='train loss');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,ncol=3)
    fig.savefig(folder/'balanced_train_loss.png',dpi=130);plt.show()


def conclusion(run,summary,metrics,audit):
    lines=['## nhận xét — '+read_json(run/'notebook_provenance.json')['date'],'']
    for row in summary.loc[summary.metric=='macro_f1'].itertuples():
        direction='tăng' if row.mean_delta>0 else 'giảm' if row.mean_delta<0 else 'không đổi'
        lines.append(f'- {row.task}: macro f1 cân bằng {row.balanced_mean:.4f} ± {row.balanced_std:.4f}; '
            f'exp010 {row.plain_mean:.4f} ± {row.plain_std:.4f}; {direction} {abs(row.mean_delta):.4f}. '
            f'{row.improved} người tăng, {row.worse} người giảm, {row.equal} người bằng nhau.')
    lines+=['',f'giữ đúng best_params của exp010 cho cả 30 outer folds; không tìm tham số lại.',
        f"{int(audit.passed.sum())}/30 kiểm tra sampler đạt; số mẫu test trước và sau bằng nhau.",
        f"có {int(metrics.warning_count.gt(0).sum())}/30 lượt fit ghi cảnh báo; "
        f"{int(metrics.reached_max_iter.sum())}/30 lượt chạm max_iter=200.",
        '', 'cân bằng lớp không đảm bảo tăng macro f1 trên người mới. tham số ở đây được chọn cho '
        'pipeline chưa cân bằng; chưa kết luận đó là tham số tốt nhất cho pipeline có ros.',
        '', 'loss được tính trên train sau oversampling, không phải loss test. phân phối và số mẫu train '
        'đã đổi nên không diễn giải chênh lệch loss tuyệt đối với exp010 như bằng chứng cải thiện.',
        '', 'audit dựng lại preprocessing và sampler với đúng train và seed, không fit thêm mlp. '
        'runner chưa xuất sample_indices_ từ chính pipeline đã fit; đó là phần có thể bổ sung sau.',
        '', 'so sánh theo cặp cùng dữ liệu, subject, tham số và môi trường. mean/sd mẫu theo 15 người. '
        'chưa kiểm định thống kê, chưa khảo sát seed khác hoặc cách resampling khác. '
        'nếu khảo sát thêm cần chọn bằng inner cv, không tối ưu theo outer test.',
        '', 'exp020 sẽ nghiên cứu kiến trúc và regularization với lưới khác exp010. giữ nguyên các kết quả hiện tại.']
    text='\n'.join(lines)
    (run/'conclusion.md').write_text(text,encoding='utf8');print(text)
    return text


def package(run,notebook):
    destination=ROOT/'experiments/03_balancing/EXP-015'
    assert not destination.exists(),'không tự ghi đè exp015 đã chốt'
    results=destination/'results';results.mkdir(parents=True)
    for name in ['config.json','config.yaml','environment.json','notebook_provenance.json','run.log',
        'summary.json','metrics_by_subject.csv','verified_metrics_by_subject.csv','predictions.csv',
        'best_params.json','splits.json','verification.json','sampler_audit.csv','loss_traces.json',
        'plain_vs_balanced_by_subject.csv','comparison_summary.csv','conclusion.md','source_snapshot.zip']:
        shutil.copy2(run/name,results/name)
    shutil.copytree(run/'folds',results/'folds');shutil.copytree(run/'figures',results/'figures')
    shutil.copy2(notebook,destination/Path(notebook).name)
    manifest={p.relative_to(results).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in results.rglob('*') if p.is_file()}
    (results/'artifact_checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('đã đóng gói:',destination)
    return destination
