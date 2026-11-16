"""Đọc kết quả EXP-010; không huấn luyện, không sửa kết quả EXP-006."""
from pathlib import Path
import hashlib
import json
import shutil
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/01_baselines/EXP-006/results'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def data_check(run):
    config=read_json(run/'config.json')
    provenance=read_json(run/'notebook_provenance.json')
    baseline_info=read_json(BASE/'run_info.json')
    data=ROOT/'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    assert digest==config['data_sha256']==baseline_info['data_sha256']
    frame=pd.read_csv(data,dtype={'subject_id':str,'window_id':str})
    assert len(frame)==6189 and frame.subject_id.nunique()==15
    assert len(config['feature_columns'])==68 and config['feature_columns']==baseline_info['features']
    assert (frame.label_original.isin([1,2,3])).all()
    print('ngày chạy:',provenance['date'],'| loại:',provenance['run_type'])
    print('dữ liệu:',len(frame),'cửa sổ,',frame.subject_id.nunique(),'người, 68 feature')
    print('sha256:',digest)
    print('môi trường train exp010:',config['environment'])
    print('baseline exp006:',{k:baseline_info[k] for k in ['python','sklearn','numpy','pandas']})
    print('không nhận là cùng toàn bộ môi trường: pandas exp006 và exp010 khác phiên bản')
    return frame,config,baseline_info


def compare(run):
    verified=read_json(run/'verification.json')
    assert verified['status']=='passed' and verified['outer_folds']==30
    current=pd.read_csv(run/'verified_metrics_by_subject.csv')
    baseline=pd.read_csv(BASE/'metrics_by_subject.csv').rename(columns={'test_subject':'subject_id'})
    assert len(current)==len(baseline)==30
    joined=current.merge(baseline,on=['task','subject_id'],validate='one_to_one',suffixes=('_tuned','_default'))
    assert (joined.n_test_tuned==joined.n_test_default).all()
    # Đối chiếu baseline từ các dự đoán đã lưu, không chỉ tin bảng điểm.
    for row in baseline.itertuples():
        pred=pd.read_csv(BASE/f'predictions_{row.task}_{row.subject_id}.csv')
        labels=[1,2,3] if row.task=='multiclass' else [0,1]
        assert np.isclose(f1_score(pred.y_true,pred.y_pred,labels=labels,average='macro',zero_division=0),row.macro_f1)
        assert np.isclose(balanced_accuracy_score(pred.y_true,pred.y_pred),row.balanced_accuracy)
    for metric in ['macro_f1','balanced_accuracy']:
        joined[f'delta_{metric}']=joined[f'{metric}_tuned']-joined[f'{metric}_default']
    joined.to_csv(run/'default_vs_tuned_by_subject.csv',index=False)
    rows=[]
    for task,part in joined.groupby('task'):
        for metric in ['macro_f1','balanced_accuracy']:
            delta=part[f'delta_{metric}']
            rows.append(dict(task=task,metric=metric,n_subjects=len(part),
                default_mean=float(part[f'{metric}_default'].mean()),
                default_std=float(part[f'{metric}_default'].std(ddof=1)),
                tuned_mean=float(part[f'{metric}_tuned'].mean()),
                tuned_std=float(part[f'{metric}_tuned'].std(ddof=1)),
                mean_delta=float(delta.mean()),improved=int((delta>1e-12).sum()),
                worse=int((delta < -1e-12).sum()),equal=int((abs(delta)<=1e-12).sum())))
    summary=pd.DataFrame(rows)
    summary.to_csv(run/'comparison_summary.csv',index=False)
    return joined,summary


def details(run):
    records=[];inners=[];losses=[]
    config=read_json(run/'config.json')['settings']
    declared=config['search_spaces']['mlp']
    candidates=[{k:v[0] for k,v in c.items()} for c in declared]
    for task in ['multiclass','binary']:
        for subject in config['expected_subjects']:
            folder=run/'folds'/task/'mlp'/subject
            detail=read_json(folder/'details.json')
            best=detail['best_params']
            candidate=next(i+1 for i,c in enumerate(candidates) if c==best)
            trace=detail['mlp_optimization']
            records.append(dict(task=task,subject_id=subject,candidate=f'c{candidate}',
                hidden_sizes=str(best['model__hidden_layer_sizes']),alpha=best['model__alpha'],
                learning_rate=best['model__learning_rate_init'],n_iter=trace['n_iter'],
                max_iter=trace['max_iter'],final_train_loss=trace['loss'],
                reached_max_iter=trace['reached_max_iter'],warning_count=len(detail['warnings']),
                warnings=' | '.join(detail['warnings'])))
            inner=pd.read_csv(folder/'inner_cv.csv')
            inner.insert(0,'candidate',[f'c{i+1}' for i in range(4)])
            inner.insert(0,'subject_id',subject);inner.insert(0,'task',task)
            inners.append(inner)
            losses.append(dict(task=task,subject_id=subject,loss_curve=trace['loss_curve']))
    convergence=pd.DataFrame(records)
    convergence.to_csv(run/'convergence.csv',index=False)
    pd.concat(inners,ignore_index=True).to_csv(run/'inner_search_all.csv',index=False)
    save_json(run/'loss_traces.json',losses)
    return convergence,losses


def figures(run,joined,convergence,losses):
    folder=run/'figures';folder.mkdir(exist_ok=True)
    subjects=sorted(joined.subject_id.unique(),key=lambda s:int(s[1:]))
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        part=joined[joined.task==task].set_index('subject_id').loc[subjects]
        pd.DataFrame({'exp006 mặc định':part.macro_f1_default,'exp010 tuning':part.macro_f1_tuned}).plot.bar(ax=ax,rot=0)
        ax.set(title=f'{task}: macro f1 từng người',xlabel='người làm outer test',ylabel='macro f1',ylim=(0,1.05))
        ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'default_vs_tuned.png',dpi=130);plt.show()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        for trace in losses:
            if trace['task']==task:ax.plot(range(1,len(trace['loss_curve'])+1),trace['loss_curve'],label=trace['subject_id'],alpha=.7)
        ax.set(title=f'{task}: train loss của outer refit',xlabel='epoch',ylabel='train loss')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,ncol=3)
    fig.savefig(folder/'outer_refit_loss.png',dpi=130);plt.show()
    predictions=pd.read_csv(run/'predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,task in zip(axes,['multiclass','binary']):
        part=predictions[predictions.task==task]
        labels=[1,2,3] if task=='multiclass' else [0,1]
        names=['baseline','stress','amusement'] if task=='multiclass' else ['non-stress','stress']
        ConfusionMatrixDisplay.from_predictions(part.y_true,part.y_pred,labels=labels,
            display_labels=names,cmap='Blues',colorbar=False,ax=ax)
        ax.set_title(f'{task}: gộp dự đoán 15 người');ax.tick_params(axis='x',rotation=15)
    fig.savefig(folder/'confusion_matrices.png',dpi=130);plt.show()
    frequency=pd.crosstab(convergence.candidate,convergence.task).reindex(['c1','c2','c3','c4'],fill_value=0)
    frequency.to_csv(run/'candidate_frequency.csv')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    frequency.plot.bar(ax=ax,rot=0)
    ax.set(title='ứng viên được chọn bằng inner cv',xlabel='ứng viên',ylabel='số outer folds')
    ax.set_yticks(range(0,16,3));ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'candidate_frequency.png',dpi=130);plt.show()


def conclusion(run,summary,convergence):
    lines=['## nhận xét kết quả — '+read_json(run/'notebook_provenance.json')['date'],'']
    for row in summary.loc[summary.metric=='macro_f1'].itertuples():
        direction='tăng' if row.mean_delta>0 else 'giảm' if row.mean_delta<0 else 'không đổi'
        lines.append(f'- {row.task}: macro f1 tuning {row.tuned_mean:.4f} ± {row.tuned_std:.4f}; '
                     f'baseline {row.default_mean:.4f} ± {row.default_std:.4f}; '
                     f'{direction} {abs(row.mean_delta):.4f}. '
                     f'{row.improved} người tăng, {row.worse} người giảm, {row.equal} người bằng nhau.')
    lines+=['',f"có {int(convergence.warning_count.gt(0).sum())}/30 outer folds ghi cảnh báo trong quá trình search/refit. "
            f"{int(convergence.reached_max_iter.sum())}/30 mô hình refit chạm max_iter=200.",
            f"số epoch refit từ {convergence.n_iter.min()} đến {convergence.n_iter.max()}.",
            '', 'train loss thuộc mô hình tốt nhất refit trên outer train, không phải validation loss. '
            'cảnh báo được runner gom cho cả search và refit, chưa gán riêng từng ứng viên.',
            '', 'tuning không đảm bảo cải thiện trên người mới. kết luận chỉ áp dụng cho bốn cấu hình '
            'đã khai báo; các cấu hình thay nhiều tham số cùng lúc nên chưa tách được ảnh hưởng riêng của alpha hay kiến trúc.',
            '', 'so sánh dùng cùng dữ liệu và subject, sklearn 1.9.1. baseline dùng pandas 3.1.1, '
            'lượt tuning dùng pandas 3.1.0; chưa kiểm tra độ nhạy với khác biệt này. '
            'đây là so sánh mô tả, chưa kiểm định ý nghĩa thống kê.',
            '', 'exp015 sẽ khảo sát cân bằng lớp train-only với tham số đã chọn. '
            'exp020 sẽ khảo sát kiến trúc và regularization khác lưới hiện tại. '
            'không sửa grid theo outer test để làm điểm đẹp hơn.']
    text='\n'.join(lines)
    (run/'conclusion.md').write_text(text,encoding='utf8')
    print(text)
    return text


def package(run,notebook):
    destination=ROOT/'experiments/02_tuning/EXP-010'
    assert not destination.exists(),'bản exp010 đã có; không ghi đè tự động'
    target=destination/'results';target.mkdir(parents=True)
    names=['config.json','config.yaml','environment.json','notebook_provenance.json',
        'summary.json','metrics_by_subject.csv','verified_metrics_by_subject.csv','predictions.csv',
        'best_params.json','splits.json','verification.json','comparison_summary.csv',
        'default_vs_tuned_by_subject.csv','convergence.csv','loss_traces.json',
        'inner_search_all.csv','candidate_frequency.csv','conclusion.md','source_snapshot.zip','run.log']
    for name in names:shutil.copy2(run/name,target/name)
    shutil.copytree(run/'figures',target/'figures')
    shutil.copytree(run/'folds',target/'folds')
    shutil.copy2(notebook,destination/Path(notebook).name)
    # Lưu checkpoint riêng để resume, bản nộp có bảng inner đủ 120 ứng viên.
    manifest={p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in target.rglob('*') if p.is_file()}
    save_json(target/'artifact_checksums.json',manifest)
    print('đã đóng gói:',destination)
    return destination
