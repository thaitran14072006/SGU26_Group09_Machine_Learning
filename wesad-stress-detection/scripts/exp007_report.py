"""Tạo CSV/PNG của EXP-007 từ output chính của runner WESAD, không fit lại mô hình."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = {f'S{i}' for i in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]}


def export_results(run_dir: Path, export_dir: Path, baseline_dir: Path | None = None):
    run_dir, export_dir = Path(run_dir), Path(export_dir)
    manifest = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('Kết quả chưa complete; không xuất báo cáo main từ run chưa đủ.')
    scores = pd.read_csv(run_dir / 'metrics_by_subject.csv', dtype={'subject_id': str})
    preds = pd.read_csv(run_dir / 'predictions.csv', dtype={'subject_id': str, 'window_id': str})
    if (len(scores) != 30 or set(scores.subject_id) != SUBJECTS or
        set(scores.task) != {'binary','multiclass'} or set(scores.model) != {'knn'} or
        len(scores.groupby(['task','subject_id'])) != 30):
        raise ValueError('Không đủ 30 fold KNN / 15 người / 2 nhiệm vụ.')
    if set(preds.model) != {'knn'} or set(preds.task) != {'binary','multiclass'} or preds.duplicated(['task','subject_id','window_id']).any():
        raise ValueError('Predictions sai bài toán hoặc có hàng trùng.')
    for task in ['binary','multiclass']:
        assert len(preds[preds.task == task]) == int(scores[scores.task == task].n_test.sum()), 'Thiếu dự đoán ngoài mẫu'
    settings = manifest.get('settings', {})
    if not settings.get('nested', False) or settings.get('models') != ['knn'] or settings.get('balance'):
        raise ValueError('Run không phải EXP-007 KNN nested tuning không cân bằng.')
    export_dir.mkdir(parents=True, exist_ok=True)
    figs=export_dir/'figures';figs.mkdir(exist_ok=True)
    scores.sort_values(['task','subject_id'],key=lambda s:s.map(lambda v:int(v[1:])) if s.name=='subject_id' else s).to_csv(export_dir/'results.csv',index=False)
    preds.to_csv(export_dir/'predictions.csv',index=False)
    pd.DataFrame(summary['results']).to_csv(export_dir/'summary.csv',index=False)
    params=json.loads((run_dir/'best_params.json').read_text(encoding='utf-8'))
    param_rows=[]
    for key, values in sorted(params.items()):
        task, model, subject = key.split('/')
        if model!='knn': continue
        param_rows.append({'task':task,'subject_id':subject,'n_neighbors':values.get('model__n_neighbors'),
                           'metric':values.get('model__metric'),'weights':values.get('model__weights')})
    if len(param_rows)!=30: raise ValueError('Không có đủ 30 bộ tham số được chọn.')
    pd.DataFrame(param_rows).to_csv(export_dir/'best_params_by_subject.csv',index=False)
    for task, labels in [('binary',[0,1]),('multiclass',[1,2,3])]:
        part=preds.loc[preds.task==task]
        cm=confusion_matrix(part.y_true,part.y_pred,labels=labels)
        fig,ax=plt.subplots(figsize=(5.4,4.6))
        ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
        names=['Non-stress','Stress'] if task=='binary' else ['Baseline','Stress','Amusement']
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=names,yticklabels=names,
               xlabel='Nhãn dự đoán',ylabel='Nhãn thật',title=f'EXP-007 - Confusion Matrix - {task}')
        fig.tight_layout();fig.savefig(figs/f'confusion_matrix_{task}.png',dpi=160);plt.close(fig)
        group=scores[scores.task==task].copy()
        group['order']=group.subject_id.str[1:].astype(int)
        group=group.sort_values('order')
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.bar(group.subject_id,group.macro_f1)
        ax.set(ylim=(0,1),xlabel='Người tham gia',ylabel='Macro-F1',title=f'EXP-007 - Macro-F1 - {task}')
        fig.tight_layout();fig.savefig(figs/f'macro_f1_by_subject_{task}.png',dpi=160);plt.close(fig)
    comparison_note='Chưa có baseline EXP-003 tương thích để so sánh; không tạo comparison giả.'
    if baseline_dir and (Path(baseline_dir)/'metrics_by_subject.csv').exists():
        base_dir=Path(baseline_dir)
        base_manifest=json.loads((base_dir/'config.json').read_text(encoding='utf-8'))
        if (base_manifest.get('data_sha256') == manifest.get('data_sha256') and
            base_manifest.get('feature_columns') == manifest.get('feature_columns') and
            not base_manifest.get('settings',{}).get('nested',False) and
            not base_manifest.get('settings',{}).get('balance',False)):
            base=pd.read_csv(base_dir/'metrics_by_subject.csv',dtype={'subject_id':str})
            base=base[base.model=='knn']
            joined=scores.merge(base,on=['task','subject_id'],suffixes=('_tuned','_default'),validate='one_to_one')
            if len(joined)==30:
                for metric in ['macro_f1','balanced_accuracy']:
                    joined[f'delta_{metric}']=joined[f'{metric}_tuned']-joined[f'{metric}_default']
                joined.to_csv(export_dir/'comparison_exp003.csv',index=False)
                fig,ax=plt.subplots(figsize=(8,4.5))
                means=joined.groupby('task')[["macro_f1_default","macro_f1_tuned"]].mean()
                means.plot(kind='bar',ax=ax,rot=0)
                ax.set(ylim=(0,1),ylabel='Macro-F1 trung bình LOSO',title='EXP-003 mặc định và EXP-007 tối ưu')
                fig.tight_layout();fig.savefig(figs/'comparison_exp003.png',dpi=160);plt.close(fig)
                comparison_note='So sánh EXP-003 và EXP-007 cùng hash dữ liệu và feature schema; xem comparison_exp003.csv.'
    provenance={'run_dir':str(run_dir),'data_sha256':manifest.get('data_sha256'),
                'settings':settings,'scope':'main complete','comparison':comparison_note,
                'note':'Macro-F1 là mean/std across 15 LOSO subjects; không coi cửa sổ chồng lấn là độc lập.'}
    (export_dir/'result_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Đã xuất báo cáo tại',export_dir,'\n',comparison_note)
    return export_dir

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default='outputs/notebook_research/EXP-007/r03_main')
    p.add_argument('--export-dir',default='experiments/02_tuning/EXP-007_KNN')
    p.add_argument('--baseline-dir',default='outputs/notebook_research/EXP-003/r02_main')
    a=p.parse_args()
    def resolve(v):
        path=Path(v)
        return path if path.is_absolute() else ROOT/path
    export_results(resolve(a.run_dir),resolve(a.export_dir),resolve(a.baseline_dir))
