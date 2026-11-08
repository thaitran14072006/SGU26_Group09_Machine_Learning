"""Render original report figures from verified G3 metrics; never fit a model."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
MODELS=['dummy','lr','knn','svm','rf','mlp']
NAMES=['Dummy','LR','KNN','SVM','RF','MLP']
SUBJECTS={f'S{i}' for i in [*range(2,12),*range(13,18)]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(path.with_suffix('.svg'),facecolor='white')
    plt.close(fig)


def pipeline(path):
    fig,ax=plt.subplots(figsize=(15,9))
    ax.set_xlim(0,15);ax.set_ylim(0,9);ax.axis('off')
    def box(x,y,w,h,text,color='#eaf3fa',size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',fc=color,ec='#35566f',lw=1.3))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.5)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#35566f',lw=1.5))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,label,ha='center',fontsize=9)
    ax.text(7.5,8.65,'Pipeline WESAD: tiền xử lý, đặc trưng và đánh giá LOSO',ha='center',fontsize=18,weight='bold')
    box(.2,6.9,2.4,1.1,'WESAD raw\n15 người\nNgực và cổ tay')
    box(3.1,6.9,3.0,1.1,'Lọc / đồng bộ / đổi đơn vị\nGiữ thời gian và tần số gốc\nProvider thực hiện một lần')
    box(6.6,6.9,3.4,1.1,'Cửa sổ 60 s, bước 5 s\nNhãn 1/2/3 đa số ≥90%\nGiữ nhãn khác trong mẫu số')
    box(10.5,6.9,4.1,1.1,'Trích xuất sinh lý 68\nACC/EMG: trung bình 12 đoạn 5 s\nNaN có lý do, Inf là lỗi')
    arrow((2.67,7.45),(3.03,7.45));arrow((6.17,7.45),(6.53,7.45));arrow((10.07,7.45),(10.43,7.45))
    box(10.5,5.25,4.1,1.0,'Bảng G2: 6.189 cửa sổ\n68 feature + 8 metadata\nKhông bỏ hàng riêng theo cảm biến',color='#edf5e8')
    arrow((12.55,6.83),(12.55,6.32))
    box(.2,5.25,9.6,1.0,'LOSO 15 lượt: mỗi lượt giữ 1 subject test, 14 subject train\nX chỉ gồm feature; subject, thời gian, nhãn và quality flag không nằm trong X',color='#fff3d6',size=12)
    arrow((10.43,5.75),(9.87,5.75))
    box(.5,3.1,6.0,1.25,'TRAIN — 14 subject\nFit median imputer trên train\nFit StandardScaler: LR / KNN / SVM / MLP',color='#e7f4eb',size=12)
    box(8.0,3.1,6.0,1.25,'TEST — 1 subject\nChỉ transform bằng imputer/scaler đã fit\nKhông fit hoặc chọn tham số trên test',color='#fcebe8',size=12)
    arrow((3.5,5.18),(3.5,4.42));arrow((11,5.18),(11,4.42));arrow((6.57,3.73),(7.93,3.73),'Bộ biến đổi từ train')
    box(.5,1.35,6.0,1.0,'Fit mô hình mặc định trên train\nDummy / LR / KNN / SVM / RF / MLP\nKhông tuning, cân bằng lớp hoặc selection',color='#e7f4eb')
    box(8.0,1.35,6.0,1.0,'Dự đoán ngoài mẫu trên test\n180 fold, 74.268 dự đoán\nBa lớp [1,2,3]; nhị phân [0,1]',color='#fcebe8')
    arrow((3.5,3.03),(3.5,2.42));arrow((11,3.03),(11,2.42));arrow((6.57,1.85),(7.93,1.85),'Mô hình đã fit')
    ax.text(7.5,.63,'Metric từng người → trung bình ± SD mẫu qua 15 người; confusion matrix từ dự đoán ngoài mẫu',ha='center',fontsize=12,weight='bold')
    arrow((11,1.28),(11,.88))
    ax.text(7.5,.15,'Dummy và RF không dùng StandardScaler trong pipeline hiện tại. NaN được xử lý trong train fold; không impute/scaler trên toàn bảng.',ha='center',fontsize=10,color='#435869')
    fig.tight_layout()
    export(fig,path)


def main():
    metrics=ROOT/'reports/g3_group/metrics_by_subject.csv'
    summary_path=ROOT/'reports/g3_group/summary.json'
    with metrics.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=180:raise ValueError('Expected 180 LOSO metric rows')
    keys={(r['task'],r['model'],r['subject_id']) for r in rows}
    if len(keys)!=180:raise ValueError('Duplicate metric keys')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    results={}
    for task in ['multiclass','binary']:
        for model in MODELS:
            group=[r for r in rows if r['task']==task and r['model']==model]
            if len(group)!=15 or {r['subject_id'] for r in group}!=SUBJECTS:raise ValueError('Incomplete model/task subjects')
            reference=next(r for r in summary['results'] if r['task']==task and r['model']==model)
            results[task,model]={}
            for metric in ['macro_f1','balanced_accuracy']:
                values=[float(r[metric]) for r in group]
                if any(not 0<=v<=1 for v in values):raise ValueError('Metric outside [0,1]')
                mean,sd=statistics.mean(values),statistics.stdev(values)
                if abs(mean-reference[metric+'_mean'])>1e-12 or abs(sd-reference[metric+'_std'])>1e-12:raise ValueError('CSV and summary differ')
                results[task,model][metric]=(mean,sd)
    out=ROOT/'docs/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    pipeline(out/'wesad_pipeline')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors=['#8d99ae','#3274a1','#e1812c','#3a923a','#c03d3e','#9372b2']
    for i,task in enumerate(['multiclass','binary']):
        for j,metric in enumerate(['macro_f1','balanced_accuracy']):
            ax=axes[i,j];values=[results[task,m][metric][0] for m in MODELS];sd=[results[task,m][metric][1] for m in MODELS]
            ax.bar(NAMES,values,yerr=sd,capsize=4,color=colors,width=.66)
            ax.set_ylim(0,max(1.03,max(a+b for a,b in zip(values,sd))+.04))
            for k,value in enumerate(values):ax.text(k,.02,f'{value:.3f}',ha='center',fontsize=10,color='black',bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
            ax.set_title(('Ba lớp' if task=='multiclass' else 'Nhị phân')+' — '+('Macro-F1' if metric=='macro_f1' else 'Balanced Accuracy'))
            ax.set_ylabel('Điểm (trung bình ± SD mẫu)');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.22)
    fig.suptitle('So sánh sáu mô hình mặc định — 68 feature sinh lý, LOSO 15 người',fontsize=15)
    export(fig,out/'g3_six_models_comparison')
    manifest={'scope':'figures_only_no_training','metric_source':str(metrics.relative_to(ROOT)),
       'metric_source_sha256':digest(metrics),'summary_source_sha256':digest(summary_path),
       'statistics':'Equal subject weight; sample SD ddof=1, not standard error or confidence interval.',
       'n_models':6,'n_tasks':2,'n_subjects':15,'n_folds':180,
       'renderer':{'python':sys.version,'matplotlib':matplotlib.__version__,'platform':platform.platform()},
       'files':{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(out.glob('*')) if p.stem in ['wesad_pipeline','g3_six_models_comparison']}}
    thai=ROOT/'reports/g3_group/thai_common_merge_verification.json'
    if thai.exists():manifest['later_common_protocol_review_sha256']=digest(thai)
    dest=ROOT/'reports/report_assets';dest.mkdir(parents=True,exist_ok=True)
    (dest/'figures_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Rendered pipeline and 6-model/2-task comparison; source statistics verified.')


if __name__=='__main__':main()
