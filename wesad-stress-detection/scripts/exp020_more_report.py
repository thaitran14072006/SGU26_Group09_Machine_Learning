"""Tổng hợp seed 42/43/44 và thêm output thật vào notebook EXP020."""
from pathlib import Path
import base64
import contextlib
import io
import json
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/04_additional/EXP-020'
OUT = PACKAGE / 'seed_sensitivity/20261122'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    from wesad.utils import read_json, file_hash, write_json
    runs = {42: PACKAGE/'results', 43: PACKAGE/'seed_sensitivity/20261120/seed_43', 44: PACKAGE/'seed_sensitivity/20261120/seed_44', 45: OUT/'seed_45', 46: OUT/'seed_46'}
    reference_manifest=read_json(runs[42]/'config.json')
    frames, warnings = [], []
    for seed, run in runs.items():
        assert read_json(run/'verification.json')['status'] == 'passed'
        manifest=read_json(run/'config.json')
        cfg = manifest['settings']
        assert cfg['seed'] == seed and cfg['nested'] and not cfg['balance']
        assert manifest['data_sha256']==reference_manifest['data_sha256']
        assert manifest['environment']==reference_manifest['environment']
        for key,value in reference_manifest['settings'].items():
            if key not in {'seed','source_paths'}:assert cfg[key]==value,(seed,key)
        assert read_json(run/'splits.json')==read_json(runs[42]/'splits.json')
        table = pd.read_csv(run/'verified_metrics_by_subject.csv')
        assert len(table)==30 and not table.duplicated(['task','subject_id']).any()
        frames.append(table.assign(seed=seed))
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(OUT/'scores_by_subject_seed.csv',index=False)
    summary = all_scores.groupby(['task','seed']).agg(
        subjects=('subject_id','nunique'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_sd_subjects=('macro_f1','std'),ba_mean=('balanced_accuracy','mean'),
        ba_sd_subjects=('balanced_accuracy','std'),warnings=('warning_count','sum'),
        max_epoch_folds=('reached_max_iter','sum'),epochs_mean=('n_iter','mean')).reset_index()
    assert summary.subjects.eq(15).all()
    summary.to_csv(OUT/'summary_by_seed.csv',index=False)
    variability = summary.groupby('task').agg(
        seed_mean_f1=('macro_f1_mean','mean'),seed_sd_f1=('macro_f1_mean','std'),
        min_seed_mean_f1=('macro_f1_mean','min'),max_seed_mean_f1=('macro_f1_mean','max'))
    variability.to_csv(OUT/'variability_across_seeds.csv')
    comparisons=[]
    p42=pd.read_csv(runs[42]/'predictions.csv')
    for seed in [43,44,45,46]:
        other=pd.read_csv(runs[seed]/'predictions.csv')
        pair=p42.merge(other,on=['task','model','subject_id','window_id'],suffixes=('_42','_new'),validate='one_to_one')
        assert len(pair)==12378 and pair.y_true_42.eq(pair.y_true_new).all()
        for task,g in pair.groupby('task'):
            comparisons.append(dict(task=task,seed=seed,n_windows=len(g),prediction_agreement=float(g.y_pred_42.eq(g.y_pred_new).mean())))
    pd.DataFrame(comparisons).to_csv(OUT/'prediction_agreement.csv',index=False)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,task,title in zip(axes,['multiclass','binary'],['Ba lớp','Nhị phân']):
        g=summary[summary.task==task].sort_values('seed')
        ax.bar(g.seed.astype(str),g.macro_f1_mean,yerr=g.macro_f1_sd_subjects,capsize=5,color=['#68879b','#bf815b','#7a936d','#997bac','#baaa55'])
        ax.set(title=title,xlabel='Seed',ylim=(0,1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(g.macro_f1_mean):ax.text(i,v+.02,f'{v:.4f}',ha='center')
    axes[0].set_ylabel('Macro-F1 trung bình theo 15 người')
    fig.suptitle('EXP020: độ nhạy theo seed — thanh sai số là SD giữa người')
    fig.tight_layout();fig.savefig(figures/'macro_f1_by_seed.png',dpi=180);plt.close(fig)
    subjects=sorted(all_scores.subject_id.unique(),key=lambda s:int(s[1:]))
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,sharey=True)
    for ax,task,title in zip(axes,['multiclass','binary'],['Ba lớp','Nhị phân']):
        for seed in [42,43,44,45,46]:
            g=all_scores[(all_scores.task==task)&(all_scores.seed==seed)].set_index('subject_id').loc[subjects]
            ax.plot(subjects,g.macro_f1,marker='o',label=f'seed {seed}')
        ax.set(title=title,ylabel='Macro-F1',ylim=(0,1.03));ax.grid(alpha=.2);ax.legend(ncol=5)
    fig.tight_layout();fig.savefig(figures/'subject_scores_by_seed.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,seed in enumerate([45,46]):
        for col,task in enumerate(['multiclass','binary']):
            ax=axes[row,col]
            for subject in ['S4','S9','S11','S17']:
                detail=read_json(runs[seed]/'folds'/task/'mlp'/subject/'details.json')
                loss=detail['mlp_optimization']['loss_curve']
                ax.plot(range(1,len(loss)+1),loss,label=subject)
            ax.set(title=f'seed {seed} - {task}',xlabel='epoch',ylabel='train objective')
            ax.grid(alpha=.2);ax.legend(ncol=2)
    fig.suptitle('Loss train của các fold S4/S9/S11/S17; không phải loss test')
    fig.tight_layout();fig.savefig(figures/'train_loss_seed45_46.png',dpi=180);plt.close(fig)
    # Execute the added cells and capture real stdout and generated figures.
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=s))
    env={'ROOT':ROOT,'OUT':OUT}
    def code(s):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(s,'exp020_seed_notebook','exec'),env)
        outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else []
        for num in plt.get_fignums():
            data=io.BytesIO();plt.figure(num).savefig(data,format='png',dpi=180,bbox_inches='tight')
            outputs.append(dict(output_type='display_data',data={'image/png':base64.b64encode(data.getvalue()).decode()},metadata={}))
        plt.close('all')
        count=sum(c['cell_type']=='code' for c in cells)+1
        cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=s,execution_count=count,outputs=outputs))
    md('## bổ sung ngày 22/11/2026 — kiểm tra độ nhạy theo seed\n\nngười thực hiện: như. EXP chung: EXP-020. Câu hỏi: kết quả thay đổi thế nào khi đổi seed khởi tạo MLP? Giữ nguyên dữ liệu, lưới bốn ứng viên, outer LOSO, ba inner GroupKFold, max_iter 200 và không ROS. Seed 42 dùng kết quả chính đã có; seed 45 và 46 được chạy nested CV mới, mỗi seed đủ 30 outer fold và 360 inner fits. Seed 45/46 được chọn trước lượt bổ sung này, sau khi đã xem kết quả seed 42/43/44. Không chọn seed có điểm cao nhất làm kết quả chính.')
    code("from pathlib import Path\nimport pandas as pd, json, sys\nroot = next(p for p in [Path.cwd(), Path.cwd()/'wesad-stress-detection', *Path.cwd().parents] if (p/'src/wesad').is_dir())\nfolder=root/'experiments/04_additional/EXP-020/seed_sensitivity/20261122'\nprint('ngày tổng hợp: 22/11/2026')\nprint('python:',sys.executable)\nsummary=pd.read_csv(folder/'summary_by_seed.csv')\nprint(summary.to_string(index=False))")
    md('Thanh sai số dưới đây là độ lệch chuẩn giữa 15 người tại từng seed. Đây không phải độ lệch chuẩn giữa các seed và không phải khoảng tin cậy.')
    code("import matplotlib.pyplot as plt\nsummary=pd.read_csv(folder/'summary_by_seed.csv')\nfig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)\nfor ax,task in zip(axes,['multiclass','binary']):\n    g=summary[summary.task==task].sort_values('seed')\n    ax.bar(g.seed.astype(str),g.macro_f1_mean,yerr=g.macro_f1_sd_subjects,capsize=5)\n    ax.set(title=task,xlabel='seed',ylabel='macro f1',ylim=(0,1))\nfig.tight_layout()")
    code("scores=pd.read_csv(folder/'scores_by_subject_seed.csv')\nprint(pd.read_csv(folder/'variability_across_seeds.csv').to_string(index=False))\nprint(pd.read_csv(folder/'prediction_agreement.csv').to_string(index=False))\nprint('số outer fold mới:',len(scores[scores.seed.isin([45,46])]))\nfig=plt.figure(figsize=(12,8))\nplt.imshow(plt.imread(folder/'figures/subject_scores_by_seed.png'))\nplt.axis('off')\nfig.tight_layout()")
    md('Loss dưới đây là mục tiêu train có regularization của lượt refit được chọn bởi inner CV. Chỉ dùng để xem diễn biến tối ưu, không so trực tiếp độ lớn giữa các fold có alpha khác nhau. Không có loss test theo epoch.')
    code("fig=plt.figure(figsize=(12,8))\nplt.imshow(plt.imread(folder/'figures/train_loss_seed45_46.png'))\nplt.axis('off')\nfig.tight_layout()")
    lines=[]
    for task in ['multiclass','binary']:
        g=summary[summary.task==task].sort_values('seed')
        lines.append(f"{task}: Macro-F1 của seed 42/43/44/45/46 lần lượt là "+' / '.join(f'{v:.4f}' for v in g.macro_f1_mean)+f". Khoảng biến động trung bình theo seed là {g.macro_f1_mean.max()-g.macro_f1_mean.min():.4f}.")
    md('### nhận xét\n\n'+'\n\n'.join(lines)+'\n\nĐiểm theo seed được tính từ cùng 15 người. Không gộp 75 lượt người-seed thành 75 người độc lập. Đây là kiểm tra độ nhạy của cả quá trình chọn tham số và huấn luyện MLP, không tách riêng ảnh hưởng khởi tạo với ảnh hưởng lựa chọn siêu tham số. Seed 42 vẫn là kết quả chính đã chốt. Các lượt bổ sung không tạo tập kiểm tra độc lập ngoài WESAD. Loss đã lưu là mục tiêu train, không phải test loss theo epoch.')
    for target in [PACKAGE/'EXP020_MLP_architecture.ipynb',ROOT/'notebooks/experiments/Nhu_EXP-020_MLP_architecture.ipynb']:
        book=read_json(target)
        marker='## bổ sung ngày 22/11/2026 — kiểm tra độ nhạy theo seed'
        old=book['cells'];start=next((i for i,c in enumerate(old) if marker in ''.join(c['source'])),len(old))
        book['cells']=old[:start]+cells
        target.write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8')
    write_json(OUT/'verification.json',dict(status='passed',date='22/11/2026',seeds=[42,43,44,45,46],new_outer_folds=60,new_inner_fits=720,feature_sha256=read_json(runs[42]/'config.json')['data_sha256'],checks=['30 folds per seed','same labels and window keys','runner verification per seed','subject mean and sample SD','captured notebook outputs']))
    (OUT/'README.md').write_text('# exp020 — bổ sung ngày 22/11/2026\n\nnhư · EXP chung: EXP-020\n\nChạy mới seed 45 và 46, giữ seed 42 làm kết quả chính. Mỗi lượt mới đủ 30 outer folds và 360 inner fits; không đổi lưới hay dữ liệu. Không chọn seed tốt nhất.\n\n'+'\n\n'.join(lines)+'\n\nXem summary_by_seed.csv, scores_by_subject_seed.csv, prediction_agreement.csv và hình trong figures. Notebook chính và notebook làm việc có phần bổ sung kèm output đã thực thi.\n',encoding='utf-8')
    readme=PACKAGE/'README.md'
    text=readme.read_text(encoding='utf-8')
    marker='\n## bổ sung ngày 22/11/2026'
    if marker in text:text=text.split(marker)[0]
    readme.write_text(text+marker+'\n\nĐã chạy thêm seed 45 và 46, mỗi seed đủ 30 outer folds và 360 inner fits. Seed 42 vẫn là kết quả chính, không chọn seed có điểm cao nhất. Xem [kết quả theo seed](seed_sensitivity/20261122/README.md) và phần bổ sung trong hai notebook.\n',encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(OUT/'analysis_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__),'scripts/exp020_more_report.py')
        z.write(ROOT/'scripts/exp020_more_seeds.py','scripts/exp020_more_seeds.py')
    write_json(OUT/'artifact_checksums.json',{p.relative_to(OUT).as_posix():file_hash(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_checksums.json'})
    print(summary.to_string(index=False));print('notebooks updated')


if __name__=='__main__':main()
