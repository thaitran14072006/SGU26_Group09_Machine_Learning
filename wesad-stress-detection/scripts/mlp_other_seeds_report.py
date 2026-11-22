"""Đọc kết quả chạy thêm MLP và ghi notebook, hình, nhật ký."""
from pathlib import Path
import json,io,base64,contextlib,uuid,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
FOLDERS={'006':'01_baselines/EXP-006_MLP','010':'02_tuning/EXP-010','015':'03_balancing/EXP-015'}
NAMES={'006':'EXP006_MLP_baseline.ipynb','010':'EXP010_MLP_tuning.ipynb','015':'EXP015_MLP_balancing.ipynb'}
def main():
    import pandas as pd,numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap=argparse.ArgumentParser();ap.add_argument('--desktop',type=Path,required=True);a=ap.parse_args();desktop=a.desktop
    out=ROOT/'reports/mlp_seed_runs_20261122';out.mkdir(exist_ok=True,parents=True)
    sources={};all_frames=[];summary_frames=[];notes=[];staged={}
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    marker='## chạy thêm ngày 22/11/2026 — seed 43 và 44'
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
    for exp,path in FOLDERS.items():
        pack=ROOT/'experiments'/path;folder=pack/'seed_runs/20261122';figures=folder/'figures';figures.mkdir(exist_ok=True)
        scores=[];runs={}
        for seed in [42,43,44]:
            run=folder/f'seed_{seed}' if seed!=42 or exp=='006' else pack/'results'
            assert read(run/'verification.json')['status']=='passed'
            manifest=read(run/'config.json');assert manifest['settings']['seed']==seed
            t=pd.read_csv(run/'verified_metrics_by_subject.csv')
            assert len(t)==30
            if 'train_macro_f1' not in t:
                old=pd.read_csv(run/'train_test/metrics.csv')
                t=t.merge(old[['task','subject_id','train_macro_f1','train_balanced_accuracy','test_macro_f1']],on=['task','subject_id'],validate='one_to_one')
                assert np.allclose(t.macro_f1,t.test_macro_f1)
            if 'max_iter_reached' not in t:t['max_iter_reached']=t.reached_max_iter
            scores.append(t.assign(seed=seed,experiment='EXP-'+exp));runs[seed]=run
            if seed!=42 or exp=='006':
                notes.extend(json.loads(line) for line in (folder/f'seed_{seed}_journal.jsonl').read_text(encoding='utf8').splitlines())
            sources[f'{exp}/{seed}']={'metrics':(run/'verified_metrics_by_subject.csv').relative_to(ROOT).as_posix(),'sha256':hashlib.sha256((run/'verified_metrics_by_subject.csv').read_bytes()).hexdigest(),'environment':manifest['environment'],'data_sha256':manifest['data_sha256']}
        table=pd.concat(scores,ignore_index=True);assert len(table)==90
        table.to_csv(folder/'scores_by_seed.csv',index=False);all_frames.append(table)
        summary=table.groupby(['experiment','task','seed']).agg(subjects=('subject_id','nunique'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),ba_mean=('balanced_accuracy','mean'),ba_sd=('balanced_accuracy','std'),train_f1=('train_macro_f1','mean'),train_ba=('train_balanced_accuracy','mean'),epochs_mean=('n_iter','mean'),warnings=('warning_count','sum'),max_iter_folds=('max_iter_reached','sum')).reset_index()
        assert summary.subjects.eq(15).all();summary.to_csv(folder/'summary.csv',index=False);summary_frames.append(summary)
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for j,task in enumerate(['multiclass','binary']):
            g=summary[summary.task==task].sort_values('seed')
            for i,(metric,sd,label) in enumerate([('macro_f1_mean','macro_f1_sd','Macro-F1'),('ba_mean','ba_sd','Balanced Accuracy')]):
                ax=axes[i,j];ax.bar(g.seed.astype(str),g[metric],yerr=g[sd],capsize=4,color=['#748ca0','#b47d61','#7d996e']);ax.set(title=task,ylabel=label,xlabel='seed',ylim=(0,1.04));ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'EXP-{exp}: trung bình và SD theo 15 người');fig.tight_layout();fig.savefig(figures/'scores.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,task in zip(axes,['multiclass','binary']):
            g=summary[summary.task==task].sort_values('seed');x=np.arange(3)
            ax.bar(x-.18,g.train_f1,.36,label='train');ax.bar(x+.18,g.macro_f1_mean,.36,label='test')
            ax.set(xticks=x,xticklabels=g.seed,title=task,xlabel='seed',ylabel='Macro-F1',ylim=(0,1.08));ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(figures/'train_test.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for row,seed in enumerate([43,44]):
            for col,task in enumerate(['multiclass','binary']):
                ax=axes[row,col]
                for subject in ['S4','S9','S11','S17']:
                    d=read(runs[seed]/'folds'/task/'mlp'/subject/'details.json');loss=d['mlp_optimization']['loss_curve'];ax.plot(range(1,len(loss)+1),loss,label=subject)
                ax.set(title=f'{task}, seed {seed}',xlabel='epoch',ylabel='train objective');ax.legend(ncol=2);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(figures/'loss.png',dpi=160);plt.close(fig)
        cells=[];env={}
        def md(s):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=s))
        def code(s):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(s,'mlp_seed_report','exec'),env)
            outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else []
            for num in plt.get_fignums():
                data=io.BytesIO();plt.figure(num).savefig(data,format='png',dpi=130,bbox_inches='tight');outputs.append(dict(output_type='display_data',data={'image/png':base64.b64encode(data.getvalue()).decode()},metadata={}))
            plt.close('all');cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=s,outputs=outputs,execution_count=1+sum(c['cell_type']=='code' for c in cells)))
        md(marker+f'\n\nnhư. EXP chung: EXP-{exp}. Chạy mới seed 43/44, mỗi seed đủ 30 fold. Câu hỏi: kết quả thay đổi bao nhiêu khi đổi seed? Giữ dữ liệu và quy trình của EXP này. Seed 42 vẫn là mốc so sánh.')
        if exp=='006':md('Chạy thêm seed 42 cùng môi trường để làm mốc. Bản 42 cũ dùng pandas khác; không ghi đè kết quả cũ.')
        if exp=='015':md('Mỗi seed dùng best_params từ EXP010 cùng seed, rồi thêm ROS trên train. Không tuning lại, không resample test. Seed thay đổi cả MLP và ROS, không tách riêng hai ảnh hưởng.')
        code(f"from pathlib import Path\nimport pandas as pd, json\nroot=next(p for p in [Path.cwd(),Path.cwd()/'wesad-stress-detection',*Path.cwd().parents] if (p/'src/wesad').is_dir())\nfolder=root/'experiments/{path}/seed_runs/20261122'\nprint('ngày chạy: 22/11/2026')\nprint(pd.read_csv(folder/'summary.csv').to_string(index=False))")
        md('Thanh sai số là SD giữa 15 người. Seed 42 của EXP010/015 đọc từ lượt chính cũ; điểm train của mốc này lấy từ refit đã kiểm tra khớp dự đoán test. Các lượt mới đo train và test bằng cùng mô hình đang fit. Train chấm trên cửa sổ gốc, trước nhân bản ROS.')
        for name in ['scores','train_test','loss']:
            code(f"import matplotlib.pyplot as plt\nfig=plt.figure(figsize=(11,7))\nplt.imshow(plt.imread(folder/'figures/{name}.png'))\nplt.axis('off')\nfig.tight_layout()")
        md('Loss là train objective, không phải loss test. Các fold có thể chọn alpha khác nhau; không dùng độ lớn loss để xếp hạng mô hình.')
        code("events=[]\nfor seed in [43,44]:\n    events.extend(json.loads(line) for line in (folder/f'seed_{seed}_journal.jsonl').read_text(encoding='utf8').splitlines())\nprint(pd.DataFrame(events).to_string(index=False))")
        md('### ghi chú\n\nGiữ cả lượt tăng và giảm điểm. Không chọn seed tốt nhất. Các lượt dùng cùng 15 người, không phải dữ liệu test mới.')
        kind={'006':'baseline','010':'tuning','015':'balancing'}[exp]
        notebook_paths=[Path('experiments')/path/NAMES[exp],Path('notebooks/experiments')/f'Nhu_EXP-{exp}_MLP_{kind}.ipynb']
        for relative in notebook_paths:
            target=desktop/relative;original=read(target);sha=hashlib.sha256(target.read_bytes()).hexdigest()
            start=next((i for i,c in enumerate(original['cells']) if marker in ''.join(c['source'])),len(original['cells']))
            original['cells']=original['cells'][:start]+cells
            staged_file=out/'notebooks'/relative.name;staged_file.parent.mkdir(exist_ok=True);staged_file.write_text(json.dumps(original,ensure_ascii=False,indent=1),encoding='utf8')
            staged[relative.as_posix()]={'file':str(staged_file),'original_sha256':sha}
    pd.concat(summary_frames,ignore_index=True).to_csv(out/'summary.csv',index=False)
    pd.concat(all_frames,ignore_index=True).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(notes).to_csv(out/'journal.csv',index=False)
    assert len(notes)==14 and sum(n['status']=='xong' for n in notes)==7
    assert len({s['data_sha256'] for s in sources.values()})==1
    (out/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'notebook_updates.json').write_text(json.dumps(staged,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'verification.json').write_text(json.dumps({'status':'passed','new_runs':7,'new_outer_folds':210,'new_inner_fits':720,'new_train_test_same_fitted_model':True,'baseline_42_rerun_same_environment':True},indent=2),encoding='utf8')
    print(pd.concat(summary_frames).to_string(index=False));print('prepared six notebook updates')
if __name__=='__main__':main()
