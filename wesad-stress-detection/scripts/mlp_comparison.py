"""Chuẩn hóa hồ sơ và so sánh kết quả MLP đã lưu, không train lại."""
from pathlib import Path
import argparse,base64,contextlib,hashlib,io,json,shutil,uuid


def main():
    import pandas as pd
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score,balanced_accuracy_score
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output-root',type=Path)
    args=ap.parse_args();root=args.project_root.resolve();dest=(args.output_root or root).resolve()
    paths={'EXP-006':'01_baselines/EXP-006_MLP','EXP-010':'02_tuning/EXP-010','EXP-015':'03_balancing/EXP-015','EXP-020':'04_additional/EXP-020'}
    out=dest/'reports/mlp_comparison_20261121';out.mkdir(parents=True,exist_ok=True)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    subjects={'S'+str(n) for n in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]}
    sources={};frames=[];tt=[];diag=[];reference=None
    for exp,path in paths.items():
        package=root/'experiments'/path;run=package/'results';saved=pd.read_csv(run/'metrics_by_subject.csv')
        normalized=saved.rename(columns={'test_subject':'subject_id'}).copy()
        assert len(normalized)==30 and not normalized.duplicated(['task','subject_id']).any()
        if exp=='EXP-006':
            info=read(run/'run_info.json');digest=info['data_sha256']
            pred=pd.concat([pd.read_csv(p).assign(task=p.name.split('_')[1]) for p in sorted(run.glob('predictions_*_S*.csv'))],ignore_index=True)
            data=pd.read_csv(root/'data/features/g2_physiological68_group15_20261107/all_subjects.csv')
            assert hashlib.sha256((root/'data/features/g2_physiological68_group15_20261107/all_subjects.csv').read_bytes()).hexdigest()==digest
            mapped=data.iloc[pred.row_index.to_numpy()]
            assert mapped.subject_id.to_numpy().tolist()==pred.subject_id.tolist()
            pred['window_id']=mapped.window_id.to_numpy()
        else:
            manifest=read(run/'config.json');digest=manifest['data_sha256'];pred=pd.read_csv(run/'predictions.csv')
        assert len(pred)==12378 and not pred.duplicated(['task','subject_id','window_id']).any()
        labelkeys=pred[['task','subject_id','window_id','y_true']].sort_values(['task','subject_id','window_id']).reset_index(drop=True)
        if reference is None:reference=labelkeys
        else:pd.testing.assert_frame_equal(reference,labelkeys,check_dtype=False)
        for task in ['multiclass','binary']:
            part=normalized[normalized.task==task];assert set(part.subject_id)==subjects and part.n_test.sum()==6189
            for row in part.itertuples():
                g=pred[(pred.task==task)&(pred.subject_id==row.subject_id)]
                assert len(g)==row.n_test
                assert np.isclose(row.macro_f1,f1_score(g.y_true,g.y_pred,labels=[1,2,3] if task=='multiclass' else [0,1],average='macro',zero_division=0))
                assert np.isclose(row.balanced_accuracy,balanced_accuracy_score(g.y_true,g.y_pred))
        frames.append(normalized.assign(experiment=exp))
        train_test=pd.read_csv(run/'train_test/metrics.csv')
        assert len(train_test)==30
        joined=normalized.merge(train_test,on=['task','subject_id'],suffixes=('_source','_tt'),validate='one_to_one')
        assert np.allclose(joined.macro_f1,joined.test_macro_f1)
        assert np.allclose(joined.balanced_accuracy,joined.test_balanced_accuracy)
        tt.append(train_test[['task','subject_id','train_macro_f1','test_macro_f1','train_balanced_accuracy','test_balanced_accuracy']].assign(experiment=exp))
        v=normalized if exp=='EXP-006' else pd.read_csv(run/'verified_metrics_by_subject.csv')
        for task,g in v.groupby('task'):
            warning_count=int(g.warning_count.sum()) if 'warning_count' in g else int(g.warnings.astype('string').fillna('').ne('').sum())
            epoch=g['epochs'] if 'epochs' in g else g.n_iter
            diag.append(dict(experiment=exp,task=task,mean_epochs=epoch.mean(),max_epochs=int(epoch.max()),folds_at_200=int((epoch>=200).sum()),warning_count=warning_count))
        sources[exp]=dict(metrics=(run/'metrics_by_subject.csv').relative_to(root).as_posix(),metrics_sha256=hashlib.sha256((run/'metrics_by_subject.csv').read_bytes()).hexdigest(),data_sha256=digest)
        if exp in ['EXP-006','EXP-010','EXP-015']:
            target=dest/'experiments'/path;target.mkdir(parents=True,exist_ok=True)
            normalized.to_csv(target/'results.csv',index=False)
            if exp=='EXP-006':
                cfg={'protocol':'exp006_saved_baseline','features':'data/features/g2_physiological68_group15_20261107/all_subjects.csv','schema':'data/features/g2_physiological68_group15_20261107/feature_schema.json','feature_sha256':digest,'models':['mlp'],'tasks':info['tasks'],'seed':42,'expected_subjects':sorted(subjects),'nested':False,'balance':False,'normalization':'global_train','model_params':{'mlp':info['pipeline']['mlp']},'source_run_info':'results/run_info.json','record_type':'configuration reconstructed from saved run_info, not a new training run','prepared_date':'21/11/2026','saved_training_record_date':info['ngay_luu'],'preprocessing_record':{k:val for k,val in info['pipeline'].items() if k!='mlp'},'environment_record':{k:info[k] for k in ['python','numpy','pandas','sklearn']}}
                (target/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf8')
            else:shutil.copy2(run/'config.yaml',target/'config.yaml')
    assert len({x['data_sha256'] for x in sources.values()})==1
    scores=pd.concat(frames,ignore_index=True);scores.to_csv(out/'metrics_by_subject.csv',index=False)
    summary=scores.groupby(['experiment','task']).agg(subjects=('subject_id','nunique'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    train_test=pd.concat(tt,ignore_index=True);train_test.to_csv(out/'train_test_by_subject.csv',index=False)
    train_summary=train_test.groupby(['experiment','task'])[['train_macro_f1','test_macro_f1','train_balanced_accuracy','test_balanced_accuracy']].mean().reset_index();train_summary['macro_f1_gap']=train_summary.train_macro_f1-train_summary.test_macro_f1;train_summary.to_csv(out/'train_test_summary.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'convergence.csv',index=False)
    pairs=[]
    for current,ref in [('EXP-010','EXP-006'),('EXP-015','EXP-010'),('EXP-020','EXP-010')]:
        merged=scores[scores.experiment==current].merge(scores[scores.experiment==ref],on=['task','subject_id'],suffixes=('_new','_ref'),validate='one_to_one')
        for r in merged.itertuples():pairs.append(dict(experiment=current,reference=ref,task=r.task,subject_id=r.subject_id,delta_macro_f1=r.macro_f1_new-r.macro_f1_ref,delta_balanced_accuracy=r.balanced_accuracy_new-r.balanced_accuracy_ref))
    pd.DataFrame(pairs).to_csv(out/'paired_differences.csv',index=False)
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,task in enumerate(['multiclass','binary']):
        g=summary[summary.task==task].sort_values('experiment')
        for i,(metric,label) in enumerate([('macro_f1','Macro-F1'),('balanced_accuracy','Balanced Accuracy')]):
            ax=axes[i,j];ax.bar(g.experiment,g[metric+'_mean'],yerr=g[metric+'_sd'],capsize=4,color=['#728ba2','#b67d59','#7c976c','#8c7aa5']);ax.set(title=task,ylabel=label,ylim=(0,1.03));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('MLP: kết quả ngoài mẫu — thanh sai số là SD giữa 15 người');fig.tight_layout();fig.savefig(figs/'mlp_four_experiments.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,task in zip(axes,['multiclass','binary']):
        g=train_summary[train_summary.task==task].sort_values('experiment');x=np.arange(4)
        ax.bar(x-.18,g.train_macro_f1,.36,label='train');ax.bar(x+.18,g.test_macro_f1,.36,label='test LOSO');ax.set(xticks=x,xticklabels=g.experiment,title=task,ylabel='Macro-F1',ylim=(0,1.08));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figs/'mlp_train_test_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,sharey=True)
    order=sorted(subjects,key=lambda s:int(s[1:]))
    for ax,task in zip(axes,['multiclass','binary']):
        for exp in paths:
            g=scores[(scores.experiment==exp)&(scores.task==task)].set_index('subject_id').loc[order]
            ax.plot(order,g.macro_f1,marker='o',label=exp)
        ax.set(title=task,ylabel='Macro-F1',ylim=(0,1.04));ax.legend(ncol=4);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figs/'mlp_subject_comparison.png',dpi=170);plt.close(fig)
    # Notebook executes the saved-result analysis; original training dates stay in source records.
    cells=[];env={};marker='reports/mlp_comparison_20261121'
    def md(s):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=s))
    def code(s):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(s,'mlp_comparison_notebook','exec'),env)
        outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else []
        for num in plt.get_fignums():
            data=io.BytesIO();plt.figure(num).savefig(data,format='png',dpi=140,bbox_inches='tight');outputs.append(dict(output_type='display_data',data={'image/png':base64.b64encode(data.getvalue()).decode()},metadata={}))
        plt.close('all');cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=s,outputs=outputs,execution_count=1+sum(c['cell_type']=='code' for c in cells)))
    md('# so sánh bốn hướng mlp\n\nngười thực hiện: như. Ngày phân tích: 21/11/2026. Trạng thái: phân tích kết quả đã lưu, không train mới. EXP chung: EXP-006, EXP-010, EXP-015, EXP-020.\n\nCâu hỏi: tuning, cân bằng lớp và thay đổi kiến trúc/regularization có cải thiện kết quả ngoài mẫu so với cấu hình đối chứng không?')
    md('## dữ liệu và cách so sánh\n\nCùng physiological68, 6.189 cửa sổ và 15 người. Chỉ lấy các lượt chính seed 42, không chọn seed tốt nhất. EXP010 và EXP020 dùng nested LOSO với ba inner GroupKFold. EXP015 giữ tham số đã chọn của EXP010 và chỉ thêm ROS trên outer train. Mỗi độ đo được tính theo người rồi lấy mean và SD mẫu.\n\nEXP006 ghi pandas 3.1.1; các lượt sau ghi pandas 3.1.0. Vì khác phiên bản, so sánh EXP010 với EXP006 là đối chiếu kết quả đã lưu, chưa phải thử nghiệm kiểm soát chỉ một yếu tố trong cùng môi trường. Ba hướng sau có cùng môi trường theo hồ sơ. Benchmark đã được dùng nhiều lần, chưa có test độc lập ngoài WESAD.')
    code(f"from pathlib import Path\nimport pandas as pd, json\ncandidates=[Path.cwd(),Path.cwd()/'wesad-stress-detection',*Path.cwd().parents,Path({str(root)!r}),Path({str(dest)!r})]\nroot=next(p for p in candidates if (p/'{marker}').is_dir())\nfolder=root/'{marker}'\nprint('ngày phân tích: 21/11/2026')\nprint(pd.read_csv(folder/'summary.csv').to_string(index=False))")
    md('## kết quả và khác biệt theo người')
    for name in ['mlp_four_experiments.png','mlp_subject_comparison.png']:
        code(f"import matplotlib.pyplot as plt\nfig=plt.figure(figsize=(12,8))\nplt.imshow(plt.imread(folder/'figures/{name}'))\nplt.axis('off')\nfig.tight_layout()")
    md('## train và test\n\nĐiểm train lấy từ các lượt refit bổ sung đã lưu, không phải điểm inner validation. Đã đối chiếu điểm test với lượt đánh giá chính cho đủ 30 fold mỗi EXP. Không có đường loss test theo epoch trong các hồ sơ này.')
    code("print(pd.read_csv(folder/'train_test_summary.csv').to_string(index=False))\nfig=plt.figure(figsize=(12,5))\nplt.imshow(plt.imread(folder/'figures/mlp_train_test_comparison.png'))\nplt.axis('off')\nfig.tight_layout()")
    md('## hội tụ và đối chứng')
    code("print(pd.read_csv(folder/'convergence.csv').to_string(index=False))\npairs=pd.read_csv(folder/'paired_differences.csv')\nprint(pairs.groupby(['experiment','reference','task'])[['delta_macro_f1','delta_balanced_accuracy']].mean().to_string())")
    md('## nhận xét tham khảo\n\nỞ ba lớp, EXP006 có Macro-F1 cao nhất trong bốn lượt chính đã lưu. Tuning trong EXP010 không làm điểm trung bình tăng so với baseline. Khi thêm ROS ở EXP015, Macro-F1 ba lớp giảm so với EXP010, còn nhị phân chỉ tăng nhẹ.\n\nEXP020 cải thiện Macro-F1 so với EXP010, rõ hơn ở nhị phân, nhưng ba lớp vẫn thấp hơn EXP006. Điểm train của các lượt refit đều đạt 1 trong khi test thấp hơn, cho thấy khoảng cách train/test còn lớn. Đạt điều kiện dừng và không có cảnh báo không có nghĩa mô hình đã tổng quát hóa tốt.\n\nLượt kiểm tra seed 43/44 của EXP020 ngày 20/11 cho thấy nhị phân nhạy với seed hơn ba lớp. Không đổi kết quả chính sang seed 44 chỉ vì điểm cao hơn.\n\nĐây mới là đối chiếu MLP. Chưa kết luận MLP là mô hình tốt nhất của nhóm và chưa chốt mô hình demo từ bảng này. Có thể tiếp tục thử môi trường đồng nhất cho EXP006/010 hoặc tách ảnh hưởng kiến trúc và alpha trong inner CV nếu nhóm cần nghiên cứu sâu hơn.')
    md('## nguồn và phạm vi\n\nCác đường dẫn và SHA256 nằm trong source_manifest.json. Chuẩn hóa config/results và tổng hợp không thay đổi kết quả gốc. Không tạo thêm lượt train hay gán ngày 21/11 cho việc huấn luyện cũ.')
    notebook=dest/'notebooks/03_modeling/Nhu_MLP_comparison.ipynb';notebook.parent.mkdir(parents=True,exist_ok=True)
    notebook.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'}},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1),encoding='utf8')
    (out/'source_manifest.json').write_text(json.dumps({'analysis_date':'21/11/2026','analysis_only':True,'sources':sources},ensure_ascii=False,indent=2),encoding='utf8')
    (out/'verification.json').write_text(json.dumps({'status':'passed','experiments':list(paths),'folds_per_experiment':30,'same_window_keys_and_labels':True,'metrics_recomputed_from_predictions':True,'train_test_metrics_match':True,'new_training_runs':0,'limitation':'EXP006 pandas version differs from EXP010/015/020'},indent=2),encoding='utf8')
    print(summary.to_string(index=False));print('saved:',notebook)


if __name__=='__main__':main()
