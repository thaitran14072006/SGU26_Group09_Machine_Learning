"""Verify completed G4 RF/MLP, compare paired subjects with G3, package handoff."""
import argparse
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from wesad.g4_nhu import verify
from wesad.utils import PROJECT_ROOT, file_hash, read_config, read_json, resolve_path, write_csv, write_json

def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='outputs/g4_nhu_rf_mlp_outer4')
    args=p.parse_args()
    run=resolve_path(args.run)
    config=read_config(run/'config.yaml')
    checked=verify(config,run)
    baseline=read_json(PROJECT_ROOT/'reports/g3_group/summary.json')
    if checked['data_sha256'] != baseline['data_sha256']: raise ValueError('G3/G4 data differ')
    before=pd.read_csv(PROJECT_ROOT/'reports/g3_group/metrics_by_subject.csv')
    before=before[before.model.isin(config['models']) & before.subject_id.isin(config['outer_subjects'])]
    after=pd.read_csv(run/'verified_metrics_by_subject.csv')
    joined=before.merge(after,on=['task','model','subject_id'],suffixes=('_g3','_g4'),how='outer',validate='one_to_one',indicator=True)
    if len(joined)!=16 or not (joined['_merge']=='both').all() or not (joined.n_test_g3==joined.n_test_g4).all():
        raise ValueError('Cannot pair the 16 assigned G3/G4 subject results')
    rows=[]
    for (task,model),part in joined.groupby(['task','model']):
        row={'task':task,'model':model,'n_subjects':len(part)}
        for metric in ('macro_f1','balanced_accuracy'):
            for phase in ('g3','g4'):
                values=part[f'{metric}_{phase}']
                row[f'{metric}_{phase}_mean']=float(values.mean())
                row[f'{metric}_{phase}_std']=float(values.std(ddof=1))
            row[metric+'_delta_mean']=float((part[metric+'_g4']-part[metric+'_g3']).mean())
        rows.append(row)
    out=PROJECT_ROOT/'reports/g4_nhu'
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'default_vs_tuned.csv',pd.DataFrame(rows))
    write_csv(out/'paired_metrics_by_subject.csv',joined.drop(columns='_merge'))
    for file in ('summary.json','best_params.json','verification.json','environment.json','verified_metrics_by_subject.csv'):
        (out/file).write_bytes((run/file).read_bytes())
    write_json(out/'provenance.json',{'run':run.relative_to(PROJECT_ROOT).as_posix(),
        'signature':read_json(run/'config.json')['signature'],
        'data_sha256':checked['data_sha256'],
        'baseline_metrics_sha256':file_hash(PROJECT_ROOT/'reports/g3_group/metrics_by_subject.csv'),
        'scope':'RF/MLP assigned outer S4/S9/S11/S17 only, not the full group 15-subject result',
        'warnings_by_fold':[{ 'task':r.task,'model':r.model,'subject_id':r.subject_id,'count':int(r.n_warnings)}
                            for r in after.itertuples() if r.n_warnings],
        'interpretation':'Descriptive paired comparison, no statistical significance test; negative tuning deltas are retained.'})
    lines=['# G4 RF/MLP: mặc định so với tuning','',
           'Chỉ lô S4/S9/S11/S17, hai bài toán. Mean ± SD mẫu của 4 người; Δ = G4 − G3. Không thay bảng 15 người của nhóm.','',
           '| Bài toán | Mô hình | Macro-F1 G3 | Macro-F1 G4 | Δ | BA G3 | BA G4 |',
           '|---|---|---|---|---|---|---|']
    for r in rows:
        def cell(metric,phase): return f"{r[f'{metric}_{phase}_mean']:.4f} ± {r[f'{metric}_{phase}_std']:.4f}"
        lines.append(f"| {r['task']} | {r['model'].upper()} | {cell('macro_f1','g3')} | {cell('macro_f1','g4')} | {r['macro_f1_delta_mean']:+.4f} | {cell('balanced_accuracy','g3')} | {cell('balanced_accuracy','g4')} |")
    lines+=['','Tuning có thể thấp hơn mặc định. Không chọn lại lưới sau khi xem outer test. Chưa kiểm định G5.',
            'Cảnh báo hội tụ nằm trong details.json từng fold; số fold có cảnh báo: '+str(int((after.n_warnings>0).sum()))+'.']
    (out/'KET_QUA.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    optimization_rows=[]
    for row,task in enumerate(config['tasks']):
        for col,subject in enumerate(config['outer_subjects']):
            detail=read_json(run/'folds'/task/'mlp'/subject/'details.json')
            opt=detail['mlp_optimization']
            optimization_rows.append({'task':task,'subject_id':subject,**{k:opt[k] for k in ('n_iter','max_iter','loss','reached_max_iter')},
                                      'warnings':json_warning(detail['warnings'])})
    write_csv(out/'mlp_convergence.csv',pd.DataFrame(optimization_rows))
    epochs=[r['n_iter'] for r in optimization_rows]
    with (out/'KET_QUA.md').open('a',encoding='utf8') as stream:
        stream.write(f'\nMLP refit: {len(epochs)} lượt, dừng sau {min(epochs)}–{max(epochs)} epoch; '
                     f"{sum(r['reached_max_iter'] for r in optimization_rows)} lượt chạm max_iter=200. "
                     'Loss train giảm về mức nhỏ trong hình đã lưu. Đây là kiểm tra tối ưu trên train, '
                     'không phải validation loss hoặc bằng chứng không overfit.\n')
    archive=PROJECT_ROOT/'outputs/g4_nhu_rf_mlp_outer4_handoff.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if path.is_file(): z.write(path,'run/'+path.relative_to(run).as_posix())
        for path in sorted(out.iterdir()):
            if path.is_file(): z.write(path,'report/'+path.name)
    print(checked)
    print((out/'KET_QUA.md').read_text(encoding='utf8'))
    print('ZIP:',archive)

def json_warning(warnings):
    import json
    return json.dumps(warnings,ensure_ascii=False)

if __name__=='__main__': main()
