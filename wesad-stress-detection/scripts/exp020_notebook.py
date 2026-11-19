"""Tạo và thực thi các ô đọc kết quả EXP-020 bằng Python hiện tại, không train lại."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/04_additional/EXP-020'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    now=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    date=now.strftime('%d/%m/%Y')
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,id=uuid.uuid4().hex[:8],metadata={},source=source.strip()+'\n')
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    md=lambda s:add('markdown',s)
    code=lambda s:add('code',s)
    md(f'''# 01 — exp020: kiến trúc và regularization của mlp

người thực hiện: như · ngày: {date} · trạng thái: new

ba lượt train mới bằng CLI; notebook thực thi các ô đọc dữ liệu, kết quả và hình thật.
R01 là kết quả chính, R02/R03 kiểm tra tái lập cùng seed. không chọn lượt có điểm cao nhất.
EXP chung: EXP-020.''')
    md('''## 02 — câu hỏi nghiên cứu

mạng nhỏ hơn và alpha lớn hơn có cải thiện macro f1 / balanced accuracy trên người được giữ lại làm test so với exp010 không?

giả thuyết: lưới [16] / [32,16] với alpha 0.01 / 0.1 có thể hạn chế khớp quá sát train.
giữ learning rate 0.001, max_iter 200, seed 42, dữ liệu, scaling và không ROS.
đây là khảo sát kết hợp kiến trúc và regularization; chưa tách riêng ảnh hưởng từng yếu tố.
lưới được chốt trước lượt exp020 nhưng đã xem benchmark ở các EXP trước: kết luận mang tính thăm dò.''')
    md('## 03 — dữ liệu và môi trường')
    code('''from pathlib import Path
import sys, json
root = next(p for p in [Path.cwd(), Path.cwd()/'wesad-stress-detection', *Path.cwd().parents] if (p/'src/wesad').is_dir())
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root/'scripts'))
import pandas as pd
from wesad.utils import read_json, read_config, file_hash, environment
from wesad.data import SUBJECTS, load_feature_table
package = root/'experiments/04_additional/EXP-020'
run = package/'results'
config = read_config(run/'config.yaml')
frame, features = load_feature_table(root/config['features'],root/config['schema'],SUBJECTS)
assert len(frame)==6189 and len(features)==68
assert file_hash(root/config['features'])==config['feature_sha256']
print('ngày chạy:',read_json(run/'notebook_provenance.json')['date'])
print('python notebook:',sys.version.split()[0])
print('python thực thi:',sys.executable)
print('dữ liệu:',config['features'])
print('sha256:',config['feature_sha256'])
print('số dòng:',len(frame),'| feature:',len(features),'| subject:',frame.subject_id.nunique())
print(pd.crosstab(frame.subject_id,frame.label_original).to_string())
print('môi trường train:',read_json(run/'environment.json'))''')
    md('''## 04 — protocol

15 outer LOSO folds cho từng task, train 14 người và test 1 người. multiclass: 1 baseline, 2 stress, 3 amusement.
binary: stress=1, còn lại=0. mỗi outer train có 3 inner GroupKFold để chọn tham số theo macro f1.
median imputation và StandardScaler fit bên trong từng train fold; không ROS.
outer test không dùng chọn tham số hoặc dừng epoch. mỗi lượt: 360 inner fits + 30 outer refits.
đánh giá bằng mean và sd mẫu (ddof=1) theo 15 người; không lấy điểm trên toàn bộ cửa sổ thay cho mean theo người.''')
    code('''verification=read_json(run/'verification.json')
assert verification['status']=='passed' and verification['outer_folds']==30
print(json.dumps(verification,ensure_ascii=False,indent=2))
print('split ví dụ multiclass/S4:')
print(json.dumps(read_json(run/'splits.json')['multiclass/mlp/S4'],ensure_ascii=False,indent=2))''')
    md('## 05 — cấu hình đã dùng')
    code('''from sklearn.model_selection import ParameterGrid
print('seed:',config['seed'],'| balance:',config['balance'])
print('tham số cố định:',config['model_params'])
for i,candidate in enumerate(ParameterGrid(config['search_spaces']['mlp']),1):
    print('ứng viên',i,candidate)
print('tham số được chọn theo inner CV:')
print(pd.read_csv(run/'convergence.csv')[['task','subject_id','hidden_sizes','alpha','n_iter']].to_string(index=False))''')
    md('''## 06 — thực thi

ba lượt train thật đã chạy và được lưu trong results cùng repeatability/R02, R03.
mặc định chỉ đọc kết quả. bật RUN_TRAIN để chạy một lượt mới với thư mục output riêng;
lượt mới chưa tự thay thế kết quả chốt. cần dùng môi trường train đúng phiên bản.
runner dùng src/wesad/evaluation.py, không viết lại LOSO trong notebook.''')
    code('''import subprocess
RUN_TRAIN=False
if RUN_TRAIN:
    assert environment()==read_json(run/'environment.json'), 'chọn đúng môi trường train trước'
    subprocess.run([sys.executable,'-X','utf8','-u',str(root/'scripts/exp020_train.py')],cwd=root,check=True)
else:
    print('chỉ đọc kết quả đã lưu; không huấn luyện lại')
print('nguồn R01:',read_json(run/'notebook_provenance.json')['command'])
print(pd.read_csv(run/'repeatability.csv').to_string(index=False))''')
    md('## 07 — kết quả thực tế')
    code('''metrics=pd.read_csv(run/'verified_metrics_by_subject.csv')
assert len(metrics)==30
summary=metrics.groupby('task')[['macro_f1','balanced_accuracy']].agg(['mean','std'])
print(summary.to_string())
print();print('điểm từng subject:')
print(metrics[['task','subject_id','macro_f1','balanced_accuracy','warning_count']].to_string(index=False))
print();print('so sánh theo cặp với exp006 và exp010:')
print(pd.read_csv(run/'comparison_summary.csv').to_string(index=False))''')
    md('''## 08 — hình minh họa

nguồn: metrics, inner CV, dự đoán và loss của R01; train/test lấy từ refit bổ sung giữ nguyên tham số.
đã đối chiếu dự đoán test ở đủ 30 folds với lượt nested gốc. trục train/test là subject, không phải epoch.
loss là mục tiêu trên train của mô hình được chọn, không có loss validation/test theo epoch.
không so loss tuyệt đối giữa alpha khác nhau để kết luận tổng quát hóa.''')
    code('''import matplotlib.pyplot as plt
for name in ['exp010_vs_exp020.png','train_vs_test.png','train_loss.png','confusion_matrices.png','selected_candidates.png']:
    path=run/'figures'/name
    print(name)
    image=plt.imread(path)
    fig,ax=plt.subplots(figsize=(13,8))
    ax.imshow(image);ax.axis('off');fig.tight_layout();plt.show()''')
    md('## 09 — phân tích và giới hạn')
    code('''print();print('cảnh báo thực tế:')
convergence=pd.read_csv(run/'convergence.csv')
print(convergence[['task','subject_id','n_iter','reached_max_iter','warning_count','warnings']].fillna('').to_string(index=False))''')
    md(f'''## 10 — kết luận và nhật ký minh chứng

{date}: tạo và chạy grid mới; giữ R01 làm kết quả chính; chạy R02/R03 để kiểm tra cùng seed.

lỗi thực tế khi khởi động hai lượt song song: FileExistsError (WinError 183), tên thư mục timestamp trùng lượt đầu.
hai lần khởi động này chưa train nên không tính là hai lượt thành công. sửa runner thêm UUID vào tên output,
sau đó chạy R02/R03 mới. giữ snapshot launcher ban đầu ở R01; mã ML chung và cấu hình không đổi.

chưa có khảo sát seed khác, grid rộng, kiểm định thống kê hoặc final holdout độc lập.
không nhận notebook này là các thao tác gõ tay trong VS Code: output được tạo bằng thực thi các ô Python với bộ thu stdout/ảnh.
mã commit bên dưới là HEAD lúc bắt đầu train; thay đổi exp020 chưa commit/push và đã lưu snapshot nguồn.''')
    code('''print('provenance R01:')
print(json.dumps(read_json(run/'notebook_provenance.json'),ensure_ascii=False,indent=2))
print('kiểm tra tái lập:')
print(json.dumps(read_json(run/'repeatability_verification.json'),ensure_ascii=False,indent=2))
print('config:',run/'config.yaml')
print('snapshots:',run/'source_snapshot.zip',run/'report_source_snapshot.zip')
print('minh chứng lỗi:',package/'launch_history.md')''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3.12.14 (WESAD)',language='python',name='python3'),
        language_info=dict(name='python',version='3.12.14'),
        execution=dict(method='sequential Python exec with captured stdout and matplotlib figures',
            date=date,python=sys.executable,training_triggered_by_notebook=False)))
    namespace={'__name__':'__main__'}
    os.chdir(ROOT)
    active_outputs=[]
    def capture_show(*args,**kwargs):
        for number in plt.get_fignums():
            fig=plt.figure(number);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=100,bbox_inches='tight')
            active_outputs.append(dict(output_type='display_data',metadata={},data={
                'image/png':base64.b64encode(buffer.getvalue()).decode(),
                'text/plain':['<matplotlib figure from executed cell>']}))
        plt.close('all')
    plt.show=capture_show
    count=0
    for cell in cells:
        if cell['cell_type']!='code':continue
        ast.parse(cell['source']);count+=1;cell['execution_count']=count
        stream=io.StringIO();active_outputs=[]
        try:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell['source'],f'exp020_cell_{count}','exec'),namespace)
        except Exception:
            raise RuntimeError(f'notebook cell {count}: '+traceback.format_exc())
        outputs=[]
        if stream.getvalue():outputs.append(dict(output_type='stream',name='stdout',text=stream.getvalue()))
        outputs.extend(active_outputs);cell['outputs']=outputs
    target=ROOT/'notebooks/experiments/Nhu_EXP-020_MLP_architecture.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf8')
    (PACKAGE/'EXP020_MLP_architecture.ipynb').write_bytes(target.read_bytes())
    print('executed',count,'cells; saved:',target)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
