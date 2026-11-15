"""Tạo và thực thi notebook RF, giữ output thật bằng Jupyter kernel."""
from pathlib import Path
import json
import time
import os
import sys
import base64
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output

MEMBER=Path(__file__).resolve().parents[1]
ROOT=MEMBER.parents[1]
PATH=MEMBER/'notebooks/15_11_2026_rf.ipynb'


def markdown(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))


def main():
    if PATH.exists():
        old=json.loads(PATH.read_text(encoding='utf8'))
        assert not any(c.get('outputs') for c in old['cells']), 'Notebook đã chạy; không ghi đè.'
    PATH.parent.mkdir(parents=True,exist_ok=True)
    cells=[
        markdown('# random forest house prices\n\nngười thực hiện: như\n\nngày: 15/11/2026\n\nchạy mới: baseline, cải tiến đặc trưng và siêu tham số, phân tích sai số, dự đoán test.'),
        markdown('## 1. câu hỏi nghiên cứu\n\nxử lý NA theo ý nghĩa và thêm đặc trưng diện tích, tuổi nhà, tiện ích có giảm sai số rf không? giới hạn độ sâu và tăng kích thước lá có giảm khoảng cách train–validation không? kiểm chứng bằng cv, chấp nhận cả phương án tăng sai số.'),
        markdown('## 2. dữ liệu và môi trường\n\ntrain có nhãn SalePrice; test chỉ dùng xuất dự đoán. loại Id khỏi đầu vào, huấn luyện log tự nhiên của SalePrice. dùng cùng danh sách Id và fold đã có trong project.'),
        code("from pathlib import Path\nimport sys\nfrom IPython.display import display\ncandidates = [q for p in [Path.cwd(), *Path.cwd().parents] for q in (p, p / 'Challenge/Challenge1')]\nroot = next(p for p in candidates if (p / 'data/raw/train.csv').is_file())\nsys.path.insert(0, str(root / 'members/Nhu/src'))\nimport rf_workflow as rf\nrun, train, test, dev, hold, folds = rf.setup()\n"),
        markdown('## 3. eda trên development\n\nkiểm tra giá, log giá, thiếu dữ liệu và quan hệ diện tích–giá. giữ nguyên holdout trong giai đoạn nghiên cứu. NA chỉ đổi thành absent ở các thuộc tính mà từ điển ghi là không có tiện ích; các giá trị chưa biết còn lại được điền trong từng fold.'),
        code('missing = rf.eda(dev, run)\ndisplay(missing)\n'),
        markdown('## 4. protocol và sáu cấu hình\n\ndevelopment 80%, holdout 20%, seed 7. cv 5 fold shuffle seed 7 trên development. imputer và encoder nằm trong pipeline; rf không cần scale. các phương án cố định trước khi chạy: baseline; xử lý NA theo nghĩa; thêm feature; depth 12; leaf 3; depth 12 kết hợp leaf 3. mọi phương án giữ 100 cây. sd là độ lệch chuẩn mẫu giữa 5 fold, không phải khoảng tin cậy.'),
        code('display(rf.pd.DataFrame(rf.TRIALS, columns=["exp_id", "features", "max_depth", "min_samples_leaf", "hypothesis"]))\nprint(rf.pipeline("baseline").get_params()["rf"])\n'),
        markdown('## 5. thực thi và chọn bằng cv\n\nkhông dùng holdout để chọn cấu hình. lưu điểm từng fold, dự đoán oof và cấu hình đã chọn trước khi đánh giá holdout.'),
        code('comparison, selected = rf.run_trials(dev, folds, run)\ndisplay(comparison)\nprint("cấu hình chọn bằng cv:", selected)\n'),
        markdown('## 6. so sánh cải tiến\n\nđối chiếu exp002 với exp001 để đánh giá xử lý NA và kiểu mã nhà; exp003 với exp002 để đánh giá nhóm feature mới. exp004 và exp005 so với exp003 để tách ảnh hưởng depth và leaf. exp006 kiểm tra kết hợp. so sánh theo từng fold, không chỉ lấy một điểm đẹp.'),
        code('rf.plot_comparison(comparison, run)\ndelta = comparison[["exp_id", "cv_rmse_mean", "cv_rmse_std"]].copy()\ndelta["delta_vs_baseline"] = delta.cv_rmse_mean - comparison.iloc[0].cv_rmse_mean\ndisplay(delta)\n'),
        markdown('## 7. holdout, sai số và importance\n\nchỉ đánh giá cấu hình đã chốt bằng cv; không quay lại chọn tham số theo holdout. phần dư log dương là dự đoán thấp hơn giá thật. importance giảm impurity tổng hợp các cột one-hot về feature gốc, lấy trung bình từ 5 train fold; không phải quan hệ nhân quả.'),
        code('holdout_report, error_groups, submission_preview = rf.finish(train, test, dev, hold, selected, comparison, run)\nprint(holdout_report)\ndisplay(error_groups)\ndisplay(submission_preview)\n'),
        markdown('## 8. dự đoán cuối và kiểm tra chạy lại\n\nfit lại cấu hình đã chọn trên toàn bộ 1.460 nhà; xuất 1.459 dự đoán, kiểm tra thứ tự Id và giá dương. nạp lại pipeline đã lưu để đối chiếu dự đoán. kiểm tra này là chạy lại inference, không tuyên bố đã train độc lập lần hai.'),
        code('rf.verify(run)\n'),
        markdown('## 9. kết luận và nhật ký\n\nđọc số đo thực tế bên dưới. chưa nộp Kaggle, không có điểm leaderboard. lịch sử lần này được lưu riêng, không gán ngày cũ và không tạo lỗi giả. có thể làm tiếp ablation từng nhóm feature, permutation importance trong cv và khảo sát max_features; không dùng holdout để tối ưu thêm.'),
        code('print((root / "reports/individual/Nhu_RF_report.md").read_text(encoding="utf-8"))\nprint("nguồn kết quả:", run)\n'),
    ]
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='wesad312 (Python 3.12)',language='python',name='wesad312-d'),language_info=dict(name='python',version='3.12.14')),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):c['id']=f'nhu-rf-{i:02}'
    def save():PATH.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf8')
    save()
    os.chdir(ROOT)
    shell=TerminalInteractiveShell.instance()
    shell.run_line_magic('matplotlib','inline')
    embedded=set()
    for i,cell in enumerate(cells):
        if cell['cell_type']!='code':continue
        print(f'cell {i}: running',flush=True)
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(''.join(cell['source']),store_history=True)
        cell['execution_count']=shell.execution_count-1
        for stream in ['stdout','stderr']:
            text=getattr(captured,stream)
            if text:
                cell['outputs'].append(dict(output_type='stream',name=stream,text=text))
                if stream=='stdout':print(text.strip(),flush=True)
        for item in captured.outputs:
            cell['outputs'].append(dict(output_type='display_data',data=item.data,metadata=item.metadata))
        current_run=shell.user_ns.get('run')
        if current_run is not None:
            for image_path in sorted((current_run/'figures').glob('*.png')):
                if image_path not in embedded:
                    if not any('image/png' in item.data for item in captured.outputs):
                        cell['outputs'].append(dict(output_type='display_data',metadata={},data={
                            'image/png':base64.b64encode(image_path.read_bytes()).decode('ascii'),
                            'text/plain':f'hình đã lưu: {image_path.name}'}))
                    embedded.add(image_path)
        error=result.error_before_exec or result.error_in_exec
        if error:
            cell['outputs'].append(dict(output_type='error',ename=type(error).__name__,evalue=str(error),traceback=[str(error)]))
        save()
        if error:raise RuntimeError(f'cell {i} failed; notebook output saved') from error
    print('NOTEBOOK COMPLETE',PATH,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
