"""Tạo 15 notebook và lưu kết quả chạy. Không ghi đè notebook đã tồn tại."""
import contextlib
import io
import traceback
import nbformat
from thai_experiments import ROOT, configs, export_excel_log

def main():
    for index, config in enumerate(configs()):
        path = ROOT / config['notebook']
        if path.exists():
            print('Giữ notebook đã có:', path.name, flush=True)
            continue
        markdown = nbformat.v4.new_markdown_cell
        code = nbformat.v4.new_code_cell
        cells = [
            markdown(f"# {config['date']} — {config['title']}\n\n{config['exp_id']} · Thái · House Prices\n\n"
                f"Giả thuyết: {config['hypothesis']}\n\n"
                'Ngày ở tiêu đề là mốc ghi nhật ký theo yêu cầu; thời điểm máy lưu riêng trong JSON.'),
            markdown('# Tìm thư mục dự án'),
            code('from pathlib import Path\nimport sys\n\n'
                'ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]\n'
                '             if (p / "data/raw/train.csv").is_file()), None)\n'
                'if ROOT is None:\n    raise RuntimeError("Không tìm thấy Challenge1.")\n'
                'sys.path.insert(0, str(ROOT / "members/Thai/src"))\n'
                'from thai_experiments import load_data, get_split, eda, configs, run_experiment, finalize\n'
                'print("Project:", ROOT)\nprint("Python:", sys.executable)'),
            markdown('# Kiểm tra dữ liệu và cách chia'),
            code('train, test = load_data()\ndev, holdout, folds = get_split(train)\n'
                'print("Train:", train.shape, "Test:", test.shape)\n'
                'print("Development:", len(dev), "Holdout:", len(holdout), "CV:", len(folds))\n'
                'print(train.iloc[dev][["Id", "GrLivArea", "OverallQual", "SalePrice"]].head().to_string(index=False))'),
            markdown('# EDA trên development'),
            code('eda()'),
            markdown('# Chốt cấu hình và chạy 5-fold CV\n\n'
                'Giữ holdout để đánh giá cuối. Điền thiếu, mã hóa và chuẩn hóa nằm trong pipeline.'),
            code(f'config = configs()[{index}]\nprint(config)\nresult = run_experiment(config)'),
            markdown('# Nhận xét\n\n'
                'So sánh CV RMSE log và độ lệch chuẩn trên cùng fold. Nhật ký tự thêm một dòng, '
                'gồm ngày, nguồn notebook và JSON minh chứng. Không chọn mô hình theo holdout.'),
        ]
        if index == 14:
            cells += [markdown('# Chọn cấu hình và xuất dự đoán\n\nChạy sau khi đủ 15 thí nghiệm.'),
                      code('summary = finalize()')]
        notebook = nbformat.v4.new_notebook(cells=cells,
            metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}})
        namespace = {'__name__': '__main__'}
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count += 1
            buffer = io.StringIO()
            cell.execution_count = count
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(compile(cell.source, str(path), 'exec'), namespace)
            except Exception as error:
                cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=buffer.getvalue()),
                    nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error),
                        traceback=traceback.format_exc().splitlines())]
                nbformat.write(notebook, path)
                raise
            cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=buffer.getvalue())]
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print('Đã lưu:', path.name, flush=True)
    export_excel_log()

if __name__ == '__main__':
    main()
