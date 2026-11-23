from pathlib import Path
import pandas as pd

SUBJECTS = [f'S{i}' for i in list(range(2,12))+list(range(13,18))]
DEFAULT_DATA = 'data/features/g2_physiological68_group15_20261107/all_subjects.csv'
LABELS = {'binary': {0:'Non-Stress', 1:'Stress'}, 'multiclass':{1:'Baseline',2:'Stress',3:'Amusement'}}

def root_dir():
    # This module is located under <root>/demo_streamlit
    root=Path(__file__).resolve().parent.parent
    return root

def load_table(path):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy: {path}')
    return pd.read_csv(path)

def feature_columns(df):
    excluded={'subject_id','window_id','label','label_binary','label_multiclass','task','variant','y_true','y_pred','timestamp','split','start_s','end_s','label_original','label_fraction'}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

def get_target(df, task):
    if task == 'multiclass':
        if 'label_multiclass' in df.columns: return df['label_multiclass'].astype(int)
        if 'label' in df.columns: return df['label'].astype(int)
        if 'label_original' in df.columns: return df['label_original'].astype(int)
        raise ValueError('Thiếu cột label_original (1/2/3), label hoặc label_multiclass')
    if 'label_binary' in df.columns: return df['label_binary'].astype(int)
    if 'label' in df.columns: return (df['label'].astype(int)==2).astype(int)
    if 'label_multiclass' in df.columns: return (df['label_multiclass'].astype(int)==2).astype(int)
    if 'label_original' in df.columns: return (df['label_original'].astype(int)==2).astype(int)
    raise ValueError('Thiếu cột nhãn để suy ra binary')

def to_absolute(root, typed):
    p=Path(typed).expanduser()
    return p if p.is_absolute() else root/p
