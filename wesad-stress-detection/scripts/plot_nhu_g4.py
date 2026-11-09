"""Plot saved G4 MLP loss in a separate matplotlib environment; never refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('outputs/g4_nhu_rf_mlp_outer4'))
    p.add_argument('--output',type=Path,default=Path('reports/g4_nhu'))
    args=p.parse_args()
    checked=json.loads((args.run/'verification.json').read_text(encoding='utf8'))
    if checked['status']!='passed' or checked['scope']!='assigned_outer4' or checked['folds']!=16:
        raise ValueError('Plot only a verified complete assigned four-subject batch')
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
    hashes={}
    for row,task in enumerate(('multiclass','binary')):
        for col,subject in enumerate(('S4','S9','S11','S17')):
            path=args.run/'folds'/task/'mlp'/subject/'details.json'
            hashes[f'{task}/{subject}']=hashlib.sha256(path.read_bytes()).hexdigest()
            opt=json.loads(path.read_text(encoding='utf8'))['mlp_optimization']
            ax=axes[row,col]
            ax.plot(range(1,opt['n_iter']+1),opt['loss_curve'],color='#087F82')
            ax.set(title=f'{task} / {subject} ({opt["n_iter"]} epochs)',xlabel='Epoch',ylabel='Train loss')
            ax.grid(alpha=.2)
    fig.suptitle('MLP best-candidate refit: outer-train loss only\nS4/S9/S11/S17 batch; not validation loss or 15-subject results',fontsize=13)
    fig.savefig(args.output/'mlp_refit_loss.png',dpi=180)
    plt.close(fig)
    (args.output/'plot_environment.json').write_text(json.dumps({'python':sys.version,
        'matplotlib':matplotlib.__version__,'source_details_sha256':hashes,
        'scope':'render saved loss only; no model fitting'},indent=2),encoding='utf8')
    print('Saved MLP loss figure')

if __name__=='__main__': main()
