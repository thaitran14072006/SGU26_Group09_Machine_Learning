"""Execute report cells with IPython in one process and retain actual outputs."""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/exp009_matplotlib')
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
os.environ['IPYTHONDIR'] = str(ROOT / 'outputs/exp009_ipython')

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    path = ROOT / 'experiments/02_tuning/EXP-009_RF/experiment.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic('matplotlib', 'inline')
    # Terminal shells default to text/plain; retain notebook tables, prose and PNGs.
    shell.display_formatter.active_types = [
        'text/plain', 'text/html', 'text/markdown', 'image/png', 'image/svg+xml', 'application/json',
    ]
    limit = 11 if args.preflight else len(notebook.cells)
    for index, cell in enumerate(notebook.cells[:limit]):
        if cell.cell_type != 'code':
            continue
        print(f'Executing cell {index}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout, captured.stderr)
            raise RuntimeError(f'Notebook cell {index} failed') from (result.error_before_exec or result.error_in_exec)
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = result.execution_count
    if not args.preflight:
        notebook.metadata['execution'] = {
            'method': 'IPython TerminalInteractiveShell.run_cell, sequential single process; actual captured stdout and MIME outputs',
            'python': sys.version,
            'run_train': False,
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        shutil.copy2(path, ROOT / 'notebooks/experiments/Mai_EXP-009_RF_tuning.ipynb')
    print('Notebook preflight passed' if args.preflight else 'Notebook executed and saved with real outputs', flush=True)


if __name__ == '__main__':
    main()
