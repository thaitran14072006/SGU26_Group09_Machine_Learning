"""Continue LR experiments in the planned order."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/lr_workflow/matplotlib_cache'))
sys.path.insert(0, str(ROOT / 'src'))

from wesad.lr_workflow import (folder, freeze, plan, run_one,
                               validate_completed, write_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-runs', type=int, choices=[1, 2], default=1)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    cfg = plan()
    for _ in range(args.max_runs):
        pending = []
        for spec in cfg['experiments']:
            if (folder(spec) / 'acceptance.json').exists():
                validate_completed(spec)
            else:
                pending.append(spec)
        if args.status:
            print(json.dumps({'next_run': pending[0]['run_id'] if pending else None,
                              'remaining': [s['run_id'] for s in pending]}, indent=2))
            return
        if not pending:
            for stage in cfg['order']:
                freeze(stage, cfg)
            write_state(cfg)
            print('WORKFLOW_COMPLETE: all stages verified; proceed to handoff.')
            return
        # The core lock prevents overlap and independently checks prerequisite gates.
        run_one(pending[0]['run_id'])
        validate_completed(pending[0])
    write_state(cfg)


if __name__ == '__main__':
    main()
