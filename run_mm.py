"""Run MirrorMind tasks with a one-liner.

Examples:
  python run_mm.py                # runs auto_best (sweep + best Phase7 run)
  python run_mm.py --mode sweep    # run sweep only
  python run_mm.py --mode phase7 --config '{"adapter_lr":0.005,"ewc_lambda":1.0,"noise_sigma":0.02}'

This script imports `airbornehrs.cli` and provides a single-file entrypoint that can be
invoked from a virtualenv or system Python without additional setup.
"""
from airbornehrs import cli
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['auto_best', 'sweep', 'phase7'], default='auto_best')
parser.add_argument('--config', type=str, help='JSON config for phase7')
parser.add_argument('--task_memory', type=str, help='path to task memory file')
args = parser.parse_args()

if args.mode == 'sweep':
    rc = cli.sweep()
    sys.exit(rc)
elif args.mode == 'phase7':
    cfg = json.loads(args.config) if args.config else None
    rc = cli.run_phase7(cfg, args.task_memory)
    sys.exit(rc)
else:
    rc = cli.auto_best()
    sys.exit(rc)
