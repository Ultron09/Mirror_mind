"""
Simple CLI helpers for MirrorMind package.
Functions:
 - auto_best(): run sweep (if needed) then full Phase7 with best config
 - sweep(): run the sweep script
 - run_phase7(config=None, task_memory=None): run Phase7 with given config

This module is intentionally lightweight — it wraps existing scripts so the user
has a programmatic API to call from a one-liner wrapper script or from Python.
"""
from pathlib import Path
import subprocess
import sys
import os
import json

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
SWEEP_SCRIPT = SCRIPTS / 'sweep_phase7.py'
AUTO_SCRIPT = SCRIPTS / 'auto_best.py'
EXPERIMENT = ROOT / 'experiments' / 'protocol_v1' / 'phase7_sota_deathmatch.py'


def sweep():
    py = ROOT / 'test_env' / 'Scripts' / 'python.exe'
    if not py.exists():
        py = sys.executable
    print('Running sweep...')
    return subprocess.run([str(py), str(SWEEP_SCRIPT)], cwd=str(ROOT)).returncode


def run_phase7(config: dict = None, task_memory: str = None):
    py = ROOT / 'test_env' / 'Scripts' / 'python.exe'
    if not py.exists():
        py = sys.executable

    env = os.environ.copy()
    if config is not None:
        env['MM_ADAPTER_LR'] = str(config.get('adapter_lr', ''))
        env['MM_EWC_LAMBDA'] = str(config.get('ewc_lambda', ''))
        env['MM_NOISE_SIGMA'] = str(config.get('noise_sigma', ''))
    if task_memory:
        env['MM_TASK_MEMORY'] = str(task_memory)

    print(f"Running Phase7 with config={config} task_memory={task_memory}")
    return subprocess.run([str(py), str(EXPERIMENT)], cwd=str(ROOT), env=env).returncode


def auto_best():
    # Prefer using the existing auto_best script if present (keeps behavior consistent)
    if AUTO_SCRIPT.exists():
        py = ROOT / 'test_env' / 'Scripts' / 'python.exe'
        if not py.exists():
            py = sys.executable
        print('Running auto_best script...')
        return subprocess.run([str(py), str(AUTO_SCRIPT)], cwd=str(ROOT)).returncode

    # Fallback: run sweep then run phase7 using sweep_results.json
    sweep_rc = sweep()
    if sweep_rc != 0:
        return sweep_rc

    # Load sweep_results.json
    sr = ROOT / 'sweep_results.json'
    if not sr.exists():
        print('No sweep_results.json found after sweep.')
        return 2
    data = json.loads(sr.read_text())
    best = data.get('best_config')
    # find checkpoint if saved
    ckpt_dir = ROOT / 'checkpoints'
    ckpts = list(ckpt_dir.glob('mm_best_*.pt')) if ckpt_dir.exists() else []
    tm = str(ckpts[-1]) if ckpts else None
    return run_phase7(best, tm)


if __name__ == '__main__':
    # allow quick CLI use
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['auto_best', 'sweep', 'phase7'], default='auto_best')
    p.add_argument('--config', type=str, help='JSON string config for phase7')
    p.add_argument('--task_memory', type=str, help='path to task memory file')
    args = p.parse_args()

    if args.mode == 'sweep':
        sys.exit(sweep())
    elif args.mode == 'phase7':
        cfg = json.loads(args.config) if args.config else None
        sys.exit(run_phase7(cfg, args.task_memory))
    else:
        sys.exit(auto_best())


def main(argv=None):
    """Console entrypoint for setuptools `console_scripts`.
    Accepts argv (list) for easy testing. If None, uses sys.argv[1:].
    """
    import argparse
    p = argparse.ArgumentParser(prog='mm')
    p.add_argument('--mode', choices=['auto_best', 'sweep', 'phase7'], default='auto_best')
    p.add_argument('--config', type=str, help='JSON string config for phase7')
    p.add_argument('--task_memory', type=str, help='path to task memory file')
    args = p.parse_args(argv)

    if args.mode == 'sweep':
        return sweep()
    elif args.mode == 'phase7':
        cfg = json.loads(args.config) if args.config else None
        return run_phase7(cfg, args.task_memory)
    else:
        return auto_best()
