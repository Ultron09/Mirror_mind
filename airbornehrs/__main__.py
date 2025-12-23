"""Module entrypoint so users can run: python -m airbornehrs
This calls the same CLI wrapper as `run_mm.py`.
"""
from . import cli
import sys

if __name__ == '__main__':
    sys.exit(cli.auto_best())
