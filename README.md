# MirrorMind

Overview
--------
MirrorMind is a modular research codebase for experimenting with adaptive meta-learning and training-time adaptation strategies. It provides a lightweight platform for implementing, testing, and comparing adaptation methods that operate at training time (e.g., dynamic learning-rate control, gradient-statistics monitoring, targeted weight adaptation, and curriculum scheduling).

Project goals
- Provide a clear separation between the base learner and an adaptation/meta-controller to support reproducible ablations.
- Supply small, runnable experiments and configuration-driven runners to make results easy to reproduce.
- Offer guidance for packaging the project as a Python package and preparing reproducibility artifacts for reviewers.

Repository layout
- `mirror_mind_agi/` — core implementation (trainer, adaptation modules, example models).
- `experiments/` — reproducible experiment runners and sample configs.
- `tests/` — lightweight pytest-based smoke tests.
- `IMPLEMENTATION_GUIDE.md` — developer-facing architecture and extension points.
- `ETHICS.md` — limitations and responsible-use guidance.

Principles
----------
Public messaging in this repo follows conservative, research-first principles: avoid unverifiable claims, include exact commands to reproduce experiments, and prefer small, well-documented runs over opaque large-scale results.

Quick links
- Runner: `experiments/run_experiment.py`
- Sample config: `experiments/configs/sample_config.json`
- Smoke test: `tests/test_easytrainer.py`

Installation
------------
Create an isolated Python environment and install the project in editable mode for development:

PowerShell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Quickstart
----------
Run a small synthetic example to validate the environment and demonstrate the experiment flow:

PowerShell
```
python experiments/run_experiment.py --config experiments/configs/sample_config.json --output_dir runs/demo
```

Reproducibility checklist
-------------------------
For any published experiment include:

1. Git commit SHA used for the run.
2. `requirements.txt` or `pyproject.toml` with pinned versions.
3. Configuration file used for the run (JSON/YAML).
4. Random seeds for PyTorch, NumPy, and Python `random`.
5. Exact command(s) or script used to run the experiment.
6. A short README in the `runs/<exp>` folder describing how to reproduce the result.

Reporting guidance
------------------
- Use paired ablations (with vs. without adaptation) and report mean  standard deviation across multiple seeds (N5 recommended for initial claims).
- Report task-specific metrics and experimental protocol clearly. Avoid product-style or global intelligence claims.
- Include negative results and limitations; documenting failure modes improves scientific value.

Packaging for PyPI (maintainer guidance)
---------------------------------------
To publish MirrorMind as a Python package:

1. Add a `pyproject.toml` specifying the build system and metadata (e.g., `setuptools` / `wheel`).
2. Choose a stable package name (e.g., `mirror_mind`) and maintain a single source of truth for `__version__`.
3. Build and sanity-check distributions:

```
python -m build
python -m pip install --upgrade twine
python -m twine check dist/*
```

4. Upload to Test PyPI for verification:

```
python -m twine upload --repository testpypi dist/*
```

Contributing
------------
Contributions are welcome. Recommended process:

1. Open an issue describing the proposed change and rationale.
2. Create small, focused pull requests with tests and example configs.
3. Add documentation (notebook or README) showing how to run the new experiment.

Ethics and limitations
----------------------
See `ETHICS.md` for guidance on responsible usage. Keep public messaging measured and avoid unverifiable claims.

Acknowledgements
----------------
This project collects experiments and code developed by contributors; see the Git history for detailed authorship.
