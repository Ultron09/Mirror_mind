This folder contains example experiment scripts and configurations for reproducible runs using the MirrorMind framework.

Usage

1. Edit or copy a config from `experiments/configs/`.
2. Run the experiment:

```powershell
python experiments\run_experiment.py --config experiments\configs\sample_config.json --output_dir runs/exp1
```

Output

- `summary.json` in the output directory containing experiment metadata and a short training summary.

Notes

- `run_experiment.py` is a minimal runner intended for quick reproducibility checks. For larger experiments integrate with your dataset loader and logging stack (TensorBoard, WandB, etc.).
