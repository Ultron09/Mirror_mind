import argparse
import json
import os
import random
import datetime

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_synthetic(data_cfg):
    n = data_cfg.get("num_samples", 256)
    seq = data_cfg.get("seq_len", 10)
    feat = data_cfg.get("feature_dim", 128)
    X = torch.randn(n, seq, feat)
    y = torch.randn(n, seq, feat)
    return X, y


def run(config_path: str, output_dir: str = None):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 0)
    set_seed(seed)

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU")
        device = "cpu"

    # Simple synthetic experiment using EasyTrainer
    data_cfg = cfg.get("data", {})
    if data_cfg.get("use_synthetic", True):
        X, y = make_synthetic(data_cfg)
    else:
        raise RuntimeError("No data loader implemented for non-synthetic data in this sample script.")

    # Lazy import of EasyTrainer to avoid import failure when not installed
    try:
        from AGITrainer import EasyTrainer
    except Exception as e:
        raise RuntimeError(f"Failed to import AGITrainer.EasyTrainer: {e}")

    trainer = EasyTrainer(device=device)

    epochs = cfg.get("training", {}).get("epochs", 3)
    batch_size = cfg.get("training", {}).get("batch_size", 32)

    summary = trainer.train(X, y, epochs=epochs, batch_size=batch_size)

    out = {
        "experiment_name": cfg.get("experiment_name", "unnamed"),
        "commit": None,
        "seed": seed,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(config_path),
        "summary": summary
    }

    if output_dir is None:
        output_dir = os.path.join("runs", cfg.get("experiment_name", "run"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved summary to {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to JSON config")
    p.add_argument("--output_dir", default=None, help="Directory to save outputs")
    args = p.parse_args()
    run(args.config, args.output_dir)
