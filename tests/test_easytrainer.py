import torch
import pytest


def test_easytrainer_smoke():
    """Smoke test: construct EasyTrainer and run a 1-epoch training on tiny synthetic data."""
    try:
        from AGITrainer import EasyTrainer
    except Exception as e:
        pytest.skip(f"AGITrainer import failed: {e}")

    X = torch.randn(16, 4, 32)
    y = torch.randn(16, 4, 32)

    trainer = EasyTrainer(device='cpu')
    summary = trainer.train(X, y, epochs=1, batch_size=8)

    assert isinstance(summary, dict), "Trainer summary should be a dict"
    assert 'best_train_loss' in summary or 'train_loss' in summary or 'loss' in summary, "Expected loss key in summary"
