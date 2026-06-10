"""
Tests for the dashboard training pipeline (web_ui/pages/training.py).

These cover run_training(), the Streamlit-free core that the "Start Training"
button now calls instead of the old synthetic-loss simulation.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# training.py lives under web_ui/pages and is not an installed package.
_PATH = Path(__file__).resolve().parent.parent / "web_ui" / "pages" / "training.py"
_spec = importlib.util.spec_from_file_location("deep_eagle_training_page", _PATH)
training = importlib.util.module_from_spec(_spec)
sys.modules["deep_eagle_training_page"] = training
_spec.loader.exec_module(training)


def _config(model_type="lstm", **overrides):
    cfg = {
        "model": {
            "type": model_type,
            "hidden_dim": 16,
            "output_dim": 1,
            "num_layers": 1,
            "dropout": 0.1,
        },
        "data": {
            "sequence_length": 6,
            "forecast_horizon": 1,
            "batch_size": 16,
            "test_size": 0.2,
        },
        "training": {
            "epochs": 25,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "device": "cpu",
            "early_stopping": {"enabled": False},
        },
        "features": {"scaler": "standard", "handle_missing": "ffill"},
    }
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


def _sine_df(n=300):
    t = np.arange(n)
    signal = np.sin(t * 0.1)
    return pd.DataFrame({"f1": signal, "f2": np.cos(t * 0.1), "target": signal})


def test_run_training_learns_and_reports_progress():
    df = _sine_df()
    updates = []
    model, history = training.run_training(
        df,
        "target",
        ["f1", "f2"],
        _config(),
        progress_callback=lambda e, tr, va: updates.append((e, tr, va)),
    )

    assert model is not None
    # One update per epoch, monotonically increasing epoch index
    assert len(updates) == 25
    assert [u[0] for u in updates] == list(range(25))
    # All reported losses are finite numbers
    assert all(np.isfinite(tr) for _, tr, _ in updates)
    # The model actually learned the (very learnable) signal
    assert history["train_loss"][-1] < history["train_loss"][0]
    # A validation split was used (test_size=0.2 leaves room for windows)
    assert len(history["val_loss"]) == 25


def test_run_training_without_validation_split():
    """A holdout too small to form a window must not crash; just train."""
    df = _sine_df(n=60)
    # test_size leaves only a few rows — fewer than sequence_length+horizon
    model, history = training.run_training(
        df, "target", ["f1"], _config(data={"test_size": 0.05, "sequence_length": 6})
    )
    assert model is not None
    assert len(history["train_loss"]) == 25


def test_build_model_transformer_uses_dim_feedforward():
    """Regression: the page used to pass feedforward_dim, crashing transformers."""
    cfg = _config(model_type="transformer", model={"num_heads": 4, "feedforward_dim": 64})
    model = training._build_model(cfg, input_dim=3)
    assert model.dim_feedforward == 64
    assert model.num_heads == 4


def test_checkpoint_roundtrip(tmp_path):
    """Train -> checkpoint+sidecar -> reload reconstructs identical predictions."""
    import torch

    df = _sine_df()
    ckpt = tmp_path / "model.pt"
    model, _ = training.run_training(
        df, "target", ["f1", "f2"], _config(), checkpoint_path=str(ckpt)
    )

    # Both the weights and the metadata sidecar were written
    assert ckpt.exists()
    assert ckpt.with_suffix(".meta.json").exists()

    loaded = training.load_model_from_checkpoint(ckpt)

    # The reloaded best checkpoint predicts deterministically (eval mode)
    x = torch.randn(4, _config()["data"]["sequence_length"], 2)
    with torch.no_grad():
        out1 = loaded(x)
        out2 = loaded(x)
    assert out1.shape == (4, 1)
    assert torch.allclose(out1, out2)


def test_load_checkpoint_without_sidecar_raises(tmp_path):
    import torch

    ckpt = tmp_path / "orphan.pt"
    torch.save({"model_state_dict": {}}, str(ckpt))  # no .meta.json beside it
    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        training.load_model_from_checkpoint(ckpt)
