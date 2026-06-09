"""Tests for the Trainer: training reduces loss, and checkpoint round-trips."""

import pytest
import torch
import torch.nn as nn

from core import LSTMModel, Trainer
from core.utils import safe_torch_load


def _make_trainer(model):
    return Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        criterion=nn.MSELoss(),
        device="cpu",
    )


def test_fit_reduces_loss(regression_loader):
    model = LSTMModel(input_dim=4, hidden_dim=16, output_dim=1)
    trainer = _make_trainer(model)

    initial = trainer.validate(regression_loader)["loss"]
    history = trainer.fit(regression_loader, epochs=15, verbose=False)
    final = history["train_loss"][-1]

    assert final < initial, f"training did not reduce loss ({initial:.4f} -> {final:.4f})"


def test_checkpoint_roundtrip(regression_loader, tmp_path):
    model = LSTMModel(input_dim=4, hidden_dim=16, output_dim=1)
    trainer = _make_trainer(model)
    trainer.fit(regression_loader, epochs=3, verbose=False)

    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(path))

    # Fresh trainer loads the checkpoint and resumes identical state
    model2 = LSTMModel(input_dim=4, hidden_dim=16, output_dim=1)
    trainer2 = _make_trainer(model2)
    trainer2.load_checkpoint(str(path))

    assert trainer2.current_epoch == trainer.current_epoch
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2)


class _LegacyObject:
    """Module-level (picklable) stand-in for a non-tensor object in a checkpoint."""


def test_legacy_checkpoint_raises_clear_error(tmp_path):
    """A checkpoint with a non-tensor object must fail loudly, not silently."""
    path = tmp_path / "legacy.pt"
    torch.save({"obj": _LegacyObject()}, str(path))

    with pytest.raises(RuntimeError, match="predates the security upgrade"):
        safe_torch_load(str(path))
