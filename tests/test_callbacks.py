"""Tests for training callbacks."""

import torch
import torch.nn as nn

from core import LSTMModel, Trainer
from core.training.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint


def test_early_stopping_triggers_on_patience():
    es = EarlyStopping(patience=2, mode="min", verbose=False)

    # Improving epoch: no stop, counter resets
    assert es.on_epoch_end(0, None, {"loss": 1.0}, {"loss": 1.0}) is False
    # No improvement for `patience` epochs -> stop on the patience-th
    assert es.on_epoch_end(1, None, {"loss": 1.0}, {"loss": 2.0}) is False
    assert es.on_epoch_end(2, None, {"loss": 1.0}, {"loss": 2.0}) is True
    assert es.early_stop is True


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, mode="min", verbose=False)
    es.on_epoch_end(0, None, {"loss": 1.0}, {"loss": 1.0})
    es.on_epoch_end(1, None, {"loss": 1.0}, {"loss": 2.0})  # counter -> 1
    es.on_epoch_end(2, None, {"loss": 1.0}, {"loss": 0.5})  # improvement -> reset
    assert es.counter == 0


def test_model_checkpoint_saves_best_only(tmp_path):
    model = LSTMModel(input_dim=4, hidden_dim=8, output_dim=1)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.MSELoss(),
    )
    path = tmp_path / "best.pt"
    ckpt = ModelCheckpoint(str(path), monitor="val_loss", mode="min", verbose=False)

    ckpt.on_epoch_end(0, trainer, {"loss": 1.0}, {"loss": 1.0})
    assert path.exists()
    mtime_first = path.stat().st_mtime_ns

    # Worse score -> should NOT overwrite
    ckpt.on_epoch_end(1, trainer, {"loss": 1.0}, {"loss": 2.0})
    assert path.stat().st_mtime_ns == mtime_first


def test_lr_scheduler_steps():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    class _Trainer:
        pass

    t = _Trainer()
    t.optimizer = optimizer
    cb = LearningRateScheduler(scheduler, verbose=False)

    optimizer.step()  # step optimizer before scheduler (avoids ordering warning)
    cb.on_epoch_end(0, t, {"loss": 1.0}, {"loss": 1.0})
    assert optimizer.param_groups[0]["lr"] == 0.05
