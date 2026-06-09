"""Tests for ensemble methods, especially the rebuilt stacking meta-learner."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core import GRUModel, LSTMModel
from core.models import EnsembleModel, optimize_ensemble_weights


def _base_models(output_dim=1, forecast_horizon=1):
    return [
        LSTMModel(4, 16, output_dim, forecast_horizon=forecast_horizon),
        GRUModel(4, 16, output_dim, forecast_horizon=forecast_horizon),
    ]


@pytest.mark.parametrize("method", ["average", "weighted", "learned"])
def test_simple_methods_shape(method):
    ens = EnsembleModel(_base_models(), method=method)
    out = ens(torch.randn(8, 10, 4))
    assert out.shape == (8, 1)


@pytest.mark.parametrize(
    ("output_dim", "forecast_horizon", "expected"),
    [(1, 1, (8, 1)), (2, 1, (8, 2)), (1, 4, (8, 4, 1)), (3, 2, (8, 2, 3))],
)
def test_stacking_handles_multi_output_and_horizon(output_dim, forecast_horizon, expected):
    """These exact shapes used to crash the old stacking implementation."""
    ens = EnsembleModel(_base_models(output_dim, forecast_horizon), method="stacking")
    out = ens(torch.randn(8, 10, 4))
    assert out.shape == expected


def test_learned_weights_broadcast_multi_dim():
    ens = EnsembleModel(_base_models(output_dim=2, forecast_horizon=3), method="learned")
    out = ens(torch.randn(5, 10, 4))
    assert out.shape == (5, 3, 2)


def test_fit_meta_learner_reduces_loss():
    torch.manual_seed(0)
    models = _base_models()
    ens = EnsembleModel(models, method="stacking")

    x = torch.randn(64, 10, 4)
    y = torch.randn(64, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    criterion = nn.MSELoss()

    before = criterion(ens(x), y).item()
    ens.fit_meta_learner(loader, epochs=80, learning_rate=0.05)
    after = criterion(ens(x), y).item()

    assert after < before


def test_fit_meta_learner_only_on_stacking():
    ens = EnsembleModel(_base_models(), method="average")
    with pytest.raises(ValueError, match="stacking"):
        ens.fit_meta_learner([], epochs=1)


def test_optimize_learned_weights_runs():
    ens = EnsembleModel(_base_models(), method="learned")
    x = torch.randn(32, 10, 4)
    y = torch.randn(32, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    before = ens.get_weights().copy()
    optimize_ensemble_weights(ens, loader, nn.MSELoss(), n_iterations=10, learning_rate=0.1)
    after = ens.get_weights()

    # Weights still form a valid simplex after optimization
    assert abs(after.sum() - 1.0) < 1e-5
    assert after.shape == before.shape


def test_ensemble_requires_two_models():
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleModel([LSTMModel(4, 8, 1)], method="average")
