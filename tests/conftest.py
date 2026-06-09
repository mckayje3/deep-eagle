"""Shared pytest fixtures for the test suite."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(autouse=True)
def _seed():
    """Make every test deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def regression_loader():
    """
    A small, genuinely learnable regression dataset:
    target is a fixed linear function of the last timestep's features, so a
    model that trains at all should drive the loss down.
    """
    n, seq, feat = 128, 8, 4
    x = torch.randn(n, seq, feat)
    w = torch.randn(feat, 1)
    y = x[:, -1, :] @ w + 0.01 * torch.randn(n, 1)  # (n, 1)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=False)
