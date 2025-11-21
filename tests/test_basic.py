"""
Basic tests to verify framework functionality

Run with: pytest tests/test_basic.py
"""

import pytest
import numpy as np
import torch


def test_imports():
    """Test that all core components can be imported"""
    from core import (
        TimeSeriesDataset,
        TimeSeriesDataLoader,
        FeatureEngine,
        LSTMModel,
        GRUModel,
        TransformerModel,
        Trainer,
        TimeSeriesSplit,
        WalkForwardSplit,
    )
    assert TimeSeriesDataset is not None
    assert LSTMModel is not None


def test_dataset_creation():
    """Test creating a time-series dataset"""
    from core import TimeSeriesDataset

    # Create synthetic data
    data = np.random.randn(100, 5)
    targets = np.random.randn(100)

    # Create dataset
    dataset = TimeSeriesDataset(
        data=data,
        targets=targets,
        sequence_length=10,
        forecast_horizon=1,
    )

    assert len(dataset) > 0
    assert dataset.n_features == 5
    assert dataset.n_targets == 1

    # Test getting item
    seq, target = dataset[0]
    assert seq.shape == (10, 5)
    assert target.shape == (1,)


def test_model_creation():
    """Test creating models"""
    from core import LSTMModel, GRUModel, TransformerModel

    # LSTM
    lstm = LSTMModel(input_dim=5, hidden_dim=32, output_dim=1)
    assert lstm.input_dim == 5
    assert lstm.hidden_dim == 32
    assert lstm.count_parameters() > 0

    # GRU
    gru = GRUModel(input_dim=5, hidden_dim=32, output_dim=1)
    assert gru.input_dim == 5

    # Transformer
    transformer = TransformerModel(
        input_dim=5,
        hidden_dim=32,
        output_dim=1,
        num_heads=2,
    )
    assert transformer.input_dim == 5


def test_model_forward():
    """Test model forward pass"""
    from core import LSTMModel

    model = LSTMModel(input_dim=5, hidden_dim=32, output_dim=1)

    # Create dummy input
    batch = torch.randn(4, 10, 5)  # (batch, seq_len, features)

    # Forward pass
    output = model(batch)

    assert output.shape == (4, 1)  # (batch, output_dim)


def test_feature_engine():
    """Test feature engineering"""
    from core import FeatureEngine
    from core.features import LagFeatures
    import pandas as pd

    # Create data
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })

    # Create feature engine
    engine = FeatureEngine(
        transformers=[LagFeatures(lags=[1, 2])],
        scaler='standard',
    )

    # Fit and transform
    transformed = engine.fit_transform(data)

    assert transformed.shape[0] == 100
    assert transformed.shape[1] > 2  # Original + lag features
    assert engine.fitted


def test_time_series_split():
    """Test time-series cross-validation"""
    from core import TimeSeriesSplit

    data = np.random.randn(100, 5)

    cv = TimeSeriesSplit(n_splits=3, test_size=10)

    splits = list(cv.split(data))
    assert len(splits) == 3

    # Check that train indices always come before test indices
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)


def test_config_manager():
    """Test configuration management"""
    from config.config_manager import Config

    config_dict = {
        'model': {
            'hidden_dim': 64,
            'dropout': 0.1,
        },
        'training': {
            'learning_rate': 0.001,
        }
    }

    config = Config.from_dict(config_dict)

    # Test get with dot notation
    assert config.get('model.hidden_dim') == 64
    assert config.get('training.learning_rate') == 0.001
    assert config.get('nonexistent.key', default=42) == 42

    # Test set
    config.set('model.hidden_dim', 128)
    assert config.get('model.hidden_dim') == 128


if __name__ == '__main__':
    # Run tests
    test_imports()
    print("✓ Imports test passed")

    test_dataset_creation()
    print("✓ Dataset creation test passed")

    test_model_creation()
    print("✓ Model creation test passed")

    test_model_forward()
    print("✓ Model forward pass test passed")

    test_feature_engine()
    print("✓ Feature engine test passed")

    test_time_series_split()
    print("✓ Time-series split test passed")

    test_config_manager()
    print("✓ Config manager test passed")

    print("\n✓ All tests passed!")
