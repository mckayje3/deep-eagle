"""
Deep Learning Framework for Time-Series Analysis
"""

__version__ = "0.1.0"

# Import core components for easier access
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

__all__ = [
    "TimeSeriesDataset",
    "TimeSeriesDataLoader",
    "FeatureEngine",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "Trainer",
    "TimeSeriesSplit",
    "WalkForwardSplit",
]
