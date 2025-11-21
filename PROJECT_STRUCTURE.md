# Project Structure

Complete structure of the Deep Learning Framework for Time-Series Analysis.

## Directory Tree

```
deep/
├── __init__.py                          # Package root
├── README.md                            # Main documentation
├── QUICKSTART.md                        # Quick start guide
├── PROJECT_STRUCTURE.md                 # This file
├── requirements.txt                     # Dependencies
├── setup.py                             # Package installation
├── .gitignore                          # Git ignore rules
├── config_template.yaml                 # Configuration template
│
├── core/                               # Core framework (80% shared)
│   ├── __init__.py                    # Core package exports
│   │
│   ├── data/                          # Data loading and datasets
│   │   ├── __init__.py
│   │   ├── dataset.py                 # TimeSeriesDataset class
│   │   └── dataloader.py              # TimeSeriesDataLoader class
│   │
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_engine.py          # FeatureEngine orchestrator
│   │   └── transforms.py              # Feature transformers:
│   │                                  #   - RollingWindow
│   │                                  #   - LagFeatures
│   │                                  #   - TechnicalIndicators
│   │                                  #   - DateTimeFeatures
│   │
│   ├── models/                        # Neural network models
│   │   ├── __init__.py
│   │   ├── base_model.py              # BaseTimeSeriesModel
│   │   ├── lstm.py                    # LSTMModel
│   │   ├── gru.py                     # GRUModel
│   │   └── transformer.py             # TransformerModel + PositionalEncoding
│   │
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Trainer class
│   │   └── callbacks.py               # Callbacks:
│   │                                  #   - EarlyStopping
│   │                                  #   - ModelCheckpoint
│   │                                  #   - LearningRateScheduler
│   │
│   ├── validation/                    # Time-aware validation
│   │   ├── __init__.py
│   │   └── time_series_split.py       # TimeSeriesSplit, WalkForwardSplit
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── metrics.py                 # Evaluation metrics (MSE, MAE, RMSE, etc.)
│       └── helpers.py                 # Helper functions (set_seed, get_device)
│
├── config/                            # Configuration management
│   └── config_manager.py              # Config class and utilities
│
├── examples/                          # Example projects (20% custom)
│   ├── stock_prediction/             # Stock price forecasting example
│   │   └── train.py                  # Training script with technical indicators
│   │
│   └── sports_analytics/             # Sports performance prediction
│       ├── custom_features.py        # Custom domain features:
│       │                             #   - PlayerPerformanceFeatures
│       │                             #   - TeamStreakFeatures
│       │                             #   - MatchupFeatures
│       └── train.py                  # Training script
│
└── tests/                            # Unit tests
    └── test_basic.py                 # Basic functionality tests
```

## File Count Summary

- **Core Framework**: 20 Python files
- **Configuration**: 2 files (Python + YAML)
- **Examples**: 3 files
- **Tests**: 1 file
- **Documentation**: 4 files (README, QUICKSTART, PROJECT_STRUCTURE, setup)
- **Total**: ~30 files

## Component Descriptions

### Core Framework Components

#### 1. Data Module (`core/data/`)
- **TimeSeriesDataset**: PyTorch Dataset with windowing and forecasting
- **TimeSeriesDataLoader**: Specialized DataLoader for time-series

#### 2. Features Module (`core/features/`)
- **FeatureEngine**: Orchestrates feature transformation pipeline
- **RollingWindow**: Rolling statistics (mean, std, min, max)
- **LagFeatures**: Lag feature creation
- **TechnicalIndicators**: RSI, MACD, Bollinger Bands for finance
- **DateTimeFeatures**: Cyclical encoding of temporal features

#### 3. Models Module (`core/models/`)
- **BaseTimeSeriesModel**: Abstract base class
- **LSTMModel**: Long Short-Term Memory network
- **GRUModel**: Gated Recurrent Unit network
- **TransformerModel**: Transformer with positional encoding

#### 4. Training Module (`core/training/`)
- **Trainer**: Training loop with validation and metrics
- **EarlyStopping**: Stop when validation loss plateaus
- **ModelCheckpoint**: Save best model weights
- **LearningRateScheduler**: Adjust learning rate during training

#### 5. Validation Module (`core/validation/`)
- **TimeSeriesSplit**: Expanding window cross-validation
- **WalkForwardSplit**: Rolling window validation
- **train_test_split_temporal**: Simple temporal split

#### 6. Utils Module (`core/utils/`)
- **Metrics**: MSE, MAE, RMSE, MAPE, R²
- **Helpers**: set_seed, get_device, count_parameters

### Configuration System

- **Config Class**: YAML/JSON configuration management
- **Default Template**: Pre-configured settings for quick start

### Example Projects

#### Stock Prediction
Demonstrates:
- Technical indicator features
- Financial time-series patterns
- LSTM for sequential prediction

#### Sports Analytics
Demonstrates:
- Custom domain features
- Extending the framework
- GRU model usage

## Usage Patterns

### Basic Usage (Core Only)
```python
from core import TimeSeriesDataset, LSTMModel, Trainer
# Use framework as-is
```

### Extended Usage (Core + Custom)
```python
from core import FeatureEngine, GRUModel
from my_features import CustomFeatures  # Your additions
# Extend with domain-specific logic
```

## Extensibility Points

1. **Custom Features**: Add transformers to `FeatureEngine`
2. **Custom Models**: Inherit from `BaseTimeSeriesModel`
3. **Custom Callbacks**: Implement `on_epoch_begin`/`on_epoch_end`
4. **Custom Metrics**: Add functions to `utils/metrics.py`
5. **Custom Validation**: Extend validation strategies

## Dependencies

Core dependencies (see `requirements.txt`):
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- Pandas ≥ 2.0.0
- scikit-learn ≥ 1.3.0
- PyYAML ≥ 6.0
- tqdm ≥ 4.65.0

## Installation Methods

### Method 1: Direct Installation
```bash
pip install -r requirements.txt
```

### Method 2: Editable Install
```bash
pip install -e .
```

### Method 3: With Development Tools
```bash
pip install -e .[dev]
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_basic.py

# With coverage
pytest --cov=core tests/
```

## Development Workflow

1. **Core Development**: Modify files in `core/`
2. **Project Creation**: Add new directory in `examples/`
3. **Custom Features**: Create feature transformers in your project
4. **Testing**: Add tests in `tests/`
5. **Documentation**: Update README.md

## Key Design Decisions

### Why This Structure?

1. **Separation of Concerns**: Each module has single responsibility
2. **80/20 Rule**: Core provides common functionality, examples show customization
3. **Framework Pattern**: Core is stable, examples demonstrate usage
4. **Testability**: Clear boundaries enable unit testing
5. **Extensibility**: Multiple extension points without modifying core

### Why These Modules?

- **data**: Essential for all time-series tasks
- **features**: Common preprocessing across domains
- **models**: Standard architectures (LSTM, GRU, Transformer)
- **training**: Consistent training interface
- **validation**: Time-awareness is critical for time-series
- **utils**: Shared utilities

### Why PyTorch?

- Flexibility for research and production
- Large ecosystem and community
- Dynamic computation graphs
- Strong GPU support

## Future Additions

Potential new modules:
- `core/losses/`: Custom loss functions
- `core/metrics/`: More evaluation metrics
- `core/augmentation/`: Data augmentation
- `core/explainability/`: Model interpretation
- `core/deployment/`: Production utilities

## Contributing Guidelines

When adding to this project:

1. **Core changes**: Only add truly universal functionality
2. **Examples**: Add new use cases as separate example projects
3. **Tests**: Write tests for all core functionality
4. **Documentation**: Update README and this file
5. **Dependencies**: Minimize new dependencies

## License

MIT License - see LICENSE file for details
