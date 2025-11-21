# Deep Learning Framework for Time-Series Analysis

A modular, extensible PyTorch-based deep learning framework designed for time-series data analysis. Built with 80% core functionality shared across projects and 20% customizable for domain-specific needs.

## Features

### Core Functionality (Shared)

- **Data Loading**: Time-series specific datasets with windowing and forecasting support
- **Feature Engineering**:
  - Rolling window statistics
  - Lag features
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Datetime features with cyclical encoding
  - Extensible transformer pipeline
- **Models**:
  - LSTM
  - GRU
  - Transformer with positional encoding
- **Training**:
  - Flexible trainer with callbacks
  - Early stopping
  - Model checkpointing
  - Learning rate scheduling
- **Validation**:
  - Time-aware cross-validation
  - Walk-forward validation
  - Temporal train/test split
- **Configuration**: YAML/JSON based configuration management
- **Web UI**: Comprehensive dashboard for visual model building and training

### Project Structure

```
deep/
├── core/                      # Core framework (80% shared)
│   ├── data/                 # Data loading and datasets
│   ├── features/             # Feature engineering
│   ├── models/               # Neural network models
│   ├── training/             # Training loops and callbacks
│   ├── validation/           # Time-series cross-validation
│   └── utils/                # Metrics and helpers
├── config/                   # Configuration management
├── web_ui/                   # Web dashboard interface
│   ├── pages/               # Dashboard pages
│   └── utils/               # UI utilities
├── tools/                    # Command-line tools
├── examples/                 # Example projects
│   ├── stock_prediction/    # Stock price forecasting
│   └── sports_analytics/    # Player performance prediction
└── tests/                    # Unit tests
```

## Installation

### Install from GitHub (Recommended for cloud deployments)

```bash
# Install directly from GitHub
pip install git+https://github.com/mckayje3/deep-eagle.git
```

For Streamlit Cloud or other cloud platforms, add this line to your `requirements.txt`:
```
git+https://github.com/mckayje3/deep-eagle.git
```

### Install from source (For local development)

```bash
# Clone the repository
git clone https://github.com/mckayje3/deep-eagle.git
cd deep-eagle

# Install in editable mode
pip install -e .
```

## Quick Start

### Using Core Framework

```python
import torch
from core import (
    TimeSeriesDataset,
    TimeSeriesDataLoader,
    LSTMModel,
    Trainer
)
from core.utils import set_seed

# Set random seed
set_seed(42)

# Create dataset
dataset = TimeSeriesDataset(
    data=your_data,           # (N, features)
    targets=your_targets,     # (N,)
    sequence_length=30,       # Look back 30 steps
    forecast_horizon=1,       # Predict 1 step ahead
)

# Create data loader
loader = TimeSeriesDataLoader(dataset, batch_size=32)

# Create model
model = LSTMModel(
    input_dim=dataset.n_features,
    hidden_dim=64,
    output_dim=1,
    num_layers=2,
)

# Train
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

trainer = Trainer(model, optimizer, criterion, device='cuda')
history = trainer.fit(loader, epochs=50)
```

### Using Configuration Files

```python
from config.config_manager import Config, save_default_config

# Generate default configuration
save_default_config('config.yaml')

# Load and modify
config = Config.from_yaml('config.yaml')
config.set('model.hidden_dim', 128)
config.set('training.learning_rate', 0.0001)

# Access values
batch_size = config.get('data.batch_size', default=32)
```

### Using the Web Dashboard

For a visual, no-code interface:

```bash
cd web_ui
pip install -r requirements.txt
streamlit run app.py
```

The dashboard provides:
- 📊 **Dataset Manager** - Upload and visualize your data
- 🏗️ **Model Builder** - Configure models visually with presets
- 🚀 **Training** - Monitor training in real-time
- 📈 **Results** - Evaluate performance with interactive charts
- 🔮 **Prediction** - Make predictions on new data
- 🔍 **Project Scanner** - Analyze usage across projects

See `web_ui/README.md` for detailed documentation.

## Example Projects

### 1. Stock Price Prediction

Demonstrates using technical indicators for financial forecasting:

```bash
cd examples/stock_prediction
python train.py
```

Features:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Rolling window features
- Lag features
- LSTM model

### 2. Sports Analytics

Shows how to extend the framework with custom features:

```bash
cd examples/sports_analytics
python train.py
```

Features:
- Custom player performance metrics
- Team streak tracking
- Rolling performance averages
- Momentum indicators
- GRU model

## Extending the Framework

### Adding Custom Features

Create custom feature transformers by implementing `transform()` and `fit_transform()` methods:

```python
class CustomFeatures:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def transform(self, data):
        # Your feature engineering logic
        result = data.copy()
        result['custom_feature'] = ...
        return result

    def fit_transform(self, data):
        return self.transform(data)

# Use in pipeline
from core import FeatureEngine

engine = FeatureEngine(
    transformers=[CustomFeatures(...)],
    scaler='standard'
)
```

### Creating Custom Models

Extend `BaseTimeSeriesModel` for consistent interface:

```python
from core.models import BaseTimeSeriesModel
import torch.nn as nn

class CustomModel(BaseTimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)

        # Your architecture
        self.layers = nn.Sequential(...)

    def forward(self, x):
        # Your forward pass
        return self.layers(x)
```

### Adding Custom Callbacks

```python
class CustomCallback:
    def on_epoch_begin(self, epoch, trainer):
        # Called at start of epoch
        pass

    def on_epoch_end(self, epoch, trainer, train_results, val_results):
        # Called at end of epoch
        # Return True to stop training
        return False

# Use in trainer
trainer = Trainer(model, optimizer, criterion, callbacks=[CustomCallback()])
```

## Usage Analytics & Tracking

The framework includes built-in analytics to track feature usage across projects and manage deprecations.

### Usage Analytics

Track which features are being used in your projects:

```python
from core import track_usage, get_usage_stats, clear_usage_stats

# Manually track feature usage (optional - automatic tracking available)
track_usage('LSTMModel')
track_usage('FeatureEngine.rolling_window')

# View usage statistics
stats = get_usage_stats()
print(stats)
# {'LSTMModel': 15, 'FeatureEngine.rolling_window': 8, ...}

# Clear statistics
clear_usage_stats()
```

Usage tracking is enabled by default and stores data in `~/.deep-timeseries/usage_stats.json`. To disable:

```bash
export DEEP_TRACK_USAGE=false
```

### Deprecation Management

Mark features as deprecated to help users migrate smoothly:

```python
from core.utils import deprecated, deprecate, deprecated_argument

# Deprecate a function
@deprecated("Too slow", replacement="fast_function", removal_version="0.3.0")
def old_function():
    pass

# Deprecate a function argument
@deprecated_argument("old_param", new_name="new_param", removal_version="0.3.0")
def my_function(new_param=None, old_param=None):
    if old_param is not None:
        new_param = old_param
    return new_param

# Issue a deprecation warning manually
deprecate("This approach is deprecated, use X instead", removal_version="0.3.0")
```

### Scanning Projects for Usage

Use the dependency scanner to analyze how deep-timeseries is being used:

```bash
# Scan current project
python tools/scan_usage.py

# Scan specific project
python tools/scan_usage.py /path/to/project

# Generate JSON report
python tools/scan_usage.py --format json --output report.json

# Check installed version
python tools/scan_usage.py --version
```

The scanner detects:
- Which classes and functions are being imported
- Frequency of usage across files
- Module import patterns
- Installed package version

See `tools/README.md` for detailed scanner documentation.

## Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Extensibility**: Easy to add custom features, models, and callbacks
3. **Time-Awareness**: All validation respects temporal order
4. **Configuration-Driven**: Projects can be configured via YAML/JSON
5. **Production-Ready**: Includes checkpointing, early stopping, and monitoring
6. **Observable**: Built-in analytics track usage patterns across projects

## Common Use Cases

### Financial Markets
- Stock price prediction
- Volatility forecasting
- Portfolio optimization signals

### Sports Analytics
- Player performance prediction
- Game outcome forecasting
- Injury risk assessment

### Energy & Utilities
- Load forecasting
- Price prediction
- Demand planning

### Healthcare
- Patient vitals monitoring
- Disease progression modeling
- Treatment response prediction

## Configuration Reference

### Model Configuration

```yaml
model:
  type: lstm              # lstm, gru, or transformer
  hidden_dim: 64
  output_dim: 1
  num_layers: 2
  dropout: 0.1
  bidirectional: false    # For LSTM/GRU
  num_heads: 4           # For Transformer
```

### Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  device: auto            # auto, cuda, or cpu
  early_stopping:
    enabled: true
    patience: 10
```

### Feature Engineering

```yaml
features:
  scaler: standard        # standard, minmax, robust
  handle_missing: ffill   # ffill, bfill, drop, zero
  technical_indicators:
    enabled: true
    include_rsi: true
    include_macd: true
```

## Performance Tips

1. **GPU Usage**: Set `device: 'cuda'` in config for GPU acceleration
2. **Batch Size**: Larger batches are faster but use more memory
3. **Sequence Length**: Longer sequences capture more patterns but train slower
4. **Feature Scaling**: Always scale features for better convergence
5. **Early Stopping**: Prevents overfitting and saves training time

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce sequence length
- Reduce model hidden dimension
- Use gradient accumulation

### Poor Performance
- Check feature scaling
- Increase model capacity
- Add more features
- Adjust learning rate
- Increase training epochs

### Overfitting
- Add dropout
- Reduce model size
- Add more training data
- Use early stopping

## Contributing

When adding new projects:
1. Use the core framework for common functionality
2. Add custom features in your project directory
3. Document domain-specific components
4. Follow the example project structure

## License

MIT License - see LICENSE file for details

## Future Enhancements

- [ ] Multi-step forecasting decoder
- [ ] Attention visualization
- [ ] AutoML hyperparameter search
- [ ] Distributed training support
- [ ] ONNX export for deployment
- [ ] Streaming inference pipeline
- [ ] Git-based cross-project usage analysis
- [ ] Automated changelog and migration guide generation
