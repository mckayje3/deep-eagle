# Quick Start Guide

Get up and running with the Deep Time-Series Framework in 5 minutes!

## Installation

```bash
# Install the framework
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Your First Model in 3 Steps

### Step 1: Prepare Your Data

```python
import numpy as np
from core import TimeSeriesDataset, TimeSeriesDataLoader

# Your time-series data (shape: [samples, features])
data = np.random.randn(1000, 5)
targets = np.random.randn(1000)

# Create dataset
dataset = TimeSeriesDataset(
    data=data,
    targets=targets,
    sequence_length=30,    # Use last 30 timesteps
    forecast_horizon=1,    # Predict 1 step ahead
)

# Create data loader
train_loader = TimeSeriesDataLoader(dataset, batch_size=32)
```

### Step 2: Build a Model

```python
from core import LSTMModel

model = LSTMModel(
    input_dim=5,      # Number of features
    hidden_dim=64,    # Hidden layer size
    output_dim=1,     # Prediction dimension
    num_layers=2,     # Stack 2 LSTM layers
    dropout=0.1,      # Dropout for regularization
)
```

### Step 3: Train

```python
import torch
from core import Trainer
from core.training import EarlyStopping

# Setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',  # or 'cpu'
    callbacks=[EarlyStopping(patience=10)]
)

# Train!
history = trainer.fit(train_loader, epochs=50)
```

That's it! You've trained your first time-series model.

## Next Steps

### Add Feature Engineering

```python
from core import FeatureEngine
from core.features import RollingWindow, LagFeatures

# Create feature pipeline
feature_engine = FeatureEngine(
    transformers=[
        RollingWindow(windows=[5, 10], functions=['mean', 'std']),
        LagFeatures(lags=[1, 2, 3]),
    ],
    scaler='standard'
)

# Transform your data
transformed_data = feature_engine.fit_transform(raw_data)
```

### Use Configuration Files

```python
from config.config_manager import Config, save_default_config

# Generate template
save_default_config('my_config.yaml')

# Load and use
config = Config.from_yaml('my_config.yaml')
batch_size = config.get('data.batch_size')
```

### Try Different Models

```python
from core import GRUModel, TransformerModel

# GRU (faster than LSTM)
model = GRUModel(input_dim=5, hidden_dim=64, output_dim=1)

# Transformer (for complex patterns)
model = TransformerModel(
    input_dim=5,
    hidden_dim=64,
    output_dim=1,
    num_heads=4,
)
```

### Add Time-Aware Validation

```python
from core import TimeSeriesSplit

# Create time-aware cross-validation
cv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in cv.split(data):
    train_data = data[train_idx]
    val_data = data[val_idx]
    # Train and validate...
```

## Example Projects

### Stock Prediction
```bash
cd examples/stock_prediction
python train.py
```

### Sports Analytics
```bash
cd examples/sports_analytics
python train.py
```

## Common Patterns

### Pattern 1: Simple Forecasting

```python
# 1. Create dataset with your data
dataset = TimeSeriesDataset(data, targets, sequence_length=30)

# 2. Build model
model = LSTMModel(input_dim=features, hidden_dim=64, output_dim=1)

# 3. Train
trainer = Trainer(model, optimizer, criterion)
trainer.fit(loader, epochs=50)

# 4. Predict
predictions = trainer.predict(test_loader)
```

### Pattern 2: With Feature Engineering

```python
# 1. Engineer features
feature_engine = FeatureEngine(transformers=[...], scaler='standard')
train_features = feature_engine.fit_transform(train_data)
test_features = feature_engine.transform(test_data)

# 2. Rest is the same
dataset = TimeSeriesDataset(train_features, targets, ...)
model = LSTMModel(...)
trainer = Trainer(...)
```

### Pattern 3: Custom Features

```python
# 1. Define your transformer
class MyFeatures:
    def transform(self, data):
        result = data.copy()
        result['my_feature'] = ...  # Your logic
        return result

    def fit_transform(self, data):
        return self.transform(data)

# 2. Add to pipeline
feature_engine = FeatureEngine(
    transformers=[MyFeatures(), RollingWindow(...)],
    scaler='standard'
)
```

## Tips for Success

1. **Always scale your features** - Use `scaler='standard'` in FeatureEngine
2. **Start simple** - Begin with LSTM before trying Transformer
3. **Use GPU** - Set `device='cuda'` for 10x+ speedup
4. **Monitor validation** - Always pass `val_loader` to trainer.fit()
5. **Save checkpoints** - Use ModelCheckpoint callback
6. **Respect temporal order** - Never shuffle time-series data

## Troubleshooting

**Q: "RuntimeError: CUDA out of memory"**
- Reduce batch_size (try 16 or 8)
- Reduce sequence_length
- Reduce hidden_dim

**Q: "Model not learning (loss not decreasing)"**
- Check if features are scaled
- Try different learning rate (0.0001 or 0.01)
- Increase model capacity (hidden_dim)

**Q: "Validation loss worse than training"**
- Add dropout
- Use early stopping
- Check for data leakage

## Getting Help

1. Check the [README](README.md) for detailed documentation
2. Look at example projects in `examples/`
3. Review configuration templates in `config/`

## What's Next?

- Read the [full documentation](README.md)
- Explore [example projects](examples/)
- Build your own project!

Happy modeling! 🚀
