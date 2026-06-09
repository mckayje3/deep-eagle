# Deep Eagle - Time-Series Deep Learning Framework

PyTorch-based framework for time-series prediction with modular models (LSTM, GRU, Transformer, Ensemble), feature engineering, and a Streamlit dashboard.

> See `~/.claude/CLAUDE.md` for Python/SQL coding standards.

---

## Key Commands

```bash
# Install
pip install -e .

# Run web dashboard
cd web_ui && streamlit run app.py
# Default login: admin / admin123

# Run tests
pytest tests/ --cov=core

# Run examples
python examples/stock_prediction/train.py
python examples/sports_analytics/train.py

# Scan project usage
python tools/scan_usage.py
```

---

## Architecture

```
core/                           # Framework core (80% shared logic)
├── data/                       # TimeSeriesDataset, TimeSeriesDataLoader
├── features/                   # FeatureEngine, transforms (rolling, lag, technical indicators)
├── models/                     # BaseTimeSeriesModel, LSTM, GRU, Transformer, Ensemble
├── training/                   # Trainer, callbacks (EarlyStopping, ModelCheckpoint)
├── validation/                 # TimeSeriesSplit, WalkForwardSplit (temporal CV)
└── utils/                      # Metrics, feature importance, helpers

config/config_manager.py        # YAML/JSON config loading

web_ui/                         # Streamlit dashboard
├── app.py                      # Entry point with routing
├── auth.py                     # User auth & sessions
└── pages/                      # Home, dataset manager, model builder, training, results

examples/                       # Domain-specific examples (20% custom)
├── stock_prediction/           # Stock forecasting
└── sports_analytics/           # Sports performance prediction

tests/                          # pytest-based tests
```

**Key files:**
- `config_template.yaml` — Master config template with all options
- `core/models/base_model.py` — Abstract base class all models inherit
- `core/training/trainer.py` — Training loop with validation, metrics, callbacks
- `core/validation/time_series_split.py` — Time-aware cross-validation (prevents data leakage)

---

## Data Storage

No SQL database. Data flows as:
- **Input**: CSV/Excel via UI or numpy arrays in code
- **Training**: PyTorch tensors
- **Checkpoints**: `.pt` files in configured checkpoint directory
- **User auth**: `.streamlit/users.json` (hashed passwords)
- **Usage stats**: `~/.deep-timeseries/usage_stats.json`

---

## Conventions

- **80/20 philosophy**: Core handles common functionality; examples show domain-specific customization
- **Time-awareness**: All validation respects temporal order — use `TimeSeriesSplit` or `WalkForwardSplit`, never random splits
- **Config-driven**: Projects use YAML configs (see `config_template.yaml`)
- **Model inheritance**: Custom models extend `BaseTimeSeriesModel` and implement `forward()`, `create_model()`
- **Feature transforms**: Custom features extend base transform classes in `core/features/transforms.py`
- **Callbacks**: Training hooks via `core/training/callbacks.py` — EarlyStopping, ModelCheckpoint, LRScheduler

**Naming:**
- Models: `<Name>Model` (e.g., `LSTMModel`, `TransformerModel`)
- Transforms: `<Name>Features` or `<Name>Transform`
- Config keys: snake_case

---

## Don't Touch

- `core/validation/time_series_split.py` — Critical for preventing data leakage; changing split logic can silently corrupt training
- `web_ui/auth.py` — Password hashing and session management; changes can lock users out
- `.streamlit/users.json` — User credentials; manual edits may corrupt auth
- `core/__init__.py` exports — Many projects import from `core` directly; breaking exports breaks downstream

---

## Common Tasks

### Add a new model type
1. Create `core/models/<name>.py`
2. Extend `BaseTimeSeriesModel`
3. Implement `forward()` and `create_model()`
4. Export in `core/models/__init__.py`
5. Add to model selection in `web_ui/pages/model_builder.py`

### Add custom features for a domain
1. Create transforms in `examples/<domain>/custom_features.py`
2. Extend base transform classes
3. Register with `FeatureEngine` in training script

### Add a new dashboard page
1. Create `web_ui/pages/<name>.py` with `render()` function
2. Add route in `web_ui/app.py`

### Debug training issues
1. Check `core/training/trainer.py` for training loop
2. Check callbacks in `core/training/callbacks.py`
3. Verify data shapes in `core/data/dataset.py`

### Test changes
```bash
pytest tests/ -v                    # All tests
pytest tests/test_basic.py -k test_name  # Specific test
```

---

## Config Reference

Key config sections (see `config_template.yaml` for full options):

```yaml
data:
  sequence_length: 30      # Input window size
  forecast_horizon: 1      # Steps ahead to predict
  batch_size: 32

model:
  type: lstm               # lstm | gru | transformer | ensemble
  hidden_dim: 64
  num_layers: 2
  dropout: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping:
    patience: 10
    min_delta: 0.0001

validation:
  method: time_series_cv   # simple_split | time_series_cv | walk_forward
  num_splits: 5
```
