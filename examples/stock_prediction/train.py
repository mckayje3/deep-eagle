"""
Stock Price Prediction Example

This example demonstrates how to use the core deep learning framework
for stock price prediction with technical indicators.
"""

import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam

# Import from core framework
from core import (
    TimeSeriesDataset,
    TimeSeriesDataLoader,
    FeatureEngine,
    LSTMModel,
    Trainer,
    TimeSeriesSplit,
)
from core.features import TechnicalIndicators, LagFeatures, RollingWindow
from core.training import EarlyStopping, ModelCheckpoint
from core.utils import set_seed, get_device, mse, mae
from config.config_manager import Config, save_default_config


def load_stock_data(filepath: str = 'stock_data.csv') -> pd.DataFrame:
    """
    Load stock data from CSV file

    Expected columns: Date, Open, High, Low, Close, Volume
    """
    # In a real scenario, you would load actual stock data
    # For this example, we'll generate synthetic data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)

    # Generate synthetic stock prices with trend and noise
    trend = np.linspace(100, 200, len(dates))
    noise = np.random.normal(0, 10, len(dates))
    close_prices = trend + noise

    df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    })

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df


def prepare_features(data: pd.DataFrame, config: Config) -> tuple:
    """
    Prepare features using the feature engineering pipeline

    Args:
        data: Raw stock data
        config: Configuration object

    Returns:
        Tuple of (train_data, test_data, feature_engine)
    """
    # Initialize feature transformers
    transformers = []

    # Add technical indicators if enabled
    if config.get('features.technical_indicators.enabled', False):
        transformers.append(TechnicalIndicators(
            include_rsi=config.get('features.technical_indicators.include_rsi', True),
            include_macd=config.get('features.technical_indicators.include_macd', True),
            include_bollinger=config.get('features.technical_indicators.include_bollinger', True),
        ))

    # Add lag features
    transformers.append(LagFeatures(lags=[1, 2, 3, 5, 10]))

    # Add rolling window features
    transformers.append(RollingWindow(windows=[5, 10, 20], functions=['mean', 'std']))

    # Create feature engine
    feature_engine = FeatureEngine(
        transformers=transformers,
        scaler=config.get('features.scaler', 'standard'),
        handle_missing=config.get('features.handle_missing', 'ffill'),
    )

    # Split data temporally
    train_size = int(len(data) * (1 - config.get('data.test_size', 0.2)))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # Fit and transform
    train_features = feature_engine.fit_transform(train_data)
    test_features = feature_engine.transform(test_data)

    return train_features, test_features, feature_engine


def create_datasets(train_features, test_features, config: Config) -> tuple:
    """
    Create PyTorch datasets

    Args:
        train_features: Transformed training features
        test_features: Transformed test features
        config: Configuration object

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    sequence_length = config.get('data.sequence_length', 30)
    forecast_horizon = config.get('data.forecast_horizon', 1)

    # Create target (predict close price - assume it's the first feature)
    train_targets = train_features[:, 0]
    test_targets = test_features[:, 0]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_features,
        targets=train_targets,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    test_dataset = TimeSeriesDataset(
        data=test_features,
        targets=test_targets,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    return train_dataset, test_dataset


def main():
    """Main training pipeline"""
    # Set random seed for reproducibility
    set_seed(42)

    # Generate default config if it doesn't exist
    try:
        config = Config.from_yaml('config.yaml')
    except FileNotFoundError:
        save_default_config('config.yaml', format='yaml')
        config = Config.from_yaml('config.yaml')
        print("Created default config.yaml - please review and adjust settings")

    # Enable technical indicators for stock prediction
    config.set('features.technical_indicators.enabled', True)

    print("=" * 50)
    print("Stock Price Prediction - Training Pipeline")
    print("=" * 50)

    # Load data
    print("\n1. Loading stock data...")
    data = load_stock_data()
    print(f"   Loaded {len(data)} days of stock data")

    # Prepare features
    print("\n2. Engineering features...")
    train_features, test_features, feature_engine = prepare_features(data, config)
    print(f"   Created {train_features.shape[1]} features")

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset, test_dataset = create_datasets(train_features, test_features, config)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # Create data loaders
    batch_size = config.get('data.batch_size', 32)
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TimeSeriesDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("\n4. Building model...")
    device = get_device() if config.get('training.device') == 'auto' else config.get('training.device')
    print(f"   Using device: {device}")

    model = LSTMModel(
        input_dim=train_dataset.n_features,
        hidden_dim=config.get('model.hidden_dim', 64),
        output_dim=config.get('model.output_dim', 1),
        num_layers=config.get('model.num_layers', 2),
        dropout=config.get('model.dropout', 0.1),
        forecast_horizon=config.get('data.forecast_horizon', 1),
    )
    print(f"   Model parameters: {model.count_parameters():,}")

    # Setup training
    optimizer = Adam(model.parameters(), lr=config.get('training.learning_rate', 0.001))
    criterion = nn.MSELoss()

    # Setup callbacks
    callbacks = []
    if config.get('training.early_stopping.enabled', True):
        callbacks.append(EarlyStopping(
            patience=config.get('training.early_stopping.patience', 10),
            min_delta=config.get('training.early_stopping.min_delta', 0.0001),
        ))

    if config.get('training.checkpoint.enabled', True):
        callbacks.append(ModelCheckpoint(
            filepath=config.get('training.checkpoint.filepath', 'best_model.pt'),
        ))

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
    )

    # Train model
    print("\n5. Training model...")
    print("-" * 50)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config.get('training.epochs', 100),
        metrics={'mae': mae},
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
