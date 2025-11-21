"""
Sports Performance Prediction Example

This example demonstrates how to extend the core framework with
custom domain-specific features for sports analytics.
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
    GRUModel,
    Trainer,
    WalkForwardSplit,
)
from core.training import EarlyStopping, ModelCheckpoint
from core.utils import set_seed, get_device, mse, mae
from config.config_manager import Config, save_default_config

# Import custom features
from custom_features import PlayerPerformanceFeatures, TeamStreakFeatures


def generate_synthetic_sports_data(n_games: int = 500) -> pd.DataFrame:
    """
    Generate synthetic sports data for demonstration

    In practice, you would load real game data here.
    """
    np.random.seed(42)

    # Generate player stats over games
    games = np.arange(n_games)

    # Base performance with trend and variability
    base_points = 20 + np.sin(games / 20) * 5 + np.random.normal(0, 3, n_games)
    base_assists = 5 + np.random.normal(0, 1.5, n_games)
    base_rebounds = 8 + np.random.normal(0, 2, n_games)
    minutes_played = 30 + np.random.normal(0, 5, n_games)

    # Wins (influenced by performance)
    win_prob = (base_points + base_assists * 2 + base_rebounds) / 60
    wins = (np.random.random(n_games) < win_prob).astype(int)

    df = pd.DataFrame({
        'game_num': games,
        'points': np.clip(base_points, 0, 50),
        'assists': np.clip(base_assists, 0, 15),
        'rebounds': np.clip(base_rebounds, 0, 20),
        'minutes_played': np.clip(minutes_played, 10, 48),
        'win': wins,
    })

    return df


def prepare_sports_features(data: pd.DataFrame, config: Config) -> tuple:
    """
    Prepare features with custom sports transformers

    This demonstrates extending the framework with domain knowledge.
    """
    # Initialize custom transformers
    transformers = [
        PlayerPerformanceFeatures(windows=[3, 5, 10]),
        TeamStreakFeatures(),
    ]

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
    """Create PyTorch datasets for sports prediction"""
    sequence_length = config.get('data.sequence_length', 10)  # Last 10 games
    forecast_horizon = config.get('data.forecast_horizon', 1)  # Predict next game

    # Predict points scored (first feature after transformation)
    train_targets = train_features[:, 0]
    test_targets = test_features[:, 0]

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
    """Main training pipeline for sports analytics"""
    # Set random seed
    set_seed(42)

    # Load or create config
    try:
        config = Config.from_yaml('config.yaml')
    except FileNotFoundError:
        save_default_config('config.yaml', format='yaml')
        config = Config.from_yaml('config.yaml')

    # Adjust config for sports analytics
    config.set('data.sequence_length', 10)  # Look at last 10 games
    config.set('model.type', 'gru')
    config.set('model.hidden_dim', 32)

    print("=" * 50)
    print("Sports Performance Prediction - Training Pipeline")
    print("=" * 50)

    # Load data
    print("\n1. Loading sports data...")
    data = generate_synthetic_sports_data(n_games=500)
    print(f"   Loaded {len(data)} games of data")
    print(f"   Columns: {list(data.columns)}")

    # Prepare features with custom transformers
    print("\n2. Engineering sports-specific features...")
    train_features, test_features, feature_engine = prepare_sports_features(data, config)
    print(f"   Created {train_features.shape[1]} features")
    print(f"   Feature names: {feature_engine.get_feature_names()[:5]}... (showing first 5)")

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset, test_dataset = create_datasets(train_features, test_features, config)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # Create data loaders
    batch_size = config.get('data.batch_size', 32)
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TimeSeriesDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model (using GRU for variety)
    print("\n4. Building GRU model...")
    device = get_device() if config.get('training.device') == 'auto' else config.get('training.device')
    print(f"   Using device: {device}")

    model = GRUModel(
        input_dim=train_dataset.n_features,
        hidden_dim=config.get('model.hidden_dim', 32),
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
    callbacks = [
        EarlyStopping(patience=15, verbose=True),
        ModelCheckpoint(filepath='best_sports_model.pt', verbose=True),
    ]

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
        epochs=config.get('training.epochs', 50),
        metrics={'mae': mae},
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Final validation MAE: {history['val_loss'][-1]:.4f}")
    print("\nThis model predicts player points based on:")
    print("  - Historical performance (rolling averages)")
    print("  - Recent form (momentum indicators)")
    print("  - Team streaks (win/loss patterns)")
    print("=" * 50)


if __name__ == '__main__':
    main()
