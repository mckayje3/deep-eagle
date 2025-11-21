"""
Sports Prediction Example using Deep Eagle
Optimized configuration for predicting sports outcomes
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import Deep Eagle components
from core import (
    TimeSeriesDataset,
    TimeSeriesDataLoader,
    LSTMModel,
    Trainer,
    FeatureEngine,
    WalkForwardSplit
)
from core.training.callbacks import EarlyStopping, ModelCheckpoint
from config.config_manager import Config


def prepare_sports_data(df, target_column='points_scored'):
    """
    Prepare sports data with rolling statistics and lag features

    Args:
        df: DataFrame with game-by-game statistics
        target_column: Column to predict

    Returns:
        features, targets as numpy arrays
    """

    # Sort by date to ensure chronological order
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    # Create rolling statistics (recent form)
    feature_cols = []

    # Rolling averages for key stats
    stats_to_roll = ['points_scored', 'points_allowed', 'point_differential']

    for stat in stats_to_roll:
        if stat in df.columns:
            # 3-game rolling average (recent form)
            df[f'{stat}_roll3'] = df[stat].rolling(window=3, min_periods=1).mean()
            feature_cols.append(f'{stat}_roll3')

            # 5-game rolling average
            df[f'{stat}_roll5'] = df[stat].rolling(window=5, min_periods=1).mean()
            feature_cols.append(f'{stat}_roll5')

            # 10-game rolling average
            df[f'{stat}_roll10'] = df[stat].rolling(window=10, min_periods=1).mean()
            feature_cols.append(f'{stat}_roll10')

    # Lag features (previous games)
    for stat in stats_to_roll:
        if stat in df.columns:
            df[f'{stat}_lag1'] = df[stat].shift(1)
            df[f'{stat}_lag2'] = df[stat].shift(2)
            feature_cols.extend([f'{stat}_lag1', f'{stat}_lag2'])

    # Winning streak
    if 'win' in df.columns:
        df['winning_streak'] = (df['win'].groupby((df['win'] != df['win'].shift()).cumsum()).cumcount() + 1) * df['win']
        feature_cols.append('winning_streak')

    # Rest days (if dates available)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['rest_days'] = df['date'].diff().dt.days.fillna(3)
        feature_cols.append('rest_days')

    # Home/Away indicator
    if 'home_away' in df.columns:
        feature_cols.append('home_away')

    # Drop rows with NaN (from rolling/lag operations)
    df = df.dropna()

    # Extract features and target
    features = df[feature_cols].values
    targets = df[target_column].values

    return features, targets


def train_sports_prediction_model(features, targets, config_path='sports_prediction_config.yaml'):
    """
    Train a sports prediction model with optimal configuration

    Args:
        features: Feature array (n_samples, n_features)
        targets: Target array (n_samples,)
        config_path: Path to configuration file

    Returns:
        Trained model, training history
    """

    # Load configuration
    config = Config.from_yaml(config_path)

    # Create dataset
    sequence_length = config.get('data.sequence_length', 15)
    forecast_horizon = config.get('data.forecast_horizon', 1)
    batch_size = config.get('data.batch_size', 32)

    dataset = TimeSeriesDataset(
        data=features,
        targets=targets,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )

    # Split train/test
    test_size = config.get('data.test_size', 0.2)
    train_size = int(len(dataset) * (1 - test_size))

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    # Create data loaders
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TimeSeriesDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = features.shape[1]
    hidden_dim = config.get('model.hidden_dim', 128)
    output_dim = config.get('model.output_dim', 1)
    num_layers = config.get('model.num_layers', 2)
    dropout = config.get('model.dropout', 0.2)

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Setup training
    learning_rate = config.get('training.learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Callbacks
    callbacks = [
        EarlyStopping(
            patience=config.get('training.early_stopping.patience', 15),
            min_delta=config.get('training.early_stopping.min_delta', 0.0001)
        ),
        ModelCheckpoint(
            filepath='checkpoints/sports_best_model.pth',
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # Create trainer
    device = config.get('training.device', 'auto')
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks
    )

    # Train
    epochs = config.get('training.epochs', 100)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs
    )

    return model, history, test_loader


def evaluate_predictions(model, test_loader, targets):
    """Evaluate model predictions"""

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    print(f"\n📊 Evaluation Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # For win/loss prediction (if binary)
    if len(np.unique(actuals)) == 2:
        accuracy = np.mean((predictions > 0.5) == actuals)
        print(f"  Accuracy: {accuracy:.2%}")

    return predictions, actuals


# Example usage
if __name__ == "__main__":
    print("🏀 Sports Prediction Model - Deep Eagle Example\n")

    # Example: Load your sports data
    # df = pd.read_csv('your_sports_data.csv')

    # For demonstration, create sample data
    np.random.seed(42)
    n_games = 200

    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_games, freq='D'),
        'points_scored': np.random.randint(80, 120, n_games),
        'points_allowed': np.random.randint(80, 120, n_games),
        'point_differential': np.random.randint(-20, 20, n_games),
        'win': np.random.randint(0, 2, n_games),
        'home_away': np.random.randint(0, 2, n_games),
    })

    print(f"📈 Loaded {len(sample_data)} games of data")

    # Prepare features
    print("\n🔧 Engineering features...")
    features, targets = prepare_sports_data(sample_data, target_column='points_scored')
    print(f"✅ Created {features.shape[1]} features from {features.shape[0]} games")

    # Train model
    print("\n🚀 Training model...")
    model, history, test_loader = train_sports_prediction_model(features, targets)

    print("\n✅ Training complete!")
    print(f"📉 Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"📉 Final val loss: {history['val_loss'][-1]:.4f}")

    # Evaluate
    print("\n🎯 Evaluating on test set...")
    predictions, actuals = evaluate_predictions(model, test_loader, targets)

    print("\n🎉 Sports prediction model ready!")
    print("\n💡 Tips for improvement:")
    print("  - Add more domain-specific features (injuries, travel distance, etc.)")
    print("  - Tune hyperparameters using walk-forward validation")
    print("  - Try ensemble methods combining multiple models")
    print("  - Use opponent-specific features (head-to-head history)")
