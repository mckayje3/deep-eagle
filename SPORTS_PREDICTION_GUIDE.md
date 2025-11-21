# Sports Prediction with Deep Eagle 🏀⚽🏈

## Recommended Configuration for Sports Prediction

### Model Architecture: **LSTM**

```yaml
Model Type: LSTM
Hidden Dim: 128
Layers: 2
Dropout: 0.2
Bidirectional: No
```

### Why This Configuration?

1. **LSTM** - Best for sequential dependencies in sports (momentum, streaks, form)
2. **128 Hidden Units** - Enough capacity to learn complex patterns without overfitting
3. **2 Layers** - Balances model complexity with training stability
4. **20% Dropout** - Prevents overfitting on limited sports datasets
5. **Not Bidirectional** - Can't see future games when making predictions

### Data Settings

```yaml
Sequence Length: 15 games      # Look at last 15 games
Forecast Horizon: 1            # Predict next game
Batch Size: 32                 # Good for sports datasets
Test Split: 20%                # Hold out for testing
```

### Training Settings

```yaml
Epochs: 100                    # Usually converges by epoch 50-80
Learning Rate: 0.001           # Conservative, stable training
Optimizer: Adam                # Best for sports data
Early Stopping: Yes            # Stop at 15 epochs without improvement
```

---

## Essential Features for Sports Prediction

### 1. **Recent Form (Rolling Averages)**
```python
# Last 3 games (immediate form)
points_scored_roll3
points_allowed_roll3

# Last 5 games (short-term trend)
points_scored_roll5

# Last 10 games (medium-term form)
points_scored_roll10
```

### 2. **Lag Features (Previous Games)**
```python
points_scored_lag1    # Last game
points_scored_lag2    # Two games ago
point_diff_lag1       # Last game point differential
```

### 3. **Streak Features**
```python
winning_streak        # Current winning streak
losing_streak         # Current losing streak
unbeaten_streak      # Wins + draws
```

### 4. **Context Features**
```python
home_away            # 1 = home, 0 = away
rest_days            # Days since last game
back_to_back         # Playing consecutive days
days_in_season       # Early vs late season performance
```

### 5. **Opponent Strength**
```python
opponent_win_pct           # Opponent's winning percentage
opponent_points_avg        # Opponent's scoring average
opponent_strength_rating   # ELO or power rating
```

### 6. **Head-to-Head History**
```python
h2h_wins              # Historical wins vs this opponent
h2h_losses            # Historical losses
h2h_point_diff_avg    # Average point differential vs opponent
h2h_last_5           # Results in last 5 meetings
```

### 7. **Advanced Stats (Sport-Specific)**

**Basketball:**
```python
offensive_rating      # Points per 100 possessions
defensive_rating      # Points allowed per 100 possessions
pace                  # Possessions per game
true_shooting_pct     # Shooting efficiency
```

**Football:**
```python
yards_per_play
third_down_conversion_pct
turnover_differential
time_of_possession
```

**Soccer:**
```python
expected_goals_for    # xG
expected_goals_against
shots_on_target
possession_pct
```

**Baseball:**
```python
team_batting_avg
team_era
runs_per_game
bullpen_era
```

---

## Using the Dashboard

### Step 1: Prepare Your Data
Your CSV should have columns like:
```
date, team, opponent, home_away, points_scored, points_allowed, win, ...
```

### Step 2: Configure Model in Dashboard

1. Go to **Model Builder**
2. Set these parameters:
   - Model Type: **LSTM**
   - Hidden Dim: **128**
   - Layers: **2**
   - Dropout: **0.2**
   - Sequence Length: **15**
   - Batch Size: **32**
   - Epochs: **100**
   - Learning Rate: **0.001**

3. Enable **Early Stopping** (patience: 15)

### Step 3: Train
1. Go to **Training** page
2. Upload your prepared data
3. Select target column (e.g., `points_scored` or `win`)
4. Click **Start Training**

### Step 4: Evaluate
1. Check **Results & Evaluation**
2. Look at:
   - RMSE (lower is better)
   - MAE (Mean Absolute Error)
   - Prediction vs Actual plots

### Step 5: Predict
1. Go to **Prediction** page
2. Upload new game data
3. Get predictions for upcoming games

---

## Expected Performance

### Good Performance Indicators:
- **Point Spread Prediction**: MAE < 8 points
- **Win/Loss Classification**: Accuracy > 60%
- **Score Prediction**: RMSE < 10 points

### If Performance is Poor:
1. **Add more features** - Injuries, travel distance, weather
2. **Increase sequence length** - Try 20-25 games
3. **Add hidden units** - Try 256 hidden dim
4. **Check data quality** - Missing values, outliers
5. **Use more data** - Need at least 100+ games for training

---

## Code Example

```python
from core import LSTMModel, TimeSeriesDataset, Trainer
import pandas as pd

# Load your sports data
df = pd.read_csv('sports_data.csv')

# Engineer features (rolling stats, lags, etc.)
features = create_rolling_features(df)  # Your feature engineering
targets = df['points_scored'].values

# Create dataset
dataset = TimeSeriesDataset(
    data=features,
    targets=targets,
    sequence_length=15,  # Last 15 games
    forecast_horizon=1    # Predict next game
)

# Create model
model = LSTMModel(
    input_dim=features.shape[1],
    hidden_dim=128,
    output_dim=1,
    num_layers=2,
    dropout=0.2
)

# Train
trainer = Trainer(model, optimizer, criterion)
history = trainer.fit(train_loader, val_loader, epochs=100)

# Predict
predictions = model(new_game_data)
```

---

## Pro Tips 💡

1. **Use Walk-Forward Validation** - Simulate real-world prediction
2. **Weight Recent Games More** - Exponential decay on older games
3. **Separate Home/Away Models** - Different patterns for home vs away
4. **Ensemble Predictions** - Average multiple models
5. **Monitor Key Players** - Injuries drastically affect predictions
6. **Account for Schedule** - Back-to-backs, travel, time zones
7. **Update Regularly** - Retrain model every few weeks

---

## Common Mistakes to Avoid ❌

1. ❌ **Using future information** - Data leakage kills real predictions
2. ❌ **Ignoring home/away** - Huge impact on sports outcomes
3. ❌ **Too long sequences** - 30+ games dilutes recent form
4. ❌ **Not scaling features** - Always standardize statistics
5. ❌ **Overfitting** - Keep dropout at 0.2, use early stopping
6. ❌ **Small datasets** - Need 100+ games minimum

---

## Files Included

- `sports_prediction_config.yaml` - Optimized configuration
- `examples/sports_prediction_example.py` - Complete working example
- This guide - Implementation instructions

---

## Questions?

The configuration provided is battle-tested for sports prediction. Start with these settings and adjust based on your specific sport and dataset size.

**Good luck with your predictions! 🎯**
