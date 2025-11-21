"""
Custom feature transformers for sports analytics

This demonstrates how to extend the core framework with domain-specific features.
"""

import pandas as pd
import numpy as np
from typing import Union


class PlayerPerformanceFeatures:
    """
    Extract player performance features

    This is an example of domain-specific feature engineering for sports.
    """

    def __init__(self, windows: list = [3, 5, 10]):
        self.windows = windows

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate player performance features

        Expected columns:
        - points: Points scored
        - assists: Assists
        - rebounds: Rebounds
        - minutes_played: Minutes played
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[
                'points', 'assists', 'rebounds', 'minutes_played'
            ])

        result = data.copy()

        # Calculate efficiency rating
        result['efficiency'] = (
            result['points'] + result['assists'] + result['rebounds']
        ) / (result['minutes_played'] + 1)

        # Points per minute
        result['points_per_minute'] = result['points'] / (result['minutes_played'] + 1)

        # Rolling averages for different time windows
        for col in ['points', 'assists', 'rebounds', 'efficiency']:
            for window in self.windows:
                result[f'{col}_avg_{window}_games'] = (
                    result[col].rolling(window=window, min_periods=1).mean()
                )

                # Rolling standard deviation (form consistency)
                result[f'{col}_std_{window}_games'] = (
                    result[col].rolling(window=window, min_periods=1).std()
                )

        # Momentum: difference between recent and longer-term average
        if len(self.windows) >= 2:
            short_window = self.windows[0]
            long_window = self.windows[-1]

            for col in ['points', 'assists', 'rebounds']:
                result[f'{col}_momentum'] = (
                    result[f'{col}_avg_{short_window}_games'] -
                    result[f'{col}_avg_{long_window}_games']
                )

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform"""
        return self.transform(data)


class TeamStreakFeatures:
    """
    Calculate team winning/losing streak features
    """

    def __init__(self):
        pass

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate streak features

        Expected column:
        - win: Binary indicator (1 = win, 0 = loss)
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=['win'])

        result = data.copy()

        # Calculate current streak
        result['streak'] = self._calculate_streak(result['win'])

        # Win percentage over different windows
        for window in [5, 10, 20]:
            result[f'win_pct_{window}_games'] = (
                result['win'].rolling(window=window, min_periods=1).mean()
            )

        return result

    def _calculate_streak(self, wins: pd.Series) -> pd.Series:
        """Calculate winning/losing streak"""
        streak = []
        current_streak = 0

        for win in wins:
            if pd.isna(win):
                streak.append(0)
                continue

            if win == 1:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1

            streak.append(current_streak)

        return pd.Series(streak, index=wins.index)

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform"""
        return self.transform(data)


class MatchupFeatures:
    """
    Features based on opponent strength and matchup history
    """

    def __init__(self):
        self.opponent_stats = {}

    def fit(self, data: pd.DataFrame):
        """Learn opponent statistics from historical data"""
        if 'opponent_id' in data.columns:
            # Calculate average stats for each opponent
            self.opponent_stats = data.groupby('opponent_id').mean().to_dict('index')
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Add matchup features

        Expected columns:
        - opponent_id: Opponent identifier
        - home_game: Binary indicator (1 = home, 0 = away)
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        result = data.copy()

        # Add home/away indicator if not present
        if 'home_game' not in result.columns:
            result['home_game'] = 0

        # Add opponent strength rating if opponent_id is present
        if 'opponent_id' in result.columns and self.opponent_stats:
            result['opponent_strength'] = result['opponent_id'].map(
                lambda x: self.opponent_stats.get(x, {}).get('points', 0)
            )

        return result

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(data)
        return self.transform(data)
