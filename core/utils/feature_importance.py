"""
Feature importance analysis for time-series models
Understand what features drive your predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Tuple
from collections import defaultdict


class FeatureImportance:
    """
    Calculate feature importance for time-series models

    Supports multiple methods:
    - Permutation importance (most reliable)
    - Gradient-based importance
    - Ablation importance (remove features one at a time)

    Args:
        model: Trained PyTorch model
        feature_names: List of feature names
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names

    def permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_repeats: int = 10,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Calculate permutation importance

        Measures how much the model's performance decreases when each
        feature is randomly shuffled. More reliable than gradient-based methods.

        Args:
            X: Input data (batch, sequence, features)
            y: Target data
            n_repeats: Number of times to permute each feature
            criterion: Loss function (default: MSE)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if criterion is None:
            criterion = nn.MSELoss()

        n_features = X.shape[-1]

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Calculate baseline loss
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_loss = criterion(baseline_pred, y).item()

        importances = {}

        for feature_idx in range(n_features):
            feature_losses = []

            for _ in range(n_repeats):
                # Create copy and permute one feature
                X_permuted = X.clone()

                # Permute across the batch dimension for this feature
                perm_indices = torch.randperm(X.shape[0])
                X_permuted[:, :, feature_idx] = X[perm_indices, :, feature_idx]

                # Calculate loss with permuted feature
                with torch.no_grad():
                    permuted_pred = self.model(X_permuted)
                    permuted_loss = criterion(permuted_pred, y).item()

                feature_losses.append(permuted_loss)

            # Importance = increase in loss when feature is permuted
            mean_permuted_loss = np.mean(feature_losses)
            importance = mean_permuted_loss - baseline_loss

            feature_name = self.feature_names[feature_idx]
            importances[feature_name] = importance

        # Normalize to sum to 1
        total = sum(max(0, v) for v in importances.values())
        if total > 0:
            importances = {k: max(0, v) / total for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def gradient_importance(
        self,
        X: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculate gradient-based feature importance

        Measures the average absolute gradient of the output with respect
        to each input feature. Fast but can be less reliable than permutation.

        Args:
            X: Input data (batch, sequence, features)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        n_features = X.shape[-1]

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        X_grad = X.clone().requires_grad_(True)

        # Forward pass
        output = self.model(X_grad)

        # Backward pass
        output.sum().backward()

        # Get gradients
        gradients = X_grad.grad.abs()

        # Average across batch and sequence dimensions
        feature_grads = gradients.mean(dim=(0, 1))

        # Convert to dictionary
        importances = {}
        for i, name in enumerate(self.feature_names):
            importances[name] = feature_grads[i].item()

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def ablation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None,
        baseline_value: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate ablation importance

        Measures model performance when each feature is set to a baseline value.

        Args:
            X: Input data (batch, sequence, features)
            y: Target data
            criterion: Loss function (default: MSE)
            baseline_value: Value to replace features with (default: 0)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if criterion is None:
            criterion = nn.MSELoss()

        n_features = X.shape[-1]

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Calculate baseline loss
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_loss = criterion(baseline_pred, y).item()

        importances = {}

        for feature_idx in range(n_features):
            # Create copy and ablate one feature
            X_ablated = X.clone()
            X_ablated[:, :, feature_idx] = baseline_value

            # Calculate loss with ablated feature
            with torch.no_grad():
                ablated_pred = self.model(X_ablated)
                ablated_loss = criterion(ablated_pred, y).item()

            # Importance = increase in loss when feature is removed
            importance = ablated_loss - baseline_loss

            feature_name = self.feature_names[feature_idx]
            importances[feature_name] = importance

        # Normalize
        total = sum(max(0, v) for v in importances.values())
        if total > 0:
            importances = {k: max(0, v) / total for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def temporal_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """
        Calculate importance of each time step

        Useful for understanding how far back in time the model looks.

        Args:
            X: Input data (batch, sequence, features)
            y: Target data
            criterion: Loss function

        Returns:
            Array of importance scores for each time step
        """
        if criterion is None:
            criterion = nn.MSELoss()

        seq_length = X.shape[1]

        # Baseline loss
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_loss = criterion(baseline_pred, y).item()

        timestep_importance = []

        for t in range(seq_length):
            # Zero out this timestep
            X_masked = X.clone()
            X_masked[:, t, :] = 0

            with torch.no_grad():
                masked_pred = self.model(X_masked)
                masked_loss = criterion(masked_pred, y).item()

            importance = masked_loss - baseline_loss
            timestep_importance.append(max(0, importance))

        # Normalize
        total = sum(timestep_importance)
        if total > 0:
            timestep_importance = [v / total for v in timestep_importance]

        return np.array(timestep_importance)


def calculate_all_importances(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Calculate feature importance using all methods

    Args:
        model: Trained model
        X: Input data
        y: Target data
        feature_names: Optional feature names
        n_repeats: Repeats for permutation importance

    Returns:
        DataFrame with importance scores from each method
    """
    fi = FeatureImportance(model, feature_names)

    # Calculate with each method
    perm_imp = fi.permutation_importance(X, y, n_repeats=n_repeats)
    grad_imp = fi.gradient_importance(X)
    ablation_imp = fi.ablation_importance(X, y)

    # Combine into DataFrame
    features = list(perm_imp.keys())

    df = pd.DataFrame({
        'feature': features,
        'permutation': [perm_imp[f] for f in features],
        'gradient': [grad_imp.get(f, 0) for f in features],
        'ablation': [ablation_imp.get(f, 0) for f in features],
    })

    # Calculate average importance
    df['average'] = df[['permutation', 'gradient', 'ablation']].mean(axis=1)

    # Sort by average importance
    df = df.sort_values('average', ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = "Feature Importance",
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot feature importance as horizontal bar chart

    Args:
        importances: Dictionary of feature importances
        title: Plot title
        top_n: Only show top N features
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None

    # Sort and optionally limit
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    if top_n:
        sorted_imp = dict(list(sorted_imp.items())[:top_n])

    features = list(sorted_imp.keys())
    values = list(sorted_imp.values())

    # Reverse for horizontal bar chart (highest at top)
    features = features[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(features, values, color='steelblue')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.2)

    plt.tight_layout()
    return fig


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) integration for model explainability

    Requires: pip install shap

    Args:
        model: Trained PyTorch model
        background_data: Background data for SHAP calculations
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.model.eval()
        self.background_data = background_data
        self._check_shap_installed()

    def _check_shap_installed(self):
        """Check if SHAP is installed"""
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.shap = None
            print("SHAP not installed. Install with: pip install shap")
            print("Using fallback importance methods instead.")

    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP to call model"""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            pred = self.model(X_tensor)
            return pred.numpy()

    def explain(
        self,
        X: Union[torch.Tensor, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate SHAP values for predictions

        Args:
            X: Input data to explain
            feature_names: Feature names

        Returns:
            Dictionary of feature -> mean absolute SHAP value
        """
        if self.shap is None:
            print("SHAP not available. Using permutation importance instead.")
            fi = FeatureImportance(self.model, feature_names)
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            # Create dummy target for permutation
            with torch.no_grad():
                y = self.model(X)
            return fi.permutation_importance(X, y)

        if isinstance(X, torch.Tensor):
            X = X.numpy()

        n_features = X.shape[-1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Reshape for SHAP: (batch, sequence * features)
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)

        if self.background_data is not None:
            bg_flat = self.background_data.numpy().reshape(self.background_data.shape[0], -1)
            explainer = self.shap.KernelExplainer(
                lambda x: self._model_predict(x.reshape(-1, *original_shape[1:])),
                bg_flat[:100]  # Limit background samples
            )
        else:
            explainer = self.shap.KernelExplainer(
                lambda x: self._model_predict(x.reshape(-1, *original_shape[1:])),
                X_flat[:100]
            )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_flat[:50])  # Limit for speed

        # Reshape back and aggregate by feature
        shap_reshaped = shap_values.reshape(-1, original_shape[1], original_shape[2])

        # Mean absolute SHAP value per feature (across batch and time)
        feature_importance = np.abs(shap_reshaped).mean(axis=(0, 1))

        # Normalize
        total = feature_importance.sum()
        if total > 0:
            feature_importance = feature_importance / total

        return dict(zip(feature_names, feature_importance))
