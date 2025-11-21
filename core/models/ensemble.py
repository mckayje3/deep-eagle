"""
Ensemble methods for combining multiple models
Provides significant accuracy improvements for predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Union, Callable
from pathlib import Path

from .base_model import BaseTimeSeriesModel


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with various combining strategies

    Supports:
    - Simple averaging
    - Weighted averaging
    - Learned weights
    - Stacking (meta-learner)

    Args:
        models: List of trained models to ensemble
        method: Ensemble method ('average', 'weighted', 'learned', 'stacking')
        weights: Optional weights for weighted averaging
    """

    def __init__(
        self,
        models: List[nn.Module],
        method: str = 'average',
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        self.models = nn.ModuleList(models)
        self.method = method
        self.n_models = len(models)

        # Set all models to eval mode by default
        for model in self.models:
            model.eval()

        # Initialize weights
        if method == 'weighted':
            if weights is None:
                # Equal weights by default
                weights = [1.0 / self.n_models] * self.n_models
            if len(weights) != self.n_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.n_models})")
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

        elif method == 'learned':
            # Learnable weights initialized equally
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models, requires_grad=True)

        elif method == 'stacking':
            # Meta-learner that combines model outputs
            # Get output dimension from first model
            self.meta_learner = nn.Sequential(
                nn.Linear(self.n_models, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        elif method != 'average':
            raise ValueError(f"Unknown ensemble method: {method}. Use 'average', 'weighted', 'learned', or 'stacking'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor (batch, sequence, features)

        Returns:
            Combined predictions
        """
        # Get predictions from all models
        predictions = []
        with torch.no_grad() if self.method != 'stacking' else torch.enable_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        # Stack predictions: (n_models, batch, output_dim)
        stacked = torch.stack(predictions, dim=0)

        if self.method == 'average':
            # Simple average
            return torch.mean(stacked, dim=0)

        elif self.method in ['weighted', 'learned']:
            # Normalize weights to sum to 1
            normalized_weights = torch.softmax(self.weights, dim=0)

            # Weighted average
            # weights: (n_models,) -> (n_models, 1, 1)
            weights_expanded = normalized_weights.view(-1, 1, 1)
            return torch.sum(stacked * weights_expanded, dim=0)

        elif self.method == 'stacking':
            # Reshape for meta-learner: (batch, n_models)
            batch_size = stacked.shape[1]
            stacked_reshaped = stacked.squeeze(-1).permute(1, 0)  # (batch, n_models)
            return self.meta_learner(stacked_reshaped)

    def get_weights(self) -> np.ndarray:
        """Get current ensemble weights"""
        if hasattr(self, 'weights'):
            normalized = torch.softmax(self.weights, dim=0)
            return normalized.detach().cpu().numpy()
        return np.ones(self.n_models) / self.n_models

    def set_weights(self, weights: List[float]):
        """Set ensemble weights manually"""
        if not hasattr(self, 'weights'):
            raise ValueError("Cannot set weights for 'average' or 'stacking' methods")
        with torch.no_grad():
            self.weights.copy_(torch.tensor(weights, dtype=torch.float32))

    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each individual model"""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        return predictions


class VotingEnsemble(nn.Module):
    """
    Voting ensemble for classification tasks

    Supports hard voting (majority) and soft voting (probability averaging)

    Args:
        models: List of classification models
        voting: 'hard' for majority voting, 'soft' for probability averaging
    """

    def __init__(
        self,
        models: List[nn.Module],
        voting: str = 'soft',
    ):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.voting = voting
        self.n_models = len(models)

        for model in self.models:
            model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with voting"""
        predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                if self.voting == 'soft':
                    # Apply softmax if not already probabilities
                    if pred.dim() > 1 and pred.shape[-1] > 1:
                        pred = torch.softmax(pred, dim=-1)
                predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)

        if self.voting == 'soft':
            # Average probabilities
            return torch.mean(stacked, dim=0)
        else:
            # Hard voting - majority vote
            # Get class predictions from each model
            classes = torch.argmax(stacked, dim=-1)  # (n_models, batch)
            # Mode across models for each sample
            result = torch.mode(classes, dim=0).values
            return result


class BootstrapEnsemble:
    """
    Create ensemble through bootstrap aggregating (bagging)

    Trains multiple models on different bootstrap samples of the data
    for improved generalization and reduced variance.

    Args:
        model_class: Class of model to instantiate
        model_kwargs: Arguments for model initialization
        n_estimators: Number of models in ensemble
        sample_ratio: Fraction of data to sample for each model
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict,
        n_estimators: int = 5,
        sample_ratio: float = 0.8,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.models = []
        self.fitted = False

    def fit(
        self,
        train_loader,
        trainer_class,
        trainer_kwargs: Dict,
        epochs: int = 100,
        verbose: bool = True,
    ) -> 'BootstrapEnsemble':
        """
        Train ensemble with bootstrap samples

        Args:
            train_loader: DataLoader with training data
            trainer_class: Trainer class to use
            trainer_kwargs: Arguments for trainer (excluding model)
            epochs: Training epochs per model
            verbose: Print training progress
        """
        from torch.utils.data import Subset, DataLoader

        dataset = train_loader.dataset
        n_samples = len(dataset)
        sample_size = int(n_samples * self.sample_ratio)

        for i in range(self.n_estimators):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training model {i+1}/{self.n_estimators}")
                print(f"{'='*50}")

            # Create bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            bootstrap_dataset = Subset(dataset, indices)
            bootstrap_loader = DataLoader(
                bootstrap_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True
            )

            # Create and train model
            model = self.model_class(**self.model_kwargs)

            trainer = trainer_class(model=model, **trainer_kwargs)
            trainer.fit(bootstrap_loader, epochs=epochs)

            self.models.append(model)

        self.fitted = True
        return self

    def get_ensemble(self, method: str = 'average') -> EnsembleModel:
        """
        Get ensemble model from trained models

        Args:
            method: Ensemble method to use

        Returns:
            EnsembleModel instance
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before get_ensemble()")

        return EnsembleModel(self.models, method=method)


def create_diverse_ensemble(
    input_dim: int,
    hidden_dims: List[int] = [64, 128, 256],
    model_types: List[str] = ['lstm', 'gru'],
    output_dim: int = 1,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> List[nn.Module]:
    """
    Create a diverse ensemble with different architectures

    Diversity improves ensemble performance by reducing correlated errors.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden dimensions to try
        model_types: List of model types ('lstm', 'gru', 'transformer')
        output_dim: Output dimension
        num_layers: Number of layers
        dropout: Dropout rate

    Returns:
        List of untrained models
    """
    from .lstm import LSTMModel
    from .gru import GRUModel
    from .transformer import TransformerModel

    MODEL_CLASSES = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'transformer': TransformerModel,
    }

    models = []

    for model_type in model_types:
        for hidden_dim in hidden_dims:
            model_class = MODEL_CLASSES.get(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")

            if model_type == 'transformer':
                model = model_class(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_heads=4,
                )
            else:
                model = model_class(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )

            models.append(model)

    return models


def optimize_ensemble_weights(
    ensemble: EnsembleModel,
    val_loader,
    criterion: nn.Module,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
) -> EnsembleModel:
    """
    Optimize ensemble weights using validation data

    Args:
        ensemble: EnsembleModel with 'learned' method
        val_loader: Validation data loader
        criterion: Loss function
        n_iterations: Optimization iterations
        learning_rate: Learning rate for weight optimization

    Returns:
        Ensemble with optimized weights
    """
    if ensemble.method != 'learned':
        raise ValueError("Weight optimization requires 'learned' ensemble method")

    optimizer = torch.optim.Adam([ensemble.weights], lr=learning_rate)

    for iteration in range(n_iterations):
        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in val_loader:
            optimizer.zero_grad()

            predictions = ensemble(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (iteration + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            weights = ensemble.get_weights()
            print(f"Iteration {iteration+1}: Loss={avg_loss:.4f}, Weights={weights}")

    return ensemble
