"""
Ensemble methods for combining multiple models
Provides significant accuracy improvements for predictions
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        models: list[nn.Module],
        method: str = "average",
        weights: list[float] | None = None,
        meta_hidden_dim: int = 32,
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
        if method == "weighted":
            if weights is None:
                # Equal weights by default
                weights = [1.0 / self.n_models] * self.n_models
            if len(weights) != self.n_models:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of models ({self.n_models})"
                )
            self.weights = nn.Parameter(
                torch.tensor(weights, dtype=torch.float32), requires_grad=False
            )

        elif method == "learned":
            # Learnable weights initialized equally
            self.weights = nn.Parameter(
                torch.ones(self.n_models) / self.n_models, requires_grad=True
            )

        elif method == "stacking":
            # Meta-learner that combines the flattened outputs of every base model.
            # Derive the per-model output size from the first model so stacking
            # works for arbitrary output_dim / forecast_horizon, not just scalars.
            self.output_dim = getattr(models[0], "output_dim", 1)
            self.forecast_horizon = getattr(models[0], "forecast_horizon", 1)
            self.meta_out_features = self.output_dim * self.forecast_horizon
            hidden = max(meta_hidden_dim, self.meta_out_features)
            self.meta_learner = nn.Sequential(
                nn.Linear(self.n_models * self.meta_out_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.meta_out_features),
            )
            # The meta-learner starts untrained; call fit_meta_learner() before use.
            self._meta_fitted = False

        elif method != "average":
            raise ValueError(
                f"Unknown ensemble method: {method}. Use 'average', 'weighted', 'learned', or 'stacking'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor (batch, sequence, features)

        Returns:
            Combined predictions
        """
        # Base models are frozen (already trained) — always compute their
        # predictions without tracking gradients. The trainable combination
        # ('learned' weights / 'stacking' meta-learner) runs *outside* this
        # block so its parameters still receive gradients. Do not move the
        # combination inside the no_grad block: that would silently stop the
        # weights / meta-learner from training.
        predictions = []
        with torch.no_grad():
            for model in self.models:
                predictions.append(model(x))

        # Stack predictions: (n_models, batch, *output_shape)
        # where output_shape is (output_dim,) or (forecast_horizon, output_dim)
        stacked = torch.stack(predictions, dim=0)

        if self.method == "average":
            # Simple average
            return torch.mean(stacked, dim=0)

        elif self.method in ["weighted", "learned"]:
            # Normalize weights to sum to 1
            normalized_weights = torch.softmax(self.weights, dim=0)

            # Broadcast weights over every trailing dim: (n_models,) -> (n_models, 1, 1, ...)
            weight_shape = (-1,) + (1,) * (stacked.dim() - 1)
            weights_expanded = normalized_weights.view(*weight_shape)
            return torch.sum(stacked * weights_expanded, dim=0)

        elif self.method == "stacking":
            batch_size = stacked.shape[1]
            # Flatten each model's output, then concatenate across models:
            # (n_models, batch, *out) -> (batch, n_models * out_features)
            flat = stacked.reshape(self.n_models, batch_size, -1)
            meta_in = flat.permute(1, 0, 2).reshape(batch_size, -1)
            out = self.meta_learner(meta_in)  # (batch, out_features)
            # Restore the original output shape
            if self.forecast_horizon > 1:
                out = out.view(batch_size, self.forecast_horizon, self.output_dim)
            return out

    def get_weights(self) -> np.ndarray:
        """Get current ensemble weights"""
        if hasattr(self, "weights"):
            normalized = torch.softmax(self.weights, dim=0)
            return normalized.detach().cpu().numpy()
        return np.ones(self.n_models) / self.n_models

    def set_weights(self, weights: list[float]) -> None:
        """Set ensemble weights manually"""
        if not hasattr(self, "weights"):
            raise ValueError("Cannot set weights for 'average' or 'stacking' methods")
        with torch.no_grad():
            self.weights.copy_(torch.tensor(weights, dtype=torch.float32))

    def fit_meta_learner(
        self,
        train_loader,
        criterion: nn.Module | None = None,
        epochs: int = 50,
        learning_rate: float = 0.01,
    ) -> EnsembleModel:
        """
        Train the stacking meta-learner on top of the frozen base models.

        Stacking is useless until this is called — until then the meta-learner is
        randomly initialised and produces worse results than simple averaging.
        Only the meta-learner is optimised; the base models stay frozen (their
        predictions are computed under ``torch.no_grad()`` in ``forward``).

        Args:
            train_loader: DataLoader yielding (batch_x, batch_y)
            criterion: Loss function (defaults to ``nn.MSELoss``)
            epochs: Number of passes over the data
            learning_rate: Adam learning rate for the meta-learner

        Returns:
            self, with a trained meta-learner
        """
        if self.method != "stacking":
            raise ValueError("fit_meta_learner() requires the 'stacking' ensemble method")

        criterion = criterion or nn.MSELoss()
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=learning_rate)

        self.meta_learner.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                logger.info(f"Meta-learner epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        self.meta_learner.eval()
        self._meta_fitted = True
        return self

    def get_individual_predictions(self, x: torch.Tensor) -> list[torch.Tensor]:
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
        models: list[nn.Module],
        voting: str = "soft",
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
                if self.voting == "soft":
                    # Apply softmax if not already probabilities
                    if pred.dim() > 1 and pred.shape[-1] > 1:
                        pred = torch.softmax(pred, dim=-1)
                predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)

        if self.voting == "soft":
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

    Trains multiple models on different bootstrap samples of the data.

    Note: bootstrap sampling is over windowed examples, which overlap heavily in
    a time series, so the resulting models are more correlated than in classic
    bagging — expect a smaller variance-reduction benefit than i.i.d. bagging.
    For genuinely diverse ensembles prefer ``create_diverse_ensemble`` (varied
    architectures) or block-bootstrap of contiguous segments.

    Args:
        model_class: Class of model to instantiate
        model_kwargs: Arguments for model initialization
        n_estimators: Number of models in ensemble
        sample_ratio: Fraction of data to sample for each model
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict,
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
        trainer_kwargs: dict,
        epochs: int = 100,
        verbose: bool = True,
    ) -> BootstrapEnsemble:
        """
        Train ensemble with bootstrap samples

        Args:
            train_loader: DataLoader with training data
            trainer_class: Trainer class to use
            trainer_kwargs: Arguments for trainer (excluding model)
            epochs: Training epochs per model
            verbose: Print training progress
        """
        from torch.utils.data import DataLoader, Subset

        dataset = train_loader.dataset
        n_samples = len(dataset)
        sample_size = int(n_samples * self.sample_ratio)

        for i in range(self.n_estimators):
            if verbose:
                logger.info("=" * 50)
                logger.info(f"Training model {i + 1}/{self.n_estimators}")
                logger.info("=" * 50)

            # Create bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            bootstrap_dataset = Subset(dataset, indices)
            bootstrap_loader = DataLoader(
                bootstrap_dataset, batch_size=train_loader.batch_size, shuffle=True
            )

            # Create and train model
            model = self.model_class(**self.model_kwargs)

            trainer = trainer_class(model=model, **trainer_kwargs)
            trainer.fit(bootstrap_loader, epochs=epochs)

            self.models.append(model)

        self.fitted = True
        return self

    def get_ensemble(self, method: str = "average") -> EnsembleModel:
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
    hidden_dims: list[int] | None = None,
    model_types: list[str] | None = None,
    output_dim: int = 1,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> list[nn.Module]:
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
    from .gru import GRUModel
    from .lstm import LSTMModel
    from .transformer import TransformerModel

    if hidden_dims is None:
        hidden_dims = [64, 128, 256]
    if model_types is None:
        model_types = ["lstm", "gru"]

    MODEL_CLASSES = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "transformer": TransformerModel,
    }

    models = []

    for model_type in model_types:
        for hidden_dim in hidden_dims:
            model_class = MODEL_CLASSES.get(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")

            if model_type == "transformer":
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
    if ensemble.method != "learned":
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
            logger.info(f"Iteration {iteration + 1}: Loss={avg_loss:.4f}, Weights={weights}")

    return ensemble
