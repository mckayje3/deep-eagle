"""Utility functions"""

from .metrics import mse, rmse, mae, mape, r2_score
from .helpers import set_seed, get_device, count_parameters
from .deprecation import (
    deprecate,
    deprecated,
    deprecated_argument,
    DeprecatedClass,
    warn_on_import,
)
from .feature_importance import (
    FeatureImportance,
    calculate_all_importances,
    plot_feature_importance,
    SHAPExplainer,
)

__all__ = [
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2_score",
    "set_seed",
    "get_device",
    "count_parameters",
    "deprecate",
    "deprecated",
    "deprecated_argument",
    "DeprecatedClass",
    "warn_on_import",
    "FeatureImportance",
    "calculate_all_importances",
    "plot_feature_importance",
    "SHAPExplainer",
]
