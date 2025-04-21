"""
Continuum: Zero-downtime ML model training and serving.

This library provides Python bindings for the Continuum Rust library,
enabling zero-downtime machine learning model updates through atomic model
swapping. It allows simultaneous model training and prediction serving
without any downtime.

Classes:
    Continuum: Main class for interacting with ML models
    ModelParameters: Configuration for model initialization
    LearningConfig: Settings for continuous learning behavior
    PredictionResponse: Result of a prediction request
    BatchPredictionResponse: Result of a batch prediction request
    ModelInfo: Information about a registered model
"""

from typing import List, Optional, Union, Dict, Any

# Import the Rust-generated Python module
from .continuum_py import (
    PyContinuum,
    PyModelParameters,
    PyLearningConfig,
    PyPredictionResponse,
    PyBatchPredictionResponse,
    PyModelInfo,
)


# Provide clean aliases with proper type annotations
class Continuum(PyContinuum):
    """
    Main interface to the Continuum ML system.

    Continuum provides zero-downtime machine learning model updates through
    atomic model swapping. It allows simultaneous training and prediction
    without any service interruptions.

    Example:
        >>> import continuum
        >>> ml_system = continuum.Continuum()
        >>> ml_system.register_model("my_model", "linear", None)
        >>> ml_system.start_continuous_learning()
    """

    pass


class ModelParameters(PyModelParameters):
    """
    Parameters for initializing a machine learning model.

    Attributes:
        with_bias: Whether to include a bias term in the model
        learning_rate: Step size for gradient-based training methods
        max_iterations: Maximum number of training iterations
        regularization: Regularization strength (for supported models)
    """

    pass


class LearningConfig(PyLearningConfig):
    """
    Configuration for continuous learning behavior.

    Attributes:
        enabled: Whether continuous learning is enabled
        interval_sec: How often to check for new training data (seconds)
        min_samples: Minimum number of samples before training
        auto_swap: Whether to automatically swap models after training
        validation_threshold: Improvement threshold for model swapping
        use_kl_divergence: Whether to use KL divergence for model comparison
    """

    pass


class PredictionResponse(PyPredictionResponse):
    """
    Response from a prediction request.

    Attributes:
        prediction: The predicted value
        model_version: Version of the model that made the prediction
    """

    pass


class BatchPredictionResponse(PyBatchPredictionResponse):
    """
    Response from a batch prediction request.

    Attributes:
        predictions: List of predicted values
        model_version: Version of the model that made the predictions
    """

    pass


class ModelInfo(PyModelInfo):
    """
    Information about a registered model.

    Attributes:
        name: Name of the model
        version: Current version of the model
        is_training: Whether the model is currently training
        stats: Performance statistics as a formatted string
    """

    pass


__all__ = [
    "Continuum",
    "ModelParameters",
    "LearningConfig",
    "PredictionResponse",
    "BatchPredictionResponse",
    "ModelInfo",
]

__version__ = "0.1.0"
