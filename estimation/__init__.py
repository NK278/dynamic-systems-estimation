"""Estimation Theory Python Package.

This package provides a standardized API for batch and sequential estimation
algorithms designed for use in the dynamic systems coursework.
"""

from .state import State
from .batch import LinearLeastSquares, NonlinearLeastSquares
from .sequential import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

__all__ = [
    'State',
    'LinearLeastSquares',
    'NonlinearLeastSquares',
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter',
]
