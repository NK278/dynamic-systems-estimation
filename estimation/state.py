"""Data containers for the estimation package.

Classes
-------
State
    Data container for the estimator state (x, P, t, z, …).
"""

import numpy as np


class State:
    """Container for the estimator state at a given time instant.

    Attributes
    ----------
    x : numpy.ndarray or None
        State vector.
    P : numpy.ndarray or None
        State covariance matrix.
    t : float or None
        Current time.
    z : numpy.ndarray or None
        Predicted observation.
    res : numpy.ndarray or None
        Innovation (measurement residual).
    innovation_cov : numpy.ndarray or None
        Innovation covariance matrix.
    """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            self.x = None
            self.P = None
            self.t = None
            self.z = None
        elif len(kwargs) == 3:
            self.x = kwargs['x']
            self.P = kwargs['P']
            self.t = kwargs['t']
        else:
            self.x = kwargs.get('x')
            self.P = kwargs.get('P')
            self.t = kwargs.get('t')
            self.z = kwargs.get('z')
            self.res = kwargs.get('res')
            self.innovation_cov = kwargs.get('innovation_cov')
