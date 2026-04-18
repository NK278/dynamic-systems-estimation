"""Batch Estimation Algorithms.

This module provides Object-Oriented implementations for batch estimation methods
such as ordinary/weighted linear least squares and non-linear least squares
(Gauss-Newton method).

Classes
-------
LinearLeastSquares
    Standard and weighted linear least squares estimator.
NonlinearLeastSquares
    Iterative Gauss-Newton estimator for nonlinear observation models.
"""

import numpy as np
from scipy.linalg import inv
from .state import State


class LinearLeastSquares:
    """Linear Least Squares Estimator.

    Solves the linear batch estimation problem:
    z = H*x + v

    where v has covariance R.

    Parameters
    ----------
    model : object
        System model with the following attributes:
        - ``H`` : numpy.ndarray — observation matrix (m x n).
    R : numpy.ndarray, optional
        Measurement-noise covariance matrix (m x m). If None, Ordinary
        Least Squares is performed (default is None).
    """

    def __init__(self, model, R=None):
        self.model = model
        self.R = R

    def estimate(self, observations):
        """Estimate the state given a batch of measurements.

        Parameters
        ----------
        observations : numpy.ndarray
            Measurement vector containing the batch observations (m,).

        Returns
        -------
        State
            The estimated state containing the state vector ``x`` and covariance ``P``.
        """
        H = self.model.H
        z = observations

        if self.R is None:
            # Ordinary Least Squares: x = (H^T * H)^-1 * H^T * z
            P = inv(H.T @ H)
            x_hat = P @ H.T @ z
        else:
            # Weighted Least Squares: x = (H^T * R^-1 * H)^-1 * H^T * R^-1 * z
            R_inv = inv(self.R)
            P = inv(H.T @ R_inv @ H)
            x_hat = P @ H.T @ R_inv @ z

        # Calculate residuals and innovation covariance (if needed)
        z_pred = H @ x_hat
        res = z - z_pred
        
        # for standard batch LS, innovation_cov is typically just H*P*H^T + R
        # but often the primary focus is x and P.
        
        state = State(x=x_hat, P=P, t=None, z=z_pred, res=res, innovation_cov=None)
        return state


class NonlinearLeastSquares:
    """Nonlinear Least Squares Estimator using Gauss-Newton iteration.

    Solves the nonlinear batch estimation problem:
    z = h(x) + v

    where v has covariance R.

    Parameters
    ----------
    model : object
        System model with the following **callable** attributes:
        - ``h(x)`` : nonlinear observation function returning (m,).
        - ``H(x)`` : Jacobian of observation function returning (m x n).
    R : numpy.ndarray, optional
        Measurement-noise covariance matrix (m x m). If None, Unweighted
        Nonlinear Least Squares is performed.
    max_iter : int, optional
        Maximum number of Gauss-Newton iterations (default 10).
    tol : float, optional
        Convergence tolerance for the state update norm (default 1e-6).
    """

    def __init__(self, model, R=None, max_iter=10, tol=1e-6):
        self.model = model
        self.R = R
        self.max_iter = max_iter
        self.tol = tol

    def estimate(self, observations, x0):
        """Iteratively estimate the state given a batch of measurements and an initial guess.

        Parameters
        ----------
        observations : numpy.ndarray
            Measurement vector containing the batch observations (m,).
        x0 : numpy.ndarray
            Initial state guess (n,).

        Returns
        -------
        State
            The estimated state containing the final state vector ``x``,
            covariance ``P``, residuals ``res``, and predicted observation ``z``.
            If it fails to converge, it returns the last estimate.
        """
        x_k = x0.copy()
        
        if self.R is not None:
            R_inv = inv(self.R)

        for i in range(self.max_iter):
            # Evaluate nonlinear function and its Jacobian at current estimate
            z_pred = self.model.h(x_k)
            H_k = self.model.H(x_k)
            
            # Measurement residual
            delta_z = observations - z_pred

            # Compute update step
            if self.R is None:
                # Normal equations without weighting
                P_k = inv(H_k.T @ H_k)
                delta_x = P_k @ H_k.T @ delta_z
            else:
                # Weighted normal equations
                P_k = inv(H_k.T @ R_inv @ H_k)
                delta_x = P_k @ H_k.T @ R_inv @ delta_z

            # Update state
            x_k = x_k + delta_x

            # Check convergence
            if np.linalg.norm(delta_x) < self.tol:
                break

        # Final predicted z and residual
        z_pred_final = self.model.h(x_k)
        res_final = observations - z_pred_final
        
        # Final covariance P is approximately (H^T * R^-1 * H)^-1 evaluated at x_hat
        H_final = self.model.H(x_k)
        if self.R is None:
            P_final = inv(H_final.T @ H_final)
        else:
            P_final = inv(H_final.T @ R_inv @ H_final)

        state = State(x=x_k, P=P_final, t=None, z=z_pred_final, res=res_final, innovation_cov=None)
        return state
