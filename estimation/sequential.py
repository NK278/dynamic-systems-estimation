"""Sequential Estimation Algorithms.

This module provides Object-Oriented implementations for sequential 
estimation methods such as the Kalman Filter (KF), Extended Kalman 
Filter (EKF), and Unscented Kalman Filter (UKF).

Classes
-------
KalmanFilter
    Linear (discrete-time) Kalman filter.
ExtendedKalmanFilter
    Continuous-Discrete Extended Kalman filter for nonlinear systems.
UnscentedKalmanFilter
    Discrete-time Unscented Kalman Filter using the Unscented Transform.
"""

import numpy as np
from scipy.linalg import inv, cholesky
from scipy.integrate import odeint
from .state import State


# ---------------------------------------------------------------------------
# Linear Kalman Filter (Discrete-Time)
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Linear Kalman filter (continuous-discrete formulation).

    The state is propagated forward by integrating the continuous-time
    dynamics ODE with ``scipy.integrate.odeint``, while the covariance
    is propagated with the discrete-time equation
    ``P⁻ = Φ P⁺ Φᵀ + Q``.

    Parameters
    ----------
    model : object
        System model with the following attributes:

        - ``dynamics`` : callable ``f(x, t)`` — continuous-time state ODE
          (compatible with ``odeint``).
        - ``Phi`` : numpy.ndarray — discrete state-transition matrix.
        - ``h``   : callable ``h(x)`` — observation function.
        - ``H``   : numpy.ndarray — observation matrix.
    Q : numpy.ndarray
        Process-noise covariance matrix (n × n).
    R : numpy.ndarray
        Measurement-noise covariance matrix (m × m).
    state0 : State
        Initial state (must have ``x``, ``P``, ``t``).
    dT : float
        Sampling period [s].
    """

    def __init__(self, model, Q, R, state0, dT):
        self.model = model
        self.Q = Q
        self.R = R
        self.state_post = state0
        self.state_priori = State()
        self.dt = dT

    # -- Prediction (time update) -------------------------------------------

    def predict(self):
        """Propagate the state and covariance one step forward."""
        x = self.state_post.x
        P = self.state_post.P
        t = self.state_post.t

        Phi = self.model.Phi

        # State propagation via odeint
        x_priori = odeint(self.model.dynamics, x, [t, t + self.dt])[-1, :]

        # Covariance propagation (discrete)
        P_priori = Phi @ P @ Phi.T + self.Q

        # Predicted observation
        z_priori = self.model.h(x_priori)

        self.state_priori.x = x_priori
        self.state_priori.P = P_priori
        self.state_priori.z = z_priori
        self.state_priori.t = t + self.dt

    # -- Correction (measurement update) ------------------------------------

    def correct(self, observation):
        """Update the state with a new measurement.

        Uses the Joseph-form covariance update for numerical stability.

        Parameters
        ----------
        observation : numpy.ndarray
            Measurement vector (m,).
        """
        x_priori = self.state_priori.x
        P_priori = self.state_priori.P
        z_priori = self.state_priori.z
        H = self.model.H
        R = self.R
        n = x_priori.shape[0]

        residue = observation - z_priori

        # Innovation covariance
        S = H @ P_priori @ H.T + R

        # Kalman gain
        K = P_priori @ H.T @ inv(S)

        # State update
        self.state_post.x = x_priori + K @ residue

        # Covariance update (Joseph form)
        I_KH = np.eye(n) - K @ H
        self.state_post.P = I_KH @ P_priori @ I_KH.T + K @ R @ K.T

        self.state_post.t = self.state_priori.t
        self.state_post.res = residue
        self.state_post.innovation_cov = S


# ---------------------------------------------------------------------------
# Extended Kalman Filter (Continuous-Discrete)
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """Continuous-Discrete Extended Kalman Filter.

    Both the state and the covariance are propagated by integrating their
    respective continuous-time ODEs with ``scipy.integrate.odeint``.
    The measurement update is the standard EKF discrete-time correction
    using a nonlinear observation function and its Jacobian.

    Parameters
    ----------
    model : object
        System model with the following **callable** attributes:

        - ``dynamics(X, t)`` — state ODE  (odeint-compatible).
        - ``Phi(X, t)``      — Jacobian of dynamics w.r.t. state (F matrix).
        - ``h(X)``           — nonlinear observation function.
        - ``H(X)``           — Jacobian of observation function.
    Q : numpy.ndarray
        Process-noise covariance (continuous-time spectral density, n × n).
    R : numpy.ndarray
        Measurement-noise covariance matrix (m × m).
    state0 : State
        Initial state (must have ``x``, ``P``, ``t``).
    dT : float
        Sampling period [s].
    """

    def __init__(self, model, Q, R, state0, dT):
        self.model = model
        self.Q = Q
        self.R = R
        self.state_post = state0
        self.state_priori = State()
        self.dt = dT

    # -- internal: augmented ODE for odeint ---------------------------------

    def _augmented_ode(self, y_aug, t):
        """Joint state + covariance ODE for odeint.

        Parameters
        ----------
        y_aug : numpy.ndarray
            Augmented vector  [x (n,) | P_flat (n*n,)].
        t : float
            Current time (used by odeint).

        Returns
        -------
        dy : numpy.ndarray
            Time derivative of the augmented vector.
        """
        n = self.state_post.x.shape[0]
        X = y_aug[:n]
        P = y_aug[n:].reshape((n, n))

        # State derivative
        dX = self.model.dynamics(X, t)

        # Jacobian at current state
        F = self.model.Phi(X, t)

        # Covariance derivative:  Ṗ = F P + P Fᵀ + Q
        dP = F @ P + P @ F.T + self.Q

        return np.concatenate([dX, dP.flatten()])

    # -- Prediction (time update) -------------------------------------------

    def predict(self):
        """Propagate state and covariance by integrating the ODEs."""
        X = self.state_post.x.copy()
        P = self.state_post.P.copy()
        t = self.state_post.t
        n = X.shape[0]

        # Build augmented initial condition
        y0 = np.concatenate([X, P.flatten()])

        # Integrate over one sampling interval
        t_span = [t, t + self.dt]
        y_out = odeint(self._augmented_ode, y0, t_span)[-1, :]

        self.state_priori.x = y_out[:n]
        self.state_priori.P = y_out[n:].reshape((n, n))
        self.state_priori.t = t + self.dt

    # -- Correction (measurement update) ------------------------------------

    def correct(self, observation):
        """EKF measurement update using nonlinear h(x) and its Jacobian.

        Uses the Joseph-form covariance update for numerical stability.

        Parameters
        ----------
        observation : numpy.ndarray
            Measurement vector (m,).
        """
        X_hat = self.state_priori.x
        P = self.state_priori.P
        n = X_hat.shape[0]

        # Predicted measurement and Jacobian
        z_pred = self.model.h(X_hat)
        H_k = self.model.H(X_hat)
        R = self.R

        # Innovation
        residue = observation - z_pred

        # Innovation covariance
        S = H_k @ P @ H_k.T + R

        # Kalman gain
        K = P @ H_k.T @ inv(S)

        # State update
        self.state_post.x = X_hat + K @ residue

        # Covariance update (Joseph form)
        I_KH = np.eye(n) - K @ H_k
        self.state_post.P = I_KH @ P @ I_KH.T + K @ R @ K.T

        self.state_post.t = self.state_priori.t
        self.state_post.res = residue
        self.state_post.innovation_cov = S


# ---------------------------------------------------------------------------
# Unscented Kalman Filter (Discrete-discrete formulation)
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """Discrete-time Unscented Kalman Filter.

    Uses the Unscented Transform to propagate the mean and covariance
    through the non-linear dynamics and measurement models.

    Parameters
    ----------
    model : object
        System model with the following **callable** attributes:
        - ``f(X)`` — nonlinear discrete-time state transition function.
        - ``h(X)`` — nonlinear observation function.
    Q : numpy.ndarray
        Process-noise covariance matrix (discrete-time, n × n).
    R : numpy.ndarray
        Measurement-noise covariance matrix (m × m).
    state0 : State
        Initial state (must have ``x``, ``P``, ``t``).
    dT : float
        Sampling period [s].
    alpha : float, optional
        (Removed to match Lecture 13 formulation, which uses original UT).
    beta : float, optional
        (Removed to match Lecture 13 formulation).
    kappa : float, optional
        Scaling parameter (default: 0.0, usually chosen such that n+kappa=3).
    """

    def __init__(self, model, Q, R, state0, dT, kappa=0.0):
        self.model = model
        self.Q = Q
        self.R = R
        self.state_post = state0
        self.state_priori = State()
        self.dt = dT

        self.n = len(state0.x)
        self.kappa = kappa
        self.gamma = self.n + self.kappa

        self.Wm, self.Wc = self._compute_weights()

    def _compute_weights(self):
        """Computes the weights for the Unscented Transform."""
        Wm = np.full(2 * self.n + 1, 1.0 / (2 * self.gamma))
        Wc = np.full(2 * self.n + 1, 1.0 / (2 * self.gamma))
        
        Wm[0] = self.kappa / self.gamma
        Wc[0] = self.kappa / self.gamma
        return Wm, Wc

    def _generate_sigma_points(self, x, P):
        """Generates 2n+1 sigma points."""
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        
        # Symmetrize to mitigate integration rounding errors
        P = (P + P.T) / 2.0
        try:
            # sqrt(P) based on Cholesky decomposition
            # Scipy cholesky returns upper triangular U where U.T @ U = P.
            P_scaled = self.gamma * P
            # Add tiny regularization proportional to diagonal
            #P_scaled += np.diag(np.abs(np.diag(P_scaled))) * 1e-12
            sqrt_P = cholesky(P_scaled, lower=False)
        except np.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails due to semi-definiteness
            U, S, Vh = np.linalg.svd(self.gamma * P)
            sqrt_P = np.dot(np.diag(np.sqrt(np.maximum(S, 0))), Vh)

        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[self.n + i + 1] = x - sqrt_P[i]
            
        return sigma_points

    def predict(self):
        """Unscented Transform for Time Update."""
        x = self.state_post.x
        P = self.state_post.P
        t = self.state_post.t

        # 1. Generate Sigma Points
        self.sigmas_f = self._generate_sigma_points(x, P)
        
        # 2. Pass sigma points through nonlinear dynamics
        sigmas_f_propagated = np.zeros_like(self.sigmas_f)
        for i in range(2 * self.n + 1):
            sigmas_f_propagated[i] = self.model.f(self.sigmas_f[i])
            
        # 3. Compute predicted mean
        x_priori = np.dot(self.Wm, sigmas_f_propagated)
        
        # 4. Compute predicted covariance
        P_priori = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = sigmas_f_propagated[i] - x_priori
            P_priori += self.Wc[i] * np.outer(diff, diff)
            
        self.sigmas_f_propagated = sigmas_f_propagated # Store for correction step
        self.state_priori.x = x_priori
        self.state_priori.P = P_priori
        self.state_priori.t = t + self.dt

    def correct(self, observation):
        """Unscented Transform for Measurement Update."""
        x_priori = self.state_priori.x
        P_priori = self.state_priori.P

        # 1. Pass propagated sigma points through nonlinear measurement model
        m = len(observation)
        sigmas_h = np.zeros((2 * self.n + 1, m))
        for i in range(2 * self.n + 1):
            sigmas_h[i] = self.model.h(self.sigmas_f_propagated[i])
            
        # 2. Compute predicted measurement
        z_pred = np.dot(self.Wm, sigmas_h)
        
        # 3. Compute Innovation Covariance (S) and Cross-Covariance (Pxz)
        S = self.R.copy()
        Pxz = np.zeros((self.n, m))
        
        for i in range(2 * self.n + 1):
            z_diff = sigmas_h[i] - z_pred
            x_diff = self.sigmas_f_propagated[i] - x_priori
            
            S += self.Wc[i] * np.outer(z_diff, z_diff)
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
            
        # 4. Kalman Gain
        K = np.dot(Pxz, inv(S))
        
        # 5. Innovation and State Update
        residue = observation - z_pred
        self.state_post.x = x_priori + np.dot(K, residue)
        self.state_post.P = P_priori - np.dot(K, np.dot(S, K.T))
        
        self.state_post.t = self.state_priori.t
        self.state_post.res = residue
        self.state_post.z = z_pred
        self.state_post.innovation_cov = S
