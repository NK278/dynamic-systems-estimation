import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Module-level logger ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO,
                  log_file: str | None = None) -> None:
    """
    Configure the root logger with a clean, timestamped format.

    Args:
        level    : Logging verbosity (e.g. logging.DEBUG / logging.INFO).
        log_file : Optional path to write logs to a file as well as stdout.
    """
    fmt = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    logger.debug("Logging initialised at level %s", logging.getLevelName(level))


# ── Model ─────────────────────────────────────────────────────────────────────
class StochasticVolatilityModel:
    """
    Defines the system and measurement models for the
    stochastic volatility tracking problem.
    """

    def __init__(self, phi: float, Q: float, R: float):
        """
        Args:
            phi : State transition coefficient (AR parameter)
            Q   : Process noise variance
            R   : Measurement noise variance
        """
        self.phi = phi
        self.Q   = Q
        self.R   = R
        logger.debug(
            "StochasticVolatilityModel created | phi=%.3f  Q=%.4f  R=%.4f",
            phi, Q, R,
        )

    def f(self, x: float) -> float:
        """State transition function: f(x) = phi * x"""
        return self.phi * x

    def h(self, x: float) -> float:
        """Nonlinear measurement function: h(x) = exp(x/2)"""
        return np.exp(x / 2)

    def H_jacobian(self, x: float) -> float:
        """
        Measurement Jacobian evaluated at x:
        H_k = dh/dx|_{x} = (1/2) * exp(x/2)
        """
        return 0.5 * np.exp(x / 2)


# ── Filter ────────────────────────────────────────────────────────────────────
class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for a scalar nonlinear system.

    Predict step (linear process model):
        x_{k|k-1}  = phi * x_{k-1|k-1}
        P_{k|k-1}  = phi^2 * P_{k-1|k-1} + Q

    Update step (linearised via Jacobian):
        H_k        = dh/dx |_{x_{k|k-1}}
        z_tilde_k  = z_k - h(x_{k|k-1})
        S_k        = H_k * P_{k|k-1} * H_k + R
        K_k        = P_{k|k-1} * H_k / S_k
        x_{k|k}    = x_{k|k-1} + K_k * z_tilde_k
        P_{k|k}    = (1 - K_k * H_k) * P_{k|k-1}
    """

    def __init__(self, model: StochasticVolatilityModel,
                 x0: float = 0.0, P0: float = 1.0):
        """
        Args:
            model : StochasticVolatilityModel instance
            x0    : Initial state estimate
            P0    : Initial error covariance
        """
        self.model = model

        # Current posterior state and covariance
        self.x_hat = x0
        self.P     = P0
        self._step = 0          # internal time-step counter

        # Storage for full trajectory
        self.x_est   = []   # posterior estimates      x_{k|k}
        self.P_est   = []   # posterior covariances    P_{k|k}
        self.x_pred  = []   # prior estimates          x_{k|k-1}
        self.P_pred  = []   # prior covariances        P_{k|k-1}
        self.innov   = []   # innovations              z_tilde_k
        self.S_innov = []   # innovation covariances   S_k
        self.K_gain  = []   # Kalman gains             K_k

        logger.info(
            "ExtendedKalmanFilter initialised | x0=%.4f  P0=%.4f", x0, P0
        )

    # ── Predict ──────────────────────────────────────────────────────────────
    def predict(self) -> tuple:
        """
        Performs the EKF predict step.

        Returns:
            (x_prior, P_prior) — prior state estimate and covariance
        """
        x_prior = self.model.f(self.x_hat)
        P_prior = ((self.model.phi ** 2) * self.P) + self.model.Q

        logger.debug(
            "k=%4d | PREDICT | x_hat=% .4f → x_prior=% .4f | "
            "P=%.5f → P_prior=%.5f",
            self._step, self.x_hat, x_prior, self.P, P_prior,
        )
        return x_prior, P_prior

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, z: float, x_prior: float, P_prior: float) -> None:
        """
        Performs the EKF update step given a measurement z.

        Args:
            z       : Observed measurement at time k
            x_prior : Prior state estimate x_{k|k-1}
            P_prior : Prior covariance     P_{k|k-1}
        """
        # Linearise measurement model at prior estimate
        H_k    = self.model.H_jacobian(x_prior)
        z_pred = self.model.h(x_prior)

        # Innovation
        innovation = z - z_pred

        # Innovation covariance and Kalman gain
        S_k = (H_k * P_prior * H_k) + self.model.R
        K_k = (P_prior * H_k )/ S_k

        # Posterior state and covariance
        self.x_hat = x_prior + (K_k * innovation)
        self.P     = (1 - K_k * H_k) * P_prior

        logger.debug(
            "k=%4d | UPDATE  | z=% .4f  z_pred=% .4f  innov=% .4f | "
            "H=%.4f  S=%.5f  K=%.4f | x_post=% .4f  P_post=%.5f",
            self._step, z, z_pred, innovation,
            H_k, S_k, K_k, self.x_hat, self.P,
        )

        # Warn if covariance goes non-positive (numerical instability)
        if self.P <= 0:
            logger.warning(
                "k=%4d | Non-positive posterior covariance P=%.6f — "
                "possible numerical instability", self._step, self.P
            )

        # Store results
        self.x_pred.append(x_prior)
        self.P_pred.append(P_prior)
        self.innov.append(innovation)
        self.S_innov.append(S_k)          # innovation covariance
        self.x_est.append(self.x_hat)
        self.P_est.append(self.P)
        self.K_gain.append(K_k)
        self._step += 1

    # ── Run ──────────────────────────────────────────────────────────────────
    def run(self, measurements: np.ndarray) -> None:
        """
        Runs the EKF over the full measurement sequence.

        Args:
            measurements : Array of observations z_1, z_2, ..., z_N
        """
        N = len(measurements)
        logger.info("Starting EKF run over %d measurements …", N)
        for z in measurements:
            x_prior, P_prior = self.predict()
            self.update(z, x_prior, P_prior)

        results = self.get_results()
        logger.info(
            "EKF run complete | N=%d | innovation mean=%.6f  std=%.6f | "
            "final x_est=%.4f  final P_est=%.6f",
            N,
            results["innov"].mean(), results["innov"].std(),
            results["x_est"][-1],   results["P_est"][-1],
        )

    # ── Results ───────────────────────────────────────────────────────────────
    def get_results(self) -> dict:
        """Returns all stored filter outputs as numpy arrays."""
        return {
            "x_est":   np.array(self.x_est),
            "P_est":   np.array(self.P_est),
            "x_pred":  np.array(self.x_pred),
            "P_pred":  np.array(self.P_pred),
            "innov":   np.array(self.innov),
            "S_innov": np.array(self.S_innov),
            "K_gain":  np.array(self.K_gain),
        }

    # ── Shared styling helper ─────────────────────────────────────────────────
    @staticmethod
    def _apply_style() -> dict:
        """Apply common rcParams and return the colour palette."""
        plt.rcParams.update({
            "font.family":       "DejaVu Sans",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "grid.alpha":        0.30,
            "grid.linestyle":    "--",
        })
        return dict(
            BLUE   = "#2E86AB",
            RED    = "#E84855",
            ORANGE = "#F4845F",
            GREEN  = "#3BB273",
            PURPLE = "#7B5EA7",
        )

    # ── Plot 1: State estimate ────────────────────────────────────────────────
    def plot_state_estimate(self,
                            x_true=None,
                            save_dir: str = "output",
                            show: bool = True):
        """
        Plot 1 — EKF posterior state estimate x̂_{k|k} vs true x_k,
        with ±3√P_{k|k} confidence band.
        Saved to: <save_dir>/plot1_state_estimate.png
        """
        c = self._apply_style()
        results = self.get_results()
        x_est, P_est = results["x_est"], results["P_est"]
        sigma = np.sqrt(P_est)
        N     = len(x_est)
        time  = np.arange(1, N + 1)
        bound = 3 * sigma

        has_truth = x_true is not None and len(x_true) == N
        fig, ax = plt.subplots(figsize=(13, 4))
        if has_truth:
            ax.plot(time, x_true, color=c["BLUE"], lw=1.2, alpha=0.85,
                    label=r"True $x_k$")
        ax.plot(time, x_est, color=c["RED"], lw=1.3,
                label=r"EKF posterior $\hat{x}_{k|k}$")
        ax.fill_between(time, x_est - bound, x_est + bound,
                        color=c["RED"], alpha=0.15,
                        label=r"$\pm 3\sqrt{P_{k|k}}$ band")
        ax.plot(time, x_est + bound, color=c["RED"], lw=0.8, ls="--", alpha=0.7)
        ax.plot(time, x_est - bound, color=c["RED"], lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("Time step $k$ (days)", fontsize=11)
        ax.set_ylabel(r"$x_k$  (log-vol)", fontsize=11)
        ax.set_title(
            r"Plot 1 — State Estimate $\hat{x}_{k|k}$ vs True $x_k$"
            r" with $\pm3\sqrt{P_{k|k}}$ Bounds",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        path = os.path.join(save_dir, "plot1_state_estimate.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", path)
        if show:
            plt.show()
        return fig

    # ── Plot 2: Estimation error ──────────────────────────────────────────────
    def plot_estimation_error(self,
                              x_true,
                              save_dir: str = "output",
                              show: bool = True):
        """
        Plot 2 — Estimation error e_k = x_k − x̂_{k|k}
        with ±3√P_{k|k} bounds.
        Saved to: <save_dir>/plot2_estimation_error.png
        """
        c = self._apply_style()
        results = self.get_results()
        x_est, P_est = results["x_est"], results["P_est"]
        sigma  = np.sqrt(P_est)
        N      = len(x_est)
        time   = np.arange(1, N + 1)
        error  = x_true - x_est          # e_k = x_k − x̂_{k|k}
        rmse   = np.sqrt(np.mean(error ** 2))
        bound  = 3 * sigma

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(time, error, color=c["ORANGE"], lw=0.9, alpha=0.9,
                label=rf"$e_k = x_k - \hat{{x}}_{{k|k}}$  (RMSE = {rmse:.4f})")
        ax.fill_between(time, bound, -bound,
                        color=c["RED"], alpha=0.13,
                        label=r"$\pm 3\sqrt{P_{k|k}}$ bound")
        ax.plot(time,  bound, color=c["RED"], lw=0.8, ls="--", alpha=0.7)
        ax.plot(time, -bound, color=c["RED"], lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("Time step $k$ (days)", fontsize=11)
        ax.set_ylabel(r"$e_k$", fontsize=11)
        ax.set_title(
            r"Plot 2 — Estimation Error $e_k = x_k - \hat{x}_{k|k}$"
            r" with $\pm3\sqrt{P_{k|k}}$ Bounds",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        logger.info("RMSE over full run: %.6f", rmse)
        path = os.path.join(save_dir, "plot2_estimation_error.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", path)
        if show:
            plt.show()
        return fig

    # ── Plot 3: Innovations ───────────────────────────────────────────────────
    def plot_innovations(self,
                         save_dir: str = "output",
                         show: bool = True):
        """
        Plot 3 — Innovation sequence z̃_k = z_k − h(x̂_{k|k-1})
        with ±3√S_k bounds (whiteness / consistency check).
        Saved to: <save_dir>/plot3_innovations.png
        """
        c = self._apply_style()
        results   = self.get_results()
        innov     = results["innov"]
        S_innov   = results["S_innov"]
        sig_innov = np.sqrt(S_innov)
        N         = len(innov)
        time      = np.arange(1, N + 1)
        bound     = 3 * sig_innov

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(time, innov, color=c["GREEN"], lw=0.75, alpha=0.9,
                label=r"$\tilde{z}_k = z_k - h(\hat{x}_{k|k-1})$")
        ax.fill_between(time,  bound, -bound,
                        color=c["GREEN"], alpha=0.12,
                        label=r"$\pm 3\sqrt{S_k}$ bound")
        ax.plot(time,  bound, color=c["GREEN"], lw=0.8, ls="--", alpha=0.7)
        ax.plot(time, -bound, color=c["GREEN"], lw=0.8, ls="--", alpha=0.7)
        ax.axhline(innov.mean(), color="black", lw=1.0, ls="--",
                   label=f"Mean = {innov.mean():.4f}  (≈ 0 ✓ if consistent)")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("Time step $k$ (days)", fontsize=11)
        ax.set_ylabel(r"$\tilde{z}_k$", fontsize=11)
        ax.set_title(
            r"Plot 3 — Innovations $\tilde{z}_k = z_k - h(\hat{x}_{k|k-1})$"
            r" with $\pm3\sqrt{S_k}$ Bounds"
            f"   [mean={innov.mean():.4f}  std={innov.std():.4f}]",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        path = os.path.join(save_dir, "plot3_innovations.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", path)
        if show:
            plt.show()
        return fig

    # ── Plot 4: Kalman gain ───────────────────────────────────────────────────
    def plot_kalman_gain(self,
                         save_dir: str = "output",
                         show: bool = True):
        """
        Plot 4 — Kalman gain K_k over time (convergence to steady state).
        Saved to: <save_dir>/plot4_kalman_gain.png
        """
        c = self._apply_style()
        K_gain = self.get_results()["K_gain"]
        N      = len(K_gain)
        time   = np.arange(1, N + 1)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(time, K_gain, color=c["PURPLE"], lw=1.0,
                label=r"Kalman gain $K_k$")
        ax.axhline(K_gain[-50:].mean(), color="black", lw=0.8, ls="--",
                   label=f"Steady-state ≈ {K_gain[-50:].mean():.4f}")
        ax.set_xlabel("Time step $k$ (days)", fontsize=11)
        ax.set_ylabel(r"$K_k$", fontsize=11)
        ax.set_title(
            "Plot 4 — Kalman Gain $K_k$ over Time  (convergence to steady state)",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        path = os.path.join(save_dir, "plot4_kalman_gain.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", path)
        if show:
            plt.show()
        return fig

    # ── Convenience: run all four plots ──────────────────────────────────────
    def plot_results(self,
                     x_true=None,
                     save_dir: str = "output",
                     show: bool = True) -> list:
        """
        Generates and saves all four diagnostic plots individually:

            plot1_state_estimate.png
            plot2_estimation_error.png   (requires x_true)
            plot3_innovations.png
            plot4_kalman_gain.png

        Args:
            x_true   : Ground-truth state array (required for Plot 2).
            save_dir : Directory to save all figures into.
            show     : Whether to call plt.show() for each figure.

        Returns:
            List of matplotlib Figure objects (length 3 without x_true, 4 with).
        """
        N = len(self.x_est)
        has_truth = x_true is not None and len(x_true) == N

        figs = []
        figs.append(self.plot_state_estimate(x_true=x_true,  save_dir=save_dir, show=show))
        if has_truth:
            figs.append(self.plot_estimation_error(x_true=x_true, save_dir=save_dir, show=show))
        else:
            logger.warning("x_true not provided — skipping Plot 2 (estimation error)")
        figs.append(self.plot_innovations(save_dir=save_dir, show=show))
        figs.append(self.plot_kalman_gain(save_dir=save_dir, show=show))
        return figs

