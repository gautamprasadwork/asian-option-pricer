"""
GBM path generators.

Two implementations:

1. `simulate_gbm_paths` — vectorised pseudo-random paths using NumPy.
   Supports antithetic variates (pair each path with its sign-flipped Brownian
   twin), which roughly halves the estimator variance for symmetric payoffs.

2. `simulate_gbm_paths_sobol` — quasi-Monte Carlo paths using a scrambled
   Sobol sequence and a **Brownian bridge** path construction. The bridge
   places most of each path's variance in the first few coordinates, which
   is exactly where the low-discrepancy Sobol sequence delivers the best
   uniformity. The combination typically beats pseudo-random MC by a factor
   of 5–20× at the same path count for smooth payoffs in moderate dimensions.

Both functions return prices on the m monitoring dates t_i = i·Δt for
i = 1..m (not including t=0).  This matches the discrete-Asian payoff
convention `A = (1/m) Σ S(t_i)`.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.stats import norm, qmc


# --------------------------------------------------------------------------- #
#  Pseudo-random GBM paths
# --------------------------------------------------------------------------- #
def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    n_paths: int,
    q: float = 0.0,
    antithetic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate GBM paths on m equally-spaced dates t_i = i·Δt.

    Parameters
    ----------
    S0, r, sigma, T, q : GBM parameters.
    m : number of monitoring dates per path.
    n_paths : total number of paths to return.
    antithetic : if True, generate n_paths/2 base draws and concatenate their
                 sign-flipped twins. n_paths must be even. This gives perfect
                 anti-correlation in the Brownian increments and roughly halves
                 the variance of any monotone payoff in the spot.
    rng : optional pre-seeded `np.random.Generator`. If None, a default
          (non-seeded) generator is used — callers that need reproducibility
          should pass one.

    Returns
    -------
    S : ndarray of shape (n_paths, m)
        S[i, j] = simulated price of path i at time t_{j+1}.
        Note: t=0 is NOT included; callers averaging "from t=0" should prepend
        the initial spot themselves.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")
    if rng is None:
        rng = np.random.default_rng()

    dt = T / m
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal((n_half, m))
        Z = np.concatenate([Z, -Z], axis=0)  # antithetic pairs
    else:
        Z = rng.standard_normal((n_paths, m))

    # log-increments → cumulative log-returns → prices
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    S = S0 * np.exp(log_paths)
    return S


# --------------------------------------------------------------------------- #
#  Sobol QMC paths with Brownian bridge
# --------------------------------------------------------------------------- #
def _brownian_bridge_indices(m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute the index schedule for a binary Brownian bridge over m steps.

    The bridge fills in W(t_1), ..., W(t_m) recursively: first the endpoint
    W(t_m), then W(t_{m/2}), then the midpoints of each half, etc. Variance
    is therefore front-loaded onto the first few normal draws, which is
    exactly the dimensions where Sobol gives the best uniformity.

    Returns four parallel arrays (i_left, i_right, i_mid, weight_array)
    describing each bridge step, suitable for vectorised filling.
    """
    # We process steps in BFS order of a balanced binary tree on (0, m].
    schedule_left = []
    schedule_right = []
    schedule_mid = []

    # The first sample placed is W(t_m), no left/right anchor on this one;
    # we handle that as a special case in the caller. Here we just generate
    # the intermediate bridge steps.
    queue = [(0, m)]
    while queue:
        next_queue = []
        for (lo, hi) in queue:
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            schedule_left.append(lo)
            schedule_right.append(hi)
            schedule_mid.append(mid)
            next_queue.append((lo, mid))
            next_queue.append((mid, hi))
        queue = next_queue

    return (
        np.array(schedule_left, dtype=np.int64),
        np.array(schedule_right, dtype=np.int64),
        np.array(schedule_mid, dtype=np.int64),
        np.array([], dtype=np.float64),  # placeholder (unused)
    )


def simulate_gbm_paths_sobol(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    n_paths: int,
    q: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate GBM paths using a scrambled Sobol sequence and a Brownian
    bridge construction.

    The Sobol sequence is m-dimensional with n_paths points; the d-th
    coordinate of point k is mapped through the inverse-normal CDF to obtain
    Z_k^{(d)}. The Brownian bridge then assigns these standard normals to
    the timeline such that the largest variance contributions sit in the
    lowest dimensions.

    For best results n_paths should be a power of 2 (Sobol's natural
    population size). If it isn't, scipy will warn — that's expected.

    Returns
    -------
    S : ndarray of shape (n_paths, m), prices on dates t_1..t_m.
    """
    sampler = qmc.Sobol(d=m, scramble=True, seed=seed)
    U = sampler.random(n=n_paths)
    # Clip away exact 0 and 1 before applying the inverse normal CDF
    eps = 1e-10
    U = np.clip(U, eps, 1.0 - eps)
    Z = norm.ppf(U)  # shape (n_paths, m)

    # ---- Brownian bridge ----
    # Build W(t_i) for i = 1..m from the Sobol normals using a bridge.
    dt = T / m
    sqrt_dt = np.sqrt(dt)

    W = np.zeros((n_paths, m + 1))  # W[..., 0] = 0, indices 0..m
    # First coordinate carries the endpoint W(t_m) ~ N(0, m·dt)
    W[:, m] = np.sqrt(T) * Z[:, 0]

    # Recursive bridge fill — BFS through the binary tree
    i_left, i_right, i_mid, _ = _brownian_bridge_indices(m)
    # The k-th bridge step uses Sobol dimension k+1 (since dim 0 was the endpoint)
    for k in range(len(i_left)):
        lo, hi, mid = int(i_left[k]), int(i_right[k]), int(i_mid[k])
        # Conditional on W[lo] and W[hi], the midpoint is normal with
        #   mean = ((hi-mid)·W[lo] + (mid-lo)·W[hi]) / (hi-lo)
        #   var  = (mid-lo)(hi-mid)·dt / (hi-lo)
        mean = ((hi - mid) * W[:, lo] + (mid - lo) * W[:, hi]) / (hi - lo)
        var = (mid - lo) * (hi - mid) * dt / (hi - lo)
        W[:, mid] = mean + np.sqrt(var) * Z[:, k + 1]

    # Now W has columns 0..m. Convert Brownian motion → GBM prices on
    # the monitoring dates t_1..t_m.
    drift_grid = (r - q - 0.5 * sigma * sigma) * np.arange(1, m + 1) * dt
    log_paths = drift_grid[None, :] + sigma * W[:, 1:]
    S = S0 * np.exp(log_paths)
    return S


# --------------------------------------------------------------------------- #
#  Path summary helpers used by the MC pricers
# --------------------------------------------------------------------------- #
def arithmetic_average(S_paths: np.ndarray) -> np.ndarray:
    """Arithmetic mean of each row of S_paths."""
    return S_paths.mean(axis=1)


def geometric_average(S_paths: np.ndarray) -> np.ndarray:
    """Geometric mean of each row of S_paths (computed in log space for
    numerical stability)."""
    return np.exp(np.log(S_paths).mean(axis=1))
