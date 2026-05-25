"""
Greeks engine.

We compute the five standard Greeks (Δ, Γ, ν, Θ, ρ) using **finite differences
with common random numbers** (CRN) — the same Brownian shocks are used at the
base parameters and at every bumped parameter, so the noise cancels in the
numerator. Without CRN the bump variance dominates the Greek signal and the
resulting estimates are useless even with millions of paths; with CRN the
Greeks are typically 100×+ more accurate at the same compute budget.

We additionally compute **pathwise Delta** (Broadie-Glasserman, 1996) as an
independent estimator. The pathwise method differentiates the payoff
*before* taking the Monte Carlo expectation, giving an unbiased estimator
that does not require bumping. The two Delta estimators agreeing within
their confidence intervals is strong evidence the engine is correct.

Sign conventions
----------------
- Δ, Γ : derivatives w.r.t. spot price S₀.
- Vega: reported **per 1 percentage point** of volatility change (i.e. the raw
  ∂P/∂σ multiplied by 0.01) — this matches market convention.
- Theta: ∂P/∂t = −∂P/∂T, reported **per calendar day** (divided by 365).
  Negative for typical long-option positions (the option loses value with
  time, all else equal). Our previous code had the sign wrong.
- Rho: reported **per 1 percentage point** of rate change.
"""

from __future__ import annotations
import time
import math
from typing import Callable, Optional, Tuple
import numpy as np

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.results import GreeksResult, GreekValue, PricingResult
from engine.paths import (
    simulate_gbm_paths, arithmetic_average, geometric_average,
)
from engine.analytic import kemna_vorst_expected_payoff_discounted


# --------------------------------------------------------------------------- #
#  Internal: a CRN-aware "price-with-shared-shocks" function
# --------------------------------------------------------------------------- #
def _price_with_shocks(
    contract: OptionContract,
    Z: np.ndarray,
    use_cv: bool = True,
) -> Tuple[float, float]:
    """
    Price `contract` using a pre-generated matrix of standard-normal shocks Z.

    Same shocks → same Brownian paths in distribution; bumping a parameter
    while holding Z fixed implements the common-random-numbers technique.

    Returns (price, std_error_of_estimator).

    use_cv=True applies the geometric-Asian control variate when applicable
    (arithmetic + fixed strike); for other contracts falls back to plain MC.
    """
    n_paths, m = Z.shape
    dt = contract.T / m
    drift = (contract.r - contract.q - 0.5 * contract.sigma ** 2) * dt
    diffusion = contract.sigma * math.sqrt(dt)

    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    S_paths = contract.S0 * np.exp(log_paths)

    # Arithmetic / geometric averages on the same paths
    A = arithmetic_average(S_paths)
    G = geometric_average(S_paths)
    S_T = S_paths[:, -1]
    K = contract.K
    disc = math.exp(-contract.r * contract.T)

    # Payoff
    averaging = contract.averaging
    A_used = A if averaging == Averaging.ARITHMETIC else G
    if contract.strike_type == StrikeType.FIXED:
        if contract.is_call:
            Y = np.maximum(A_used - K, 0.0)
        else:
            Y = np.maximum(K - A_used, 0.0)
    else:
        if contract.is_call:
            Y = np.maximum(S_T - A_used, 0.0)
        else:
            Y = np.maximum(A_used - S_T, 0.0)
    Y_disc = disc * Y

    cv_applicable = (
        use_cv
        and contract.averaging == Averaging.ARITHMETIC
        and contract.strike_type == StrikeType.FIXED
    )

    if not cv_applicable:
        price = float(np.mean(Y_disc))
        se = float(np.std(Y_disc, ddof=1) / math.sqrt(n_paths))
        return price, se

    # CV adjustment using the (geometric, fixed-strike) Asian as control
    if contract.is_call:
        X = np.maximum(G - K, 0.0)
    else:
        X = np.maximum(K - G, 0.0)
    X_disc = disc * X
    mu_X = kemna_vorst_expected_payoff_discounted(contract)

    var_X = float(np.var(X_disc, ddof=1))
    if var_X < 1e-14:
        price = float(np.mean(Y_disc))
        se = float(np.std(Y_disc, ddof=1) / math.sqrt(n_paths))
        return price, se

    cov_YX = float(np.cov(Y_disc, X_disc, ddof=1)[0, 1])
    beta = cov_YX / var_X
    Z_adj = Y_disc - beta * (X_disc - mu_X)
    price = float(np.mean(Z_adj))
    se = float(np.std(Z_adj, ddof=1) / math.sqrt(n_paths))
    return price, se


# --------------------------------------------------------------------------- #
#  Finite-difference Greeks with common random numbers
# --------------------------------------------------------------------------- #
def compute_greeks(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: Optional[int] = 42,
    use_cv: bool = True,
    bumps: Optional[dict] = None,
) -> GreeksResult:
    """
    Compute Δ, Γ, ν, Θ, ρ by finite difference with **common random numbers**.

    Algorithm
    ---------
    1. Generate one Z matrix of shape (n_paths, m) using the given seed.
    2. Price the base contract using these shocks.
    3. For each parameter, price (base + bump) and (base − bump) using the
       SAME Z matrix.
    4. Form the FD estimator and propagate the per-path standard error.

    Bump sizes
    ----------
    Defaults reflect what real desks use for these estimators:
      - dS  = 1% of S0       (Δ, Γ)
      - dσ  = 0.0025 (=0.25pp) (ν)
      - dT  = 1/365          (Θ — daily)
      - dr  = 0.0001 (=1bp)   (ρ)

    Returns
    -------
    GreeksResult with point estimate + standard error per Greek.
    """
    if bumps is None:
        bumps = {
            "S": 0.01 * contract.S0,      # absolute bump in S0
            "sigma": 0.0025,               # absolute bump in σ
            "T": 1.0 / 365.0,              # absolute bump in T (one day)
            "r": 0.0001,                   # absolute bump in r (1 bp)
        }

    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, contract.m))

    # --- Base ---
    p0, se0 = _price_with_shocks(contract, Z, use_cv=use_cv)

    # --- Δ and Γ: central difference in S0 ---
    dS = bumps["S"]
    p_S_up,   se_S_up   = _price_with_shocks(contract.with_(S0=contract.S0 + dS), Z, use_cv=use_cv)
    p_S_down, se_S_down = _price_with_shocks(contract.with_(S0=contract.S0 - dS), Z, use_cv=use_cv)
    delta_v = (p_S_up - p_S_down) / (2 * dS)
    gamma_v = (p_S_up - 2 * p0 + p_S_down) / (dS ** 2)
    delta_se = math.sqrt(se_S_up ** 2 + se_S_down ** 2) / (2 * dS)
    gamma_se = math.sqrt(se_S_up ** 2 + 4 * se0 ** 2 + se_S_down ** 2) / (dS ** 2)

    # --- ν Vega: forward difference in σ, reported per 1% vol ---
    dsig = bumps["sigma"]
    p_sig_up, se_sig_up = _price_with_shocks(contract.with_(sigma=contract.sigma + dsig), Z, use_cv=use_cv)
    p_sig_dn, se_sig_dn = _price_with_shocks(contract.with_(sigma=max(contract.sigma - dsig, 1e-6)), Z, use_cv=use_cv)
    vega_raw = (p_sig_up - p_sig_dn) / (2 * dsig)
    vega_se_raw = math.sqrt(se_sig_up ** 2 + se_sig_dn ** 2) / (2 * dsig)
    vega_v = vega_raw * 0.01     # per 1 percentage point
    vega_se = vega_se_raw * 0.01

    # --- Θ Theta: ∂P/∂t = -∂P/∂T, per calendar day ---
    dT = bumps["T"]
    if contract.T - dT > 0:
        p_T_down, se_T_down = _price_with_shocks(contract.with_(T=contract.T - dT), Z, use_cv=use_cv)
        # ∂P/∂T ≈ (p0 - p_T_down) / dT, so Θ = -∂P/∂T = (p_T_down - p0)/dT
        theta_raw = (p_T_down - p0) / dT
        theta_se_raw = math.sqrt(se0 ** 2 + se_T_down ** 2) / dT
    else:
        theta_raw = 0.0
        theta_se_raw = 0.0
    theta_v = theta_raw / 365.0     # per calendar day
    theta_se = theta_se_raw / 365.0

    # --- ρ Rho: forward difference in r, reported per 1% rate ---
    dr = bumps["r"]
    p_r_up, se_r_up = _price_with_shocks(contract.with_(r=contract.r + dr), Z, use_cv=use_cv)
    p_r_dn, se_r_dn = _price_with_shocks(contract.with_(r=contract.r - dr), Z, use_cv=use_cv)
    rho_raw = (p_r_up - p_r_dn) / (2 * dr)
    rho_se_raw = math.sqrt(se_r_up ** 2 + se_r_dn ** 2) / (2 * dr)
    rho_v = rho_raw * 0.01          # per 1 percentage point of r
    rho_se = rho_se_raw * 0.01

    # --- Pathwise Delta (cross-check) ---
    pw_delta = pathwise_delta(contract, Z=Z)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return GreeksResult(
        delta=GreekValue(delta_v, delta_se),
        gamma=GreekValue(gamma_v, gamma_se),
        vega=GreekValue(vega_v, vega_se),
        theta=GreekValue(theta_v, theta_se),
        rho=GreekValue(rho_v, rho_se),
        pathwise_delta=pw_delta,
        method="Finite Difference + Common Random Numbers (CV)" if use_cv
               else "Finite Difference + Common Random Numbers",
        runtime_ms=runtime_ms,
    )


# --------------------------------------------------------------------------- #
#  Pathwise Delta (Broadie-Glasserman, 1996)
# --------------------------------------------------------------------------- #
def pathwise_delta(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: Optional[int] = 42,
    Z: Optional[np.ndarray] = None,
) -> GreekValue:
    """
    Pathwise estimator of ∂P/∂S₀ for the discrete Asian option.

    Derivation (fixed-strike call, arithmetic averaging)
    ----------------------------------------------------
    For each path,  S(t_i) = S₀ · M_i   where M_i depends only on the Brownian
    shocks (not on S₀). The average is A = (1/m) Σ S(t_i) = S₀ · M̄ with
    M̄ = (1/m) Σ M_i. The payoff is `max(S₀ M̄ - K, 0)`, which is a.s.
    differentiable in S₀ (the kink at S₀ M̄ = K has measure zero), and

        ∂payoff/∂S₀ = M̄ · 𝟙{S₀ M̄ > K}.

    Discounting:  Δ_pw = e^{-rT} · M̄ · 𝟙{S₀ M̄ > K}.

    The pathwise estimator is the sample mean of this quantity.

    For other contract variants:
      - put / call:    flip the indicator sign convention.
      - geometric avg: replace M̄ with the geometric mean of the M_i.
      - floating strike: differentiate `max(S_T − A, 0) = S₀·(M_T − M̄)·𝟙{…}`.
    """
    if Z is None:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, contract.m))
    n_paths, m = Z.shape

    dt = contract.T / m
    drift = (contract.r - contract.q - 0.5 * contract.sigma ** 2) * dt
    diffusion = contract.sigma * math.sqrt(dt)

    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    M = np.exp(log_paths)                     # shape (N, m)
    S_paths = contract.S0 * M
    disc = math.exp(-contract.r * contract.T)

    if contract.strike_type == StrikeType.FIXED:
        if contract.averaging == Averaging.ARITHMETIC:
            M_bar = M.mean(axis=1)
            A = contract.S0 * M_bar
            if contract.is_call:
                payoff_grad = disc * M_bar * (A > contract.K)
            else:
                payoff_grad = -disc * M_bar * (A < contract.K)
        else:  # geometric
            M_geo = np.exp(np.log(M).mean(axis=1))
            G = contract.S0 * M_geo
            if contract.is_call:
                payoff_grad = disc * M_geo * (G > contract.K)
            else:
                payoff_grad = -disc * M_geo * (G < contract.K)
    else:
        # Floating strike: payoff = max(S_T - A, 0) (call) = S0 · max(M_T - M̄, 0)
        if contract.averaging == Averaging.ARITHMETIC:
            M_bar = M.mean(axis=1)
        else:
            M_bar = np.exp(np.log(M).mean(axis=1))
        M_T = M[:, -1]
        if contract.is_call:
            payoff_grad = disc * (M_T - M_bar) * (M_T > M_bar)
        else:
            payoff_grad = disc * (M_bar - M_T) * (M_bar > M_T)

    val = float(np.mean(payoff_grad))
    se = float(np.std(payoff_grad, ddof=1) / math.sqrt(n_paths))
    return GreekValue(val, se)
