"""
Monte Carlo pricers for Asian options.

Four estimators, each progressively lower variance:

1. `price_mc`           — plain pseudo-random MC.
2. `price_mc_antithetic`— MC with antithetic variates (~2× variance reduction).
3. `price_mc_cv`        — MC with antithetic + **geometric-Asian control variate**
                           (typically 50–500× variance reduction for arithmetic
                           Asians; uses Kemna-Vorst as the known mean of the
                           control).
4. `price_qmc`          — quasi-Monte Carlo with Sobol + Brownian bridge.

All four return a `PricingResult` and produce *unbiased* estimates of the
arithmetic-Asian price; the differences are purely in standard error and
runtime.
"""

from __future__ import annotations
import time
from typing import Optional
import numpy as np

from engine.contracts import (
    OptionContract, OptionType, Averaging, StrikeType,
)
from engine.results import PricingResult
from engine.paths import (
    simulate_gbm_paths, simulate_gbm_paths_sobol,
    arithmetic_average, geometric_average,
)
from engine.analytic import kemna_vorst_expected_payoff_discounted


# --------------------------------------------------------------------------- #
#  Payoff functions
# --------------------------------------------------------------------------- #
def _compute_payoffs(
    S_paths: np.ndarray,
    contract: OptionContract,
    averaging_override: Optional[Averaging] = None,
) -> np.ndarray:
    """
    Compute the undiscounted payoff of `contract` on each simulated path.

    `averaging_override` lets the CV estimator compute geometric payoffs
    from arithmetic-Asian paths without rebuilding the simulation.
    """
    averaging = averaging_override if averaging_override is not None else contract.averaging
    if averaging == Averaging.ARITHMETIC:
        A = arithmetic_average(S_paths)
    else:
        A = geometric_average(S_paths)

    K = contract.K
    S_T = S_paths[:, -1]

    if contract.strike_type == StrikeType.FIXED:
        if contract.is_call:
            return np.maximum(A - K, 0.0)
        return np.maximum(K - A, 0.0)
    # Floating strike: strike = final spot
    if contract.is_call:
        return np.maximum(S_T - A, 0.0)
    return np.maximum(A - S_T, 0.0)


# --------------------------------------------------------------------------- #
#  1. Plain Monte Carlo
# --------------------------------------------------------------------------- #
def price_mc(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: Optional[int] = 42,
) -> PricingResult:
    """Plain pseudo-random Monte Carlo (no variance reduction)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    S_paths = simulate_gbm_paths(
        contract.S0, contract.r, contract.sigma, contract.T, contract.m,
        n_paths=n_paths, q=contract.q, antithetic=False, rng=rng,
    )
    payoffs = _compute_payoffs(S_paths, contract)
    discounted = np.exp(-contract.r * contract.T) * payoffs
    price = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return PricingResult(
        price=price, std_error=se, method="Plain Monte Carlo",
        n_paths=n_paths, runtime_ms=runtime_ms,
    )


# --------------------------------------------------------------------------- #
#  2. Monte Carlo with antithetic variates
# --------------------------------------------------------------------------- #
def price_mc_antithetic(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: Optional[int] = 42,
) -> PricingResult:
    """Pseudo-random MC with antithetic variates (~2× variance reduction)."""
    if n_paths % 2 != 0:
        n_paths += 1
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    S_paths = simulate_gbm_paths(
        contract.S0, contract.r, contract.sigma, contract.T, contract.m,
        n_paths=n_paths, q=contract.q, antithetic=True, rng=rng,
    )
    payoffs = _compute_payoffs(S_paths, contract)
    discounted = np.exp(-contract.r * contract.T) * payoffs

    # SE for antithetic: variance is across antithetic *pairs*, not individuals
    n_pairs = n_paths // 2
    pair_means = 0.5 * (discounted[:n_pairs] + discounted[n_pairs:])
    price = float(np.mean(pair_means))
    se = float(np.std(pair_means, ddof=1) / np.sqrt(n_pairs))
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return PricingResult(
        price=price, std_error=se, method="MC + Antithetic Variates",
        n_paths=n_paths, runtime_ms=runtime_ms,
    )


# --------------------------------------------------------------------------- #
#  3. Monte Carlo with geometric-Asian control variate
# --------------------------------------------------------------------------- #
def price_mc_cv(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: Optional[int] = 42,
) -> PricingResult:
    """
    Pseudo-random MC with antithetic variates AND a geometric-Asian
    control variate.

    The control variate construction
    --------------------------------
    Let Y_i = discounted arithmetic-Asian payoff on path i (what we want),
    X_i = discounted geometric-Asian payoff on the *same path* (the control).
    The geometric Asian has a known expectation `mu_X = E[X]` from Kemna-Vorst.
    The CV estimator is

        Ŷ_cv = (1/N) Σ_i [ Y_i  -  β (X_i − mu_X) ],   β = Cov(Y, X) / Var(X)

    which is unbiased for E[Y] and has variance

        Var(Ŷ_cv) = Var(Ŷ) · (1 − ρ²)

    where ρ = corr(Y, X). For arithmetic Asians under GBM, ρ ≈ 0.99, so
    variance is reduced by roughly two orders of magnitude.

    Applicability
    -------------
    Falls back to plain antithetic MC if the contract is geometric (CV would
    have ρ=1 and be redundant) or floating-strike (the geometric expectation
    used here is for fixed-strike — extending CV to floating-strike Asians
    is a known extension we don't pursue here).
    """
    # Sanity: CV with KV mean only makes sense for fixed-strike arithmetic
    if contract.averaging == Averaging.GEOMETRIC:
        return price_mc_antithetic(contract, n_paths=n_paths, seed=seed)
    if contract.strike_type == StrikeType.FLOATING:
        return price_mc_antithetic(contract, n_paths=n_paths, seed=seed)

    if n_paths % 2 != 0:
        n_paths += 1

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    S_paths = simulate_gbm_paths(
        contract.S0, contract.r, contract.sigma, contract.T, contract.m,
        n_paths=n_paths, q=contract.q, antithetic=True, rng=rng,
    )

    Y = _compute_payoffs(S_paths, contract)                                   # arithmetic
    X = _compute_payoffs(S_paths, contract, averaging_override=Averaging.GEOMETRIC)  # geometric

    disc = np.exp(-contract.r * contract.T)
    Y_disc = disc * Y
    X_disc = disc * X

    # Known expectation of the geometric-Asian discounted payoff
    mu_X = kemna_vorst_expected_payoff_discounted(contract)

    # Estimate β from the sample (introduces a tiny finite-sample bias that
    # vanishes as N → ∞; standard practice)
    var_X = float(np.var(X_disc, ddof=1))
    if var_X < 1e-14:
        # Degenerate — the control variate has no variance (e.g. deep OTM with K huge).
        # Fall back to antithetic-only.
        return price_mc_antithetic(contract, n_paths=n_paths, seed=seed)

    cov_YX = float(np.cov(Y_disc, X_disc, ddof=1)[0, 1])
    beta = cov_YX / var_X

    # CV-adjusted per-path payoffs, then average across antithetic pairs
    Z = Y_disc - beta * (X_disc - mu_X)
    n_pairs = n_paths // 2
    pair_means = 0.5 * (Z[:n_pairs] + Z[n_pairs:])
    price = float(np.mean(pair_means))
    se = float(np.std(pair_means, ddof=1) / np.sqrt(n_pairs))

    # Correlation diagnostic
    rho = cov_YX / np.sqrt(var_X * float(np.var(Y_disc, ddof=1) + 1e-30))

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return PricingResult(
        price=price, std_error=se,
        method="MC + Antithetic + Geometric Control Variate",
        n_paths=n_paths, runtime_ms=runtime_ms,
        extra={"beta": beta, "rho": float(rho), "mu_X": mu_X},
    )


# --------------------------------------------------------------------------- #
#  4. Quasi-Monte Carlo (Sobol + Brownian bridge)
# --------------------------------------------------------------------------- #
def price_qmc(
    contract: OptionContract,
    n_paths: int = 16_384,
    seed: Optional[int] = 42,
) -> PricingResult:
    """
    Quasi-Monte Carlo with a scrambled Sobol sequence and Brownian-bridge
    path construction.

    Standard-error reporting note
    -----------------------------
    QMC has *deterministic* error bounds rather than statistical ones, so a
    "standard error" is only meaningful via randomization (Owen scrambling).
    We report the empirical sample standard deviation divided by √N as a
    rough, slightly-conservative confidence indicator — this is the standard
    practice for scrambled-Sobol QMC reporting.
    """
    t0 = time.perf_counter()
    S_paths = simulate_gbm_paths_sobol(
        contract.S0, contract.r, contract.sigma, contract.T, contract.m,
        n_paths=n_paths, q=contract.q, seed=seed,
    )
    payoffs = _compute_payoffs(S_paths, contract)
    discounted = np.exp(-contract.r * contract.T) * payoffs
    price = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return PricingResult(
        price=price, std_error=se,
        method="Quasi-MC (Sobol + Brownian Bridge)",
        n_paths=n_paths, runtime_ms=runtime_ms,
    )
