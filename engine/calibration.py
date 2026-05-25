"""
Implied volatility calibration.

Given an observed market price for the contract, find σ such that the
model price equals the market price. This is the *inverse problem* and is
how every option desk actually thinks: market prices come from trades,
volatilities are derived.

Method
------
We use **Brent's method** (`scipy.optimize.brentq`) on the function

    f(σ) = model_price(σ) − market_price

bracketed between σ_low = 0.01% and σ_high = 500%. Brent's method
combines bisection (guaranteed convergence) with inverse quadratic
interpolation (superlinear convergence when smooth), giving robust and
fast root-finding for a monotone, increasing function like option-price-
in-σ.

Caveat
------
For arithmetic Asians, evaluating `model_price(σ)` requires re-running
the Monte Carlo at each candidate σ. To keep the solver fast we use a
smaller path count (default 8,192) with the geometric control variate;
this is enough to get σ to ~10 bps accuracy in seconds. The user can
crank `n_paths` up for tighter tolerance.

For geometric Asians we use the Kemna-Vorst closed form, which makes
calibration effectively instantaneous.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional
import numpy as np
from scipy.optimize import brentq

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.analytic import kemna_vorst_price
from engine.mc import price_mc_cv


@dataclass
class CalibrationResult:
    """Output of an IV calibration call."""

    implied_vol: float
    market_price: float
    fitted_price: float
    n_iterations: int
    converged: bool
    runtime_ms: float
    method: str = ""

    @property
    def fit_error(self) -> float:
        return abs(self.fitted_price - self.market_price)

    def __str__(self) -> str:
        status = "✓" if self.converged else "✗"
        return (
            f"{status} σ_impl = {self.implied_vol:.2%}  "
            f"(market {self.market_price:.4f}, fitted {self.fitted_price:.4f}, "
            f"|err| {self.fit_error:.4f}, {self.n_iterations} evals, "
            f"{self.runtime_ms:.0f} ms, {self.method})"
        )


def implied_volatility(
    contract: OptionContract,
    market_price: float,
    n_paths: int = 8_192,
    seed: Optional[int] = 42,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    xtol: float = 1e-4,
) -> CalibrationResult:
    """
    Solve for the implied volatility σ that reproduces `market_price`.

    Returns a CalibrationResult; `converged=False` if no σ inside
    [sigma_low, sigma_high] reproduces the price (e.g. arbitrage
    violation in the input).
    """
    t0 = time.perf_counter()
    eval_count = {"n": 0}

    # Choose the fastest pricer for the contract type
    if contract.averaging == Averaging.GEOMETRIC and contract.strike_type == StrikeType.FIXED:
        method = "Kemna-Vorst (closed form)"

        def price_at(sigma: float) -> float:
            eval_count["n"] += 1
            return kemna_vorst_price(contract.with_(sigma=sigma)).price
    else:
        method = f"MC + CV ({n_paths:,} paths)"

        def price_at(sigma: float) -> float:
            eval_count["n"] += 1
            return price_mc_cv(contract.with_(sigma=sigma), n_paths=n_paths, seed=seed).price

    def f(sigma: float) -> float:
        return price_at(sigma) - market_price

    # Check bracket
    f_lo, f_hi = f(sigma_low), f(sigma_high)
    if f_lo * f_hi > 0:
        # No sign change → no solution
        # Try to give a useful return: report the σ whose price is closest to market.
        sigma_best = sigma_low if abs(f_lo) < abs(f_hi) else sigma_high
        fitted = market_price + (f_lo if sigma_best == sigma_low else f_hi)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return CalibrationResult(
            implied_vol=sigma_best,
            market_price=market_price,
            fitted_price=fitted,
            n_iterations=eval_count["n"],
            converged=False,
            runtime_ms=runtime_ms,
            method=method,
        )

    sigma_impl = brentq(f, sigma_low, sigma_high, xtol=xtol, maxiter=80)
    fitted = price_at(sigma_impl)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return CalibrationResult(
        implied_vol=float(sigma_impl),
        market_price=market_price,
        fitted_price=fitted,
        n_iterations=eval_count["n"],
        converged=True,
        runtime_ms=runtime_ms,
        method=method,
    )
