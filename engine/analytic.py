"""
Closed-form / analytical pricers.

This module implements two analytical results used throughout the project:

1. **Kemna-Vorst (1990) formula** for the discrete geometric Asian option.
   This is the foundation of our variance-reduction technique: the geometric
   Asian payoff has analytical expectation under GBM, so it serves as a
   high-correlation control variate for the (otherwise un-closed-form)
   arithmetic Asian payoff.

2. **Black-Scholes (1973) formula** for vanilla European options.
   Used to validate the CRR binomial tree and as a reference price.

Both formulas support continuous dividend yield q.
"""

from __future__ import annotations
import math
import time
from typing import Optional
import numpy as np
from scipy.stats import norm

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.results import PricingResult


# --------------------------------------------------------------------------- #
#  Black-Scholes (vanilla European)
# --------------------------------------------------------------------------- #
def black_scholes_price(
    S0: float, K: float, r: float, sigma: float, T: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton price for a European vanilla option.

    Used as a sanity-check benchmark for the CRR binomial tree and as
    a fallback control variate.

    Parameters
    ----------
    S0, K, r, sigma, T : standard BS inputs
    option_type : "call" or "put"
    q : continuous dividend yield (default 0)

    Returns
    -------
    float : option price
    """
    if T <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return intrinsic

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)


# --------------------------------------------------------------------------- #
#  Kemna-Vorst (discrete geometric Asian)
# --------------------------------------------------------------------------- #
def _kv_adjusted_params(sigma: float, r: float, q: float, T: float, m: int):
    """
    Return (sigma_g, nu, mu_kv) — the adjusted GBM parameters for the
    geometric average G under discrete monitoring at m equally-spaced dates.

    The geometric average G = (∏_{i=1..m} S(t_i))^{1/m} satisfies, under GBM,

        ln(G_T) ~ Normal( ln(S0) + nu * T ,  sigma_g^2 * T )

    where

        nu       = (r - q - sigma^2 / 2) * (m+1)/(2m)
        sigma_g^2 = sigma^2 * (m+1)(2m+1) / (6 m^2)

    For m → ∞ this reduces to the classical continuous-monitoring limit
    (nu → (r-q-σ²/2)/2 ,  sigma_g² → sigma²/3).
    """
    factor_drift = (m + 1) / (2 * m)
    factor_var   = (m + 1) * (2 * m + 1) / (6 * m * m)

    sigma_g_sq = sigma * sigma * factor_var
    sigma_g    = math.sqrt(sigma_g_sq)
    nu         = (r - q - 0.5 * sigma * sigma) * factor_drift
    mu_kv      = nu + 0.5 * sigma_g_sq  # so that E[G_T] = S0 * exp(mu_kv * T)
    return sigma_g, nu, mu_kv


def kemna_vorst_price(contract: OptionContract) -> PricingResult:
    """
    Closed-form price for a **discrete geometric Asian option** (fixed strike).

    Reference: Kemna A. & Vorst A. (1990), "A pricing method for options
    based on average asset values", Journal of Banking & Finance 14, 113–129.

    The trick: the geometric mean of correlated lognormals is itself
    lognormal, so the price has a Black-Scholes-style closed form with
    adjusted drift and volatility.

    Raises
    ------
    ValueError if the contract is not a fixed-strike geometric Asian.
    """
    if contract.averaging != Averaging.GEOMETRIC:
        raise ValueError(
            "Kemna-Vorst is the closed form for *geometric* Asians only. "
            f"Got averaging={contract.averaging.value}."
        )
    if contract.strike_type != StrikeType.FIXED:
        raise ValueError(
            "Kemna-Vorst closed form requires a fixed strike. "
            f"Got strike_type={contract.strike_type.value}."
        )

    t0 = time.perf_counter()

    S0, K, r, sigma, T, m, q = (
        contract.S0, contract.K, contract.r, contract.sigma,
        contract.T, contract.m, contract.q,
    )

    sigma_g, nu, mu_kv = _kv_adjusted_params(sigma, r, q, T, m)
    sg_sqrtT = sigma_g * math.sqrt(T)

    # Forward-of-geometric-average (E[G_T] under the risk-neutral measure)
    forward_G = S0 * math.exp(mu_kv * T)

    d1 = (math.log(S0 / K) + (mu_kv + 0.5 * sigma_g * sigma_g) * T) / sg_sqrtT
    d2 = d1 - sg_sqrtT

    disc = math.exp(-r * T)
    if contract.is_call:
        price = disc * (forward_G * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = disc * (K * norm.cdf(-d2) - forward_G * norm.cdf(-d1))

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return PricingResult(
        price=price,
        std_error=0.0,
        method="Kemna-Vorst (closed form)",
        n_paths=0,
        runtime_ms=runtime_ms,
        extra={
            "sigma_g": sigma_g,
            "mu_kv": mu_kv,
            "forward_G": forward_G,
        },
    )


def kemna_vorst_expected_payoff_discounted(contract: OptionContract) -> float:
    """
    Convenience: return the *discounted expected payoff* of the geometric
    Asian, i.e. just the price as a float.

    This is used as the known expectation of the control variate inside
    the MC engine. The geometric variant of `contract` is constructed
    automatically.
    """
    geom_contract = contract.with_(averaging=Averaging.GEOMETRIC)
    return kemna_vorst_price(geom_contract).price
