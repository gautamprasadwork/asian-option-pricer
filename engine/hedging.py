"""
Delta-hedging simulation.

A pricing model is only as good as the hedge it implies. This module
simulates the *strategy* of writing the Asian option, collecting the
premium, and dynamically delta-hedging with the underlying at a chosen
rebalance frequency. The output is the **terminal P&L distribution after
hedging** — what the writer actually keeps after running the strategy
to maturity.

Why this matters for the project
--------------------------------
Showing "the price is €142.37" is a textbook exercise. Showing that
"writing this option and delta-hedging weekly produces a P&L distribution
with mean ≈ €0 and standard deviation ≈ €4" is what the instrument
actually *means* in operational terms. A wide post-hedge P&L distribution
is the cost of discrete rebalancing and the model's inability to hedge
gamma & vega — exactly the conversation a real risk manager has every
morning.

Algorithm
---------
For each of N sample paths:
  1. Sell the option at time 0 for the model price → cash account +P.
  2. At each rebalance date t_k, compute Δ_k by re-pricing the *remaining*
     Asian option conditional on the running average so far.
  3. Adjust the stock position to Δ_k shares; the cash account absorbs
     the trade and accrues interest at rate r.
  4. At maturity, settle the option payoff against the hedge book.
  5. Record the terminal P&L (cash + stock − option payoff).

Approximations and honest caveats
---------------------------------
- The exact Asian Delta at intermediate times is path-dependent — it
  depends on the *running* average up to t_k. We use the standard
  practitioner approximation: at time t_k after k of m observations, the
  residual contract is the (m−k)/m fraction of a *fresh Asian* on the
  remaining (m−k) dates with an adjusted strike

      K_eff = (K · m − Σ_{i≤k} S(t_i)) / (m − k).

  Its Delta is then approximated by a vanilla BS Delta on (S(t_k), K_eff,
  T−t_k), scaled by (m−k)/m. Crucially, the (m−k)/m factor is what makes
  the Asian Delta *decay to zero* as t → T (since the average is locked
  in), so the hedge size shrinks correctly near maturity. Without that
  factor the hedge over-trades dramatically and the post-hedge P&L
  variance is actually *worse* than no hedge.
- No transaction costs (a simple flat-bps model is provided as an
  optional knob).
- Continuous flat rate r, no funding spread.

These caveats are stated up-front in the UI's "Model limitations" panel.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple
import numpy as np
from scipy.stats import norm

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.results import PricingResult
from engine.paths import simulate_gbm_paths


@dataclass
class HedgingResult:
    """Per-path and aggregate output of the hedging simulation."""

    rebalance_dates: int
    n_paths: int
    pnl: np.ndarray              # shape (n_paths,) terminal P&L after hedging
    pnl_no_hedge: np.ndarray     # shape (n_paths,) terminal P&L of un-hedged short
    runtime_ms: float
    tc_bps: float = 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def std(self) -> float:
        return float(np.std(self.pnl, ddof=1))

    @property
    def std_no_hedge(self) -> float:
        return float(np.std(self.pnl_no_hedge, ddof=1))

    @property
    def var_95(self) -> float:
        return float(-np.quantile(self.pnl, 0.05))

    @property
    def es_95(self) -> float:
        cutoff = np.quantile(self.pnl, 0.05)
        return float(-np.mean(self.pnl[self.pnl <= cutoff]))

    @property
    def hedge_efficiency(self) -> float:
        """1 - σ_hedged / σ_unhedged, in [0, 1] (higher = better hedge)."""
        s_no = self.std_no_hedge
        return 1.0 - self.std / s_no if s_no > 0 else 0.0


def _residual_asian_delta_approx(
    S_t: float,
    K_strike: float,
    sum_observed: float,
    k_done: int,
    m_total: int,
    sigma: float,
    r: float,
    q: float,
    tau: float,
    is_call: bool,
) -> float:
    """
    Practitioner-grade approximation of the Asian Delta at intermediate time.

    The exact computation requires conditioning on the running average so
    far. We use the following decomposition (standard in textbook
    treatments of Asian-option hedging):

        payoff = max(A_T − K, 0)
               = (m − k) / m · max(A_remaining − K_eff, 0)

    where  K_eff = (K · m − Σ_{i ≤ k} S(t_i)) / (m − k)
    and    A_remaining = mean of S(t_{k+1}), …, S(t_m).

    The Delta of the residual ≈ vanilla BS Delta on (S_t, K_eff, τ=T−t_k),
    multiplied by the scaling factor (m − k) / m. The scaling factor is
    what drives the Asian Delta to zero as k → m (most of the average is
    already realised and no longer sensitive to S).

    Edge cases:
      - k == m: no future observations, Delta = 0.
      - K_eff ≤ 0: the residual payoff is *guaranteed* in the money for a
                   call (it's already past the strike on average); Delta
                   collapses to the (m-k)/m scaling × 1.
      - tau ≤ 0 : return intrinsic-style sign.
    """
    if k_done >= m_total:
        return 0.0
    scale = (m_total - k_done) / m_total
    K_eff = (K_strike * m_total - sum_observed) / (m_total - k_done)

    if K_eff <= 0:
        # Already certain to finish above strike (call) / below (put).
        return scale if is_call else -scale

    if tau <= 0 or sigma <= 0 or S_t <= 0:
        if is_call:
            return scale * (1.0 if S_t > K_eff else 0.0)
        return -scale * (1.0 if S_t < K_eff else 0.0)

    d1 = (math.log(S_t / K_eff) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    if is_call:
        return scale * math.exp(-q * tau) * norm.cdf(d1)
    return -scale * math.exp(-q * tau) * norm.cdf(-d1)


def simulate_delta_hedge(
    contract: OptionContract,
    model_price: float,
    rebalance_every: int = 5,
    n_paths: int = 5_000,
    seed: Optional[int] = 7,
    tc_bps: float = 0.0,
) -> HedgingResult:
    """
    Simulate dynamic delta-hedging of a short Asian option position.

    Parameters
    ----------
    contract : the option being written.
    model_price : the premium received at t=0 (use the engine's price).
    rebalance_every : rebalance every k monitoring steps (1 = each step,
                      contract.m = no rebalance / static hedge).
    n_paths : number of paths to simulate.
    seed : RNG seed for reproducibility.
    tc_bps : round-trip transaction cost in basis points of notional traded.

    Returns
    -------
    HedgingResult with the per-path terminal P&L (after hedging) and the
    per-path P&L of the un-hedged short for comparison.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    S_paths = simulate_gbm_paths(
        contract.S0, contract.r, contract.sigma, contract.T, contract.m,
        n_paths=n_paths, q=contract.q, antithetic=False, rng=rng,
    )
    # Prepend S₀ for convenience: shape (n_paths, m+1)
    S_full = np.hstack([np.full((n_paths, 1), contract.S0), S_paths])

    dt = contract.T / contract.m
    r = contract.r
    q = contract.q

    cash = np.full(n_paths, model_price, dtype=np.float64)     # premium received
    shares = np.zeros(n_paths, dtype=np.float64)
    tc_total = np.zeros(n_paths, dtype=np.float64)
    # Running sum of *observed* monitoring values per path (excludes S(t_0))
    running_sum = np.zeros(n_paths, dtype=np.float64)

    # Walk forward through k = 0, 1, …, m.
    # k=0 is t_0 (S_0, before any monitoring), k=m is t_m (final monitoring date).
    # The monitoring value S(t_k) for k>=1 is added to running_sum *after* any
    # hedge trade for that date is recorded (the trade decision is made on the
    # information set just *before* the new monitoring value is fixed).
    for k in range(contract.m + 1):
        t_k = k * dt
        tau = contract.T - t_k                      # remaining time
        S_k = S_full[:, k]

        # Accrue cash at risk-free rate
        if k > 0:
            cash *= math.exp(r * dt)

        # Decide whether to rebalance at this step
        is_rebal_date = (k % rebalance_every == 0) and (k < contract.m)

        if is_rebal_date and tau > 0:
            # Approximate Δ of the residual contract using the
            # running-sum-aware K_eff and the (m−k)/m scaling.
            # Note: k_done = number of monitoring dates already passed = k
            # (since at time k=0 no observations have been made, at k=1 one
            # has, etc.).
            new_shares = np.array([
                _residual_asian_delta_approx(
                    S_t=S_k[i],
                    K_strike=contract.K,
                    sum_observed=running_sum[i],
                    k_done=k,
                    m_total=contract.m,
                    sigma=contract.sigma,
                    r=r, q=q, tau=tau,
                    is_call=contract.is_call,
                )
                for i in range(n_paths)
            ])
            trade = new_shares - shares
            cash -= trade * S_k
            if tc_bps > 0:
                cost = tc_bps * 1e-4 * np.abs(trade) * S_k
                cash -= cost
                tc_total += cost
            shares = new_shares

        # Record the monitoring value for k>=1 (S(t_k) is the value just observed)
        if k >= 1:
            running_sum += S_k

    # ---- Terminal settlement ----
    # Realised Asian average over the monitoring dates t_1..t_m (not t_0)
    A = S_full[:, 1:].mean(axis=1)
    if contract.strike_type == StrikeType.FIXED:
        if contract.is_call:
            payoff = np.maximum(A - contract.K, 0.0)
        else:
            payoff = np.maximum(contract.K - A, 0.0)
    else:
        S_T = S_full[:, -1]
        if contract.is_call:
            payoff = np.maximum(S_T - A, 0.0)
        else:
            payoff = np.maximum(A - S_T, 0.0)

    # Liquidate the share position at S_T
    S_T = S_full[:, -1]
    cash += shares * S_T
    if tc_bps > 0:
        cost = tc_bps * 1e-4 * np.abs(shares) * S_T
        cash -= cost
        tc_total += cost

    pnl = cash - payoff
    # Un-hedged P&L: just premium grown at r, minus payoff
    pnl_no_hedge = model_price * math.exp(r * contract.T) - payoff

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return HedgingResult(
        rebalance_dates=contract.m // rebalance_every,
        n_paths=n_paths,
        pnl=pnl,
        pnl_no_hedge=pnl_no_hedge,
        runtime_ms=runtime_ms,
        tc_bps=tc_bps,
    )
