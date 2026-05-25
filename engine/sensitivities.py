"""
Sensitivity analysis.

This module produces the inputs for every sensitivity visual in the UI:

  - `sweep_1d`         : 1D price curve vs one parameter (spot, vol, strike, T, r).
  - `surface_2d`       : 2D price surface for a heat-map (e.g. S × σ).
  - `tornado_ranking`  : ranked impact of ±ε bumps across all parameters.
  - `scenario_table`   : classical stress matrix (±10% S, ±5pp σ, +30d, ±50bp r).
  - `pnl_attribution`  : Taylor decomposition of a scenario shock into
                         Δ·ΔS + ½Γ·ΔS² + ν·Δσ + Θ·Δt + ρ·Δr + residual.

Everything is engine-agnostic — it just calls a `price_fn(contract) -> float`
so the user can plug in MC, MC+CV, QMC, or KV depending on the use case.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import math
import numpy as np

from engine.contracts import OptionContract, Averaging, StrikeType
from engine.results import PricingResult, GreeksResult
from engine.analytic import kemna_vorst_price
from engine.mc import price_mc_cv, price_mc_antithetic


# --------------------------------------------------------------------------- #
#  Default price function: pick the right pricer for the contract
# --------------------------------------------------------------------------- #
def _default_price_fn(n_paths: int = 16_000, seed: int = 42) -> Callable[[OptionContract], float]:
    """Return a fast, contract-aware pricing closure."""
    def fn(c: OptionContract) -> float:
        if c.averaging == Averaging.GEOMETRIC and c.strike_type == StrikeType.FIXED:
            return kemna_vorst_price(c).price
        if c.averaging == Averaging.ARITHMETIC and c.strike_type == StrikeType.FIXED:
            return price_mc_cv(c, n_paths=n_paths, seed=seed).price
        return price_mc_antithetic(c, n_paths=n_paths, seed=seed).price
    return fn


# --------------------------------------------------------------------------- #
#  1D sweep
# --------------------------------------------------------------------------- #
_PARAM_LABELS = {
    "S0":    ("Spot price S₀",     "{:,.2f}"),
    "K":     ("Strike price K",    "{:,.2f}"),
    "sigma": ("Volatility σ",      "{:.1%}"),
    "T":     ("Maturity T (years)", "{:.2f}"),
    "r":     ("Risk-free rate r",  "{:.2%}"),
    "q":     ("Dividend yield q",  "{:.2%}"),
}


def sweep_1d(
    contract: OptionContract,
    param: str,
    values: Sequence[float],
    price_fn: Optional[Callable[[OptionContract], float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Sweep `param` across `values`, returning (values, prices).

    Example
    -------
    >>> S_range = np.linspace(60, 100, 30)
    >>> xs, ys = sweep_1d(contract, "S0", S_range)
    """
    if param not in _PARAM_LABELS:
        raise ValueError(f"Unknown parameter '{param}'. Choose from {list(_PARAM_LABELS)}")
    pricer = price_fn or _default_price_fn()
    prices: List[float] = []
    for v in values:
        c = contract.with_(**{param: float(v)})
        prices.append(pricer(c))
    return list(values), prices


# --------------------------------------------------------------------------- #
#  2D surface
# --------------------------------------------------------------------------- #
def surface_2d(
    contract: OptionContract,
    param_x: str,
    values_x: Sequence[float],
    param_y: str,
    values_y: Sequence[float],
    price_fn: Optional[Callable[[OptionContract], float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D price surface for a heat-map.

    Returns
    -------
    X, Y, Z : 2D ndarrays of shape (len(values_y), len(values_x)).
              Z[i, j] = price at param_x=values_x[j], param_y=values_y[i].
    """
    pricer = price_fn or _default_price_fn()
    Z = np.zeros((len(values_y), len(values_x)))
    for i, vy in enumerate(values_y):
        for j, vx in enumerate(values_x):
            c = contract.with_(**{param_x: float(vx), param_y: float(vy)})
            Z[i, j] = pricer(c)
    X, Y = np.meshgrid(np.array(values_x), np.array(values_y))
    return X, Y, Z


# --------------------------------------------------------------------------- #
#  Tornado: ranked single-parameter sensitivities
# --------------------------------------------------------------------------- #
@dataclass
class TornadoBar:
    param: str
    label: str
    price_down: float
    price_up: float
    shock_down_str: str
    shock_up_str: str

    @property
    def impact(self) -> float:
        """Symmetric absolute price impact at this bump size."""
        return 0.5 * (abs(self.price_up - self._base) + abs(self.price_down - self._base))

    _base: float = 0.0


def tornado_ranking(
    contract: OptionContract,
    base_price: float,
    price_fn: Optional[Callable[[OptionContract], float]] = None,
    shocks: Optional[dict] = None,
) -> List[TornadoBar]:
    """
    For each parameter, compute the price at +shock and −shock and return
    the bars sorted by absolute impact.

    Default shocks (chosen to be roughly comparable in *practical* terms):
        S0    : ±10% of S₀
        K     : ±10% of K (if fixed strike)
        sigma : ±5 percentage points
        T     : ±30 calendar days
        r     : ±50 basis points
    """
    pricer = price_fn or _default_price_fn()
    if shocks is None:
        shocks = {
            "S0":    (0.10, lambda c, s: c.S0 * (1 + s), lambda s: f"{s:+.0%}"),
            "K":     (0.10, lambda c, s: c.K * (1 + s),  lambda s: f"{s:+.0%}"),
            "sigma": (0.05, lambda c, s: c.sigma + s,     lambda s: f"{s:+.0%} pp"),
            "T":     (30/365, lambda c, s: max(c.T + s, 1e-3), lambda s: f"{int(s*365):+d}d"),
            "r":     (0.005, lambda c, s: c.r + s,         lambda s: f"{s:+.1%}"),
        }

    bars: List[TornadoBar] = []
    for param, (shock, apply_fn, fmt_fn) in shocks.items():
        if param == "K" and contract.strike_type != StrikeType.FIXED:
            continue
        up_val   = apply_fn(contract, +shock)
        down_val = apply_fn(contract, -shock)
        c_up   = contract.with_(**{param: up_val})
        c_down = contract.with_(**{param: down_val})
        bars.append(TornadoBar(
            param=param,
            label=_PARAM_LABELS[param][0],
            price_down=pricer(c_down),
            price_up=pricer(c_up),
            shock_down_str=fmt_fn(-shock),
            shock_up_str=fmt_fn(+shock),
            _base=base_price,
        ))

    bars.sort(key=lambda b: b.impact, reverse=True)
    return bars


# --------------------------------------------------------------------------- #
#  Scenario stress table
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    label: str
    overrides: dict   # e.g. {"S0": 88.0} or {"sigma": 0.37}


@dataclass
class ScenarioPnL:
    scenario: Scenario
    new_price: float
    base_price: float

    @property
    def delta_pnl(self) -> float:
        return self.new_price - self.base_price

    @property
    def pct_change(self) -> float:
        return (self.new_price / self.base_price - 1.0) if self.base_price else 0.0


def scenario_table(
    contract: OptionContract,
    base_price: float,
    scenarios: Optional[List[Scenario]] = None,
    price_fn: Optional[Callable[[OptionContract], float]] = None,
) -> List[ScenarioPnL]:
    """
    Apply a list of named scenarios to `contract` and report the price
    change for each.

    Default scenario set (the classical "risk ladder"):
        Spot +10%, Spot -10%
        Vol  +5pp, Vol  -5pp
        T -30 days
        r +50bp, r -50bp
    """
    pricer = price_fn or _default_price_fn()
    if scenarios is None:
        S = contract.S0
        sig = contract.sigma
        r0 = contract.r
        scenarios = [
            Scenario("Spot +10%",  {"S0": S * 1.10}),
            Scenario("Spot −10%",  {"S0": S * 0.90}),
            Scenario("Vol +5 pp",  {"sigma": sig + 0.05}),
            Scenario("Vol −5 pp",  {"sigma": max(sig - 0.05, 1e-3)}),
            Scenario("T −30 days", {"T": max(contract.T - 30/365, 1e-3)}),
            Scenario("r +50 bp",   {"r": r0 + 0.005}),
            Scenario("r −50 bp",   {"r": r0 - 0.005}),
        ]
    out: List[ScenarioPnL] = []
    for s in scenarios:
        c = contract.with_(**s.overrides)
        out.append(ScenarioPnL(s, pricer(c), base_price))
    return out


# --------------------------------------------------------------------------- #
#  P&L attribution (Taylor decomposition of a scenario shock)
# --------------------------------------------------------------------------- #
@dataclass
class PnLComponent:
    name: str
    value: float

    def __str__(self) -> str:
        return f"{self.name}: {self.value:+.4f}"


def pnl_attribution(
    base_price: float,
    new_price: float,
    greeks: GreeksResult,
    shock_S: float = 0.0,
    shock_sigma: float = 0.0,
    shock_T_days: float = 0.0,
    shock_r: float = 0.0,
) -> List[PnLComponent]:
    """
    Decompose (new_price - base_price) into the standard Greek contributions:

        ΔPnL ≈ Δ·ΔS + ½·Γ·(ΔS)² + ν·Δσ_pp + Θ·ΔT_days + ρ·Δr_pp + residual

    where ν, Θ, ρ are reported in the per-1pp / per-day conventions.

    Returns a list of components ending with the residual.
    """
    delta_pnl    = greeks.delta.value * shock_S
    gamma_pnl    = 0.5 * greeks.gamma.value * shock_S ** 2
    vega_pnl     = greeks.vega.value * (shock_sigma * 100)        # Vega is per 1 pp
    theta_pnl    = greeks.theta.value * shock_T_days              # Θ is per day
    rho_pnl      = greeks.rho.value * (shock_r * 100)             # ρ is per 1 pp
    total_explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
    actual = new_price - base_price
    residual = actual - total_explained
    return [
        PnLComponent("Δ · ΔS",          delta_pnl),
        PnLComponent("½ · Γ · ΔS²",      gamma_pnl),
        PnLComponent("ν · Δσ",          vega_pnl),
        PnLComponent("Θ · ΔT",          theta_pnl),
        PnLComponent("ρ · Δr",          rho_pnl),
        PnLComponent("Residual (higher-order)", residual),
    ]
