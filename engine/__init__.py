"""
Asian Option Pricing Engine
============================

A research-grade pricing engine for arithmetic and geometric Asian options
under Black-Scholes (GBM) dynamics with discrete monitoring.

Primary pricing method: Monte Carlo simulation with antithetic variates
and a geometric-Asian control variate (Kemna-Vorst, 1990).

Validation benchmarks: Kemna-Vorst closed-form (geometric Asian) and
Cox-Ross-Rubinstein binomial tree (vanilla European).

Public API:
    from engine import OptionContract, price, greeks, sensitivities
"""

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.results import PricingResult, GreeksResult, GreekValue
from engine.analytic import (
    kemna_vorst_price,
    kemna_vorst_expected_payoff_discounted,
    black_scholes_price,
)
from engine.mc import (
    price_mc, price_mc_antithetic, price_mc_cv, price_qmc,
)
from engine.trees import price_crr_tree
from engine.greeks import compute_greeks, pathwise_delta
from engine.paths import (
    simulate_gbm_paths, simulate_gbm_paths_sobol,
    arithmetic_average, geometric_average,
)
from engine.presets import PRESETS, PRESET_ORDER, default_preset, get_preset, Preset
from engine.diagnostics import run_diagnostics, Diagnostic, Severity
from engine.hedging import simulate_delta_hedge, HedgingResult
from engine.calibration import implied_volatility, CalibrationResult
from engine.sensitivities import (
    sweep_1d, surface_2d, tornado_ranking, scenario_table,
    pnl_attribution, Scenario, ScenarioPnL,
)

__all__ = [
    # Contracts & results
    "OptionContract", "OptionType", "Averaging", "StrikeType",
    "PricingResult", "GreeksResult", "GreekValue",
    # Analytic
    "kemna_vorst_price", "kemna_vorst_expected_payoff_discounted",
    "black_scholes_price",
    # MC
    "price_mc", "price_mc_antithetic", "price_mc_cv", "price_qmc",
    # Trees
    "price_crr_tree",
    # Greeks
    "compute_greeks", "pathwise_delta",
    # Paths
    "simulate_gbm_paths", "simulate_gbm_paths_sobol",
    "arithmetic_average", "geometric_average",
    # Presets / diagnostics / hedging / calibration / sensitivities
    "PRESETS", "PRESET_ORDER", "default_preset", "get_preset", "Preset",
    "run_diagnostics", "Diagnostic", "Severity",
    "simulate_delta_hedge", "HedgingResult",
    "implied_volatility", "CalibrationResult",
    "sweep_1d", "surface_2d", "tornado_ranking", "scenario_table",
    "pnl_attribution", "Scenario", "ScenarioPnL",
    # Convenience
    "price_auto",
]

__version__ = "1.0.0"


def price_auto(contract: OptionContract, n_paths: int = 50_000, seed: int = 42) -> PricingResult:
    """
    Convenience: pick the best pricer for the given contract automatically.

    - Geometric + fixed strike  → Kemna-Vorst closed form (exact).
    - Arithmetic + fixed strike → MC + Antithetic + Geometric CV (primary).
    - Anything floating-strike  → MC + Antithetic (no CV available).
    """
    if contract.averaging == Averaging.GEOMETRIC and contract.strike_type == StrikeType.FIXED:
        return kemna_vorst_price(contract)
    if contract.averaging == Averaging.ARITHMETIC and contract.strike_type == StrikeType.FIXED:
        return price_mc_cv(contract, n_paths=n_paths, seed=seed)
    return price_mc_antithetic(contract, n_paths=n_paths, seed=seed)
