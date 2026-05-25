"""
Reality-check / contextual warnings.

A pricing number on its own can mislead. A €0.02 price isn't "wrong" —
it just means the option is deep out-of-the-money and the user should
know that explicitly, not figure it out from the digit count. A vol
input of 120% might be a typo or might be a crisis regime; either way,
the user should be flagged.

`run_diagnostics(contract, result, greeks=None)` returns a list of
`Diagnostic` records — short, contextual messages with a severity and
an actionable next step. The UI renders them as colored alert strips.

Each check is small, named, and independently testable.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType
from engine.results import PricingResult, GreeksResult


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    title: str
    message: str
    suggestion: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.title}: {self.message}"


# --------------------------------------------------------------------------- #
#  Individual checks
# --------------------------------------------------------------------------- #
def _check_essentially_worthless(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """Price below 0.1% of spot → effectively zero."""
    if result.price < 0.001 * contract.S0:
        return Diagnostic(
            Severity.WARN,
            "Option is essentially worthless",
            f"Price of {result.price:.4f} is less than 0.1% of spot ({contract.S0:.2f}). "
            "This contract is so far out-of-the-money that the model treats "
            "exercise as nearly impossible.",
            "Check the strike — for a call, K shouldn't be much above S₀; "
            "for a put, K shouldn't be much below S₀.",
        )
    return None


def _check_deep_itm(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """Price ≈ intrinsic value (within 1%) → all time value gone, option behaves like the underlying."""
    if contract.strike_type != StrikeType.FIXED:
        return None  # intrinsic ill-defined for floating strike
    # Approximate intrinsic at maturity by spot vs strike (rough heuristic)
    if contract.is_call:
        intrinsic = max(contract.S0 - contract.K, 0)
    else:
        intrinsic = max(contract.K - contract.S0, 0)
    if intrinsic > 0 and result.price < 1.05 * intrinsic and result.price < contract.S0 * 0.5:
        return Diagnostic(
            Severity.INFO,
            "Deep in-the-money — almost no time value",
            f"Price {result.price:.2f} is close to immediate-exercise value "
            f"({intrinsic:.2f}). The option behaves much like the underlying.",
            "If the goal is leverage or volatility exposure, an at-the-money "
            "strike will have more time value per dollar of premium.",
        )
    return None


def _check_noisy_estimate(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """SE > 5% of price → re-run with more paths."""
    if result.std_error == 0.0:
        return None  # closed-form / tree, no SE
    if result.price > 0.01 and result.std_error / result.price > 0.05:
        return Diagnostic(
            Severity.WARN,
            "Monte Carlo estimate is noisy",
            f"Standard error ({result.std_error:.4f}) is "
            f"{100 * result.std_error / result.price:.1f}% of the price — wide for a decision input.",
            f"Increase paths from {result.n_paths:,} to "
            f"{min(result.n_paths * 4, 500_000):,} to roughly halve the SE.",
        )
    return None


def _check_extreme_vol(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """Volatility outside typical regime."""
    if contract.sigma > 1.0:
        return Diagnostic(
            Severity.WARN,
            "Volatility is extreme (>100%)",
            f"σ = {contract.sigma:.1%}. This is a crisis-regime level — "
            "typical equity index implied vols sit in the 10–40% range, "
            "commodities 20–60%.",
            "Double-check the input. If intentional (e.g. modelling a crisis "
            "scenario), state that in the report.",
        )
    if contract.sigma < 0.03:
        return Diagnostic(
            Severity.WARN,
            "Volatility is unusually low (<3%)",
            f"σ = {contract.sigma:.1%}. Most liquid options have vols above 5%; "
            "this may be a typo or stale historical estimate.",
            "Use a recent implied vol from market quotes if possible.",
        )
    return None


def _check_near_expiry(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """T < 1 week → model approximations break down."""
    if contract.T < 7.0 / 365.0:
        return Diagnostic(
            Severity.INFO,
            "Close to expiry",
            f"Maturity = {contract.T * 365:.1f} days. With so few monitoring dates "
            "left, the average is dominated by realised history (which our model "
            "ignores) rather than future randomness.",
            "For live trades inside the last week, use the *running* average so far "
            "and re-price the residual.",
        )
    return None


def _check_arbitrage_bounds(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """Check classical no-arbitrage bounds for fixed-strike Asians."""
    if contract.strike_type != StrikeType.FIXED:
        return None
    disc_K = math.exp(-contract.r * contract.T) * contract.K
    # Upper bound: call price ≤ S₀; put price ≤ K·e^(−rT)
    if contract.is_call and result.price > contract.S0 * 1.001:
        return Diagnostic(
            Severity.ERROR,
            "Arbitrage bound violated",
            f"Call price {result.price:.2f} exceeds spot {contract.S0:.2f}. "
            "An Asian call cannot be worth more than the underlying.",
            "This usually means the simulation is broken or the parameters are "
            "inconsistent. Re-check inputs.",
        )
    if (not contract.is_call) and result.price > disc_K * 1.001:
        return Diagnostic(
            Severity.ERROR,
            "Arbitrage bound violated",
            f"Put price {result.price:.2f} exceeds discounted strike {disc_K:.2f}. "
            "An Asian put cannot be worth more than the present value of K.",
            "This usually means the simulation is broken or the parameters are "
            "inconsistent. Re-check inputs.",
        )
    if result.price < -1e-6:
        return Diagnostic(
            Severity.ERROR,
            "Negative price",
            f"Price came out as {result.price:.4f}. Prices must be non-negative.",
            "Almost certainly a Monte Carlo control-variate sign error or a "
            "broken estimator. File a bug.",
        )
    return None


def _check_extreme_moneyness(
    contract: OptionContract, result: PricingResult
) -> Optional[Diagnostic]:
    """|ln(S/K)|/(σ√T) > 3 → far from at-the-money, CV efficiency degrades."""
    if contract.strike_type != StrikeType.FIXED:
        return None
    if contract.K <= 0:
        return None
    z = abs(math.log(contract.S0 / contract.K)) / (contract.sigma * math.sqrt(contract.T) + 1e-12)
    if z > 3.0:
        return Diagnostic(
            Severity.INFO,
            "Strike is far from at-the-money",
            f"S₀ and K differ by {z:.1f}σ-equivalent units. Most paths land "
            "on the same side of the strike, so the geometric control "
            "variate is less effective here.",
            "If the SE looks large, try more paths (which scales worse for "
            "deep OTM) or accept the wider CI.",
        )
    return None


def _check_greeks_consistency(
    contract: OptionContract,
    result: PricingResult,
    greeks: GreeksResult,
) -> Optional[Diagnostic]:
    """FD-Delta vs pathwise-Delta agreement."""
    if greeks.pathwise_delta is None:
        return None
    diff = abs(greeks.delta.value - greeks.pathwise_delta.value)
    tol = 1.96 * math.sqrt(greeks.delta.std_error ** 2 + greeks.pathwise_delta.std_error ** 2)
    # Add a small absolute floor to handle the case where both SEs are tiny
    tol = max(tol, 0.01)
    if diff > tol:
        return Diagnostic(
            Severity.WARN,
            "Delta estimators disagree",
            f"FD-Delta = {greeks.delta.value:.4f}, pathwise-Delta = "
            f"{greeks.pathwise_delta.value:.4f}. The gap "
            f"({diff:.4f}) exceeds their joint 95% confidence band ({tol:.4f}).",
            "Re-run with more paths. If the disagreement persists, the "
            "engine has a bug — investigate.",
        )
    return None


def _check_call_delta_range(
    contract: OptionContract,
    result: PricingResult,
    greeks: GreeksResult,
) -> Optional[Diagnostic]:
    """Δ_call ∈ [0, 1], Δ_put ∈ [-1, 0]."""
    d = greeks.delta.value
    tol = 0.05
    if contract.strike_type != StrikeType.FIXED:
        return None  # different bounds for floating-strike
    if contract.is_call and not (-tol <= d <= 1 + tol):
        return Diagnostic(
            Severity.ERROR,
            "Delta out of bounds for a call",
            f"Δ = {d:.4f} but call Delta should sit in [0, 1].",
            "Check the FD bump direction and the payoff sign convention.",
        )
    if (not contract.is_call) and not (-1 - tol <= d <= tol):
        return Diagnostic(
            Severity.ERROR,
            "Delta out of bounds for a put",
            f"Δ = {d:.4f} but put Delta should sit in [-1, 0].",
            "Check the FD bump direction and the payoff sign convention.",
        )
    return None


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #
_PRICE_LEVEL_CHECKS = [
    _check_essentially_worthless,
    _check_deep_itm,
    _check_noisy_estimate,
    _check_extreme_vol,
    _check_near_expiry,
    _check_arbitrage_bounds,
    _check_extreme_moneyness,
]

_GREEK_LEVEL_CHECKS = [
    _check_greeks_consistency,
    _check_call_delta_range,
]


def run_diagnostics(
    contract: OptionContract,
    result: PricingResult,
    greeks: Optional[GreeksResult] = None,
) -> List[Diagnostic]:
    """Return the list of all triggered diagnostics, in order of severity."""
    out: List[Diagnostic] = []
    for check in _PRICE_LEVEL_CHECKS:
        d = check(contract, result)
        if d is not None:
            out.append(d)
    if greeks is not None:
        for check in _GREEK_LEVEL_CHECKS:
            d = check(contract, result, greeks)
            if d is not None:
                out.append(d)
    # Sort by severity: ERROR > WARN > INFO
    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    out.sort(key=lambda d: order[d.severity])
    return out
