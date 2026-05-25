"""
Typed result containers returned by the pricing engine.

All pricers return a PricingResult so the UI / tests / report-generator
never have to do arithmetic — they just read fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import json
import math


@dataclass
class PricingResult:
    """
    Result of a single pricing call.

    Attributes
    ----------
    price : float
        Point estimate of the option price.
    std_error : float
        Standard error of the estimate. 0.0 for closed-form / tree methods.
    method : str
        Human-readable method label, e.g. "MC + Antithetic + Geometric CV".
    n_paths : int
        Number of Monte Carlo paths used. For trees, the number of time steps.
    runtime_ms : float
        Wall-clock runtime in milliseconds.
    extra : dict
        Optional method-specific diagnostics
        (e.g. {"beta": 0.974, "geom_cv_price": 138.42}).
    """

    price: float
    std_error: float
    method: str
    n_paths: int
    runtime_ms: float
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def ci_95(self) -> Tuple[float, float]:
        """Two-sided 95% confidence interval (price ± 1.96 · SE)."""
        return (
            self.price - 1.96 * self.std_error,
            self.price + 1.96 * self.std_error,
        )

    @property
    def variance(self) -> float:
        """Estimator variance (SE²)."""
        return self.std_error ** 2

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["ci_95"] = list(self.ci_95)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        lo, hi = self.ci_95
        return (
            f"{self.method}: {self.price:.4f}  "
            f"(SE={self.std_error:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], "
            f"N={self.n_paths:,}, {self.runtime_ms:.1f} ms)"
        )


@dataclass
class GreekValue:
    """A single Greek with its uncertainty."""

    value: float
    std_error: float = 0.0

    @property
    def ci_95(self) -> Tuple[float, float]:
        return (
            self.value - 1.96 * self.std_error,
            self.value + 1.96 * self.std_error,
        )

    def __str__(self) -> str:
        if self.std_error > 0:
            return f"{self.value:+.4f} ± {1.96 * self.std_error:.4f}"
        return f"{self.value:+.4f}"


@dataclass
class GreeksResult:
    """
    Full Greeks risk report.

    Each Greek is a GreekValue with a point estimate and standard error.
    pathwise_delta is provided as an independent cross-check on FD Delta.
    """

    delta: GreekValue
    gamma: GreekValue
    vega: GreekValue
    theta: GreekValue
    rho: GreekValue
    pathwise_delta: Optional[GreekValue] = None
    method: str = "Finite Difference (CRN)"
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict:
        out = {
            "delta": {"value": self.delta.value, "se": self.delta.std_error},
            "gamma": {"value": self.gamma.value, "se": self.gamma.std_error},
            "vega": {"value": self.vega.value, "se": self.vega.std_error},
            "theta": {"value": self.theta.value, "se": self.theta.std_error},
            "rho": {"value": self.rho.value, "se": self.rho.std_error},
            "method": self.method,
            "runtime_ms": self.runtime_ms,
        }
        if self.pathwise_delta is not None:
            out["pathwise_delta"] = {
                "value": self.pathwise_delta.value,
                "se": self.pathwise_delta.std_error,
            }
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        lines = [
            f"Δ Delta  : {self.delta}",
            f"Γ Gamma  : {self.gamma}",
            f"ν Vega   : {self.vega}    (per 1% vol)",
            f"Θ Theta  : {self.theta}    (per day)",
            f"ρ Rho    : {self.rho}    (per 1% rate)",
        ]
        if self.pathwise_delta is not None:
            lines.append(f"Δ_pathwise: {self.pathwise_delta}   (cross-check)")
        return "\n".join(lines)
