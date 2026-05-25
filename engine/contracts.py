"""
Contract specifications for Asian options.

Supports the full standard family:
- Arithmetic vs Geometric averaging
- Fixed vs Floating strike
- Call vs Put
- Discrete monitoring at m equally-spaced dates over [0, T]
- Continuous dividend yield q
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Averaging(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class StrikeType(str, Enum):
    FIXED = "fixed"
    FLOATING = "floating"


@dataclass(frozen=True)
class OptionContract:
    """
    Specification of an Asian option contract under GBM dynamics.

    Parameters
    ----------
    S0 : float
        Spot price of the underlying at t=0.
    K : float
        Strike price. For floating-strike Asians, K is ignored.
    r : float
        Continuously-compounded risk-free rate (e.g. 0.03 for 3%).
    sigma : float
        Annualized volatility (e.g. 0.20 for 20%).
    T : float
        Time to maturity in years.
    option_type : OptionType
        CALL or PUT.
    averaging : Averaging
        ARITHMETIC (no closed form, must be priced numerically) or
        GEOMETRIC (has closed form via Kemna-Vorst).
    strike_type : StrikeType
        FIXED ('max(A - K, 0)' for call) or
        FLOATING ('max(S_T - A, 0)' for call).
    m : int
        Number of equally-spaced monitoring dates over (0, T].
        Common conventions: m=12 (monthly), m=252 (daily).
    q : float
        Continuous dividend yield (default 0).
    """

    S0: float
    K: float
    r: float
    sigma: float
    T: float
    option_type: OptionType = OptionType.CALL
    averaging: Averaging = Averaging.ARITHMETIC
    strike_type: StrikeType = StrikeType.FIXED
    m: int = 252
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.S0 <= 0:
            raise ValueError("Spot price S0 must be positive")
        if self.K <= 0 and self.strike_type == StrikeType.FIXED:
            raise ValueError("Strike K must be positive for fixed-strike options")
        if self.sigma <= 0:
            raise ValueError("Volatility sigma must be positive")
        if self.T <= 0:
            raise ValueError("Maturity T must be positive")
        if self.m < 1:
            raise ValueError("Number of monitoring dates m must be >= 1")
        # Normalise enum values (allow string inputs)
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        object.__setattr__(self, "averaging", Averaging(self.averaging))
        object.__setattr__(self, "strike_type", StrikeType(self.strike_type))

    @property
    def dt(self) -> float:
        """Time step between monitoring dates."""
        return self.T / self.m

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def with_(self, **kwargs) -> "OptionContract":
        """Return a copy with selected fields replaced (immutable update)."""
        from dataclasses import replace
        return replace(self, **kwargs)

    def describe(self) -> str:
        """Short human-readable description for UI / logs."""
        return (
            f"{self.strike_type.value.title()}-strike "
            f"{self.averaging.value} Asian "
            f"{self.option_type.value} "
            f"(S0={self.S0:g}, K={self.K:g}, T={self.T:g}y, "
            f"σ={self.sigma:.1%}, r={self.r:.2%}, m={self.m})"
        )
