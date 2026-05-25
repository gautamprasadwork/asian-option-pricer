"""
Lattice methods.

We provide a **Cox-Ross-Rubinstein (CRR) binomial tree** for vanilla
European and American options. The tree is used in this project for two
purposes:

  1. **Independent validation** of the Monte Carlo engine: in the
     vanilla-European limit, CRR converges to Black-Scholes as the number
     of steps grows, so a passing test against BS proves both the
     risk-neutral drift and the discounting in our codebase are correct.
  2. To round out the "numerical methods" story alongside Monte Carlo —
     the project specification explicitly cites "Monte Carlo simulations
     or pricing on trees".

We do **not** price Asian options on a tree in this module. The standard
algorithm (Hull-White, 1993, Forward-Shooting Grid) augments each tree
node with K representative running averages and interpolates between
them on each backward step. It is a textbook reference algorithm but
its O(K · N²) cost and the discretization-of-the-average bias make
Monte Carlo with the geometric control variate (this project's primary
pricer) both faster and more accurate at the same compute budget. The
tree is therefore reserved for the vanilla benchmark.
"""

from __future__ import annotations
import math
import time
from typing import Literal
import numpy as np

from engine.results import PricingResult


def price_crr_tree(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int = 1000,
    option_type: Literal["call", "put"] = "call",
    exercise: Literal["european", "american"] = "european",
    q: float = 0.0,
) -> PricingResult:
    """
    Cox-Ross-Rubinstein binomial tree for a **vanilla** option.

    Parameters
    ----------
    S0, K, r, sigma, T, q : standard GBM inputs.
    steps : number of time steps in the tree. CRR convergence is O(1/steps).
    option_type : "call" or "put".
    exercise : "european" (terminal payoff only) or "american" (early
               exercise allowed at every node).

    Convergence note
    ----------------
    The CRR tree converges to Black-Scholes from below in an oscillatory
    fashion (the well-known "saw-tooth" pattern). For the validation test
    in this project we use ≥ 1000 steps so the saw-tooth amplitude is
    well below 1 basis point.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    t0 = time.perf_counter()
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside (0, 1). "
            "Try more steps or check parameters."
        )
    disc = math.exp(-r * dt)

    # Vectorised terminal payoff
    j = np.arange(steps + 1)
    S_T = S0 * (u ** j) * (d ** (steps - j))
    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        if exercise == "american":
            S_i = S0 * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            if option_type == "call":
                intrinsic = np.maximum(S_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_i, 0.0)
            values = np.maximum(values, intrinsic)

    price = float(values[0])
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return PricingResult(
        price=price, std_error=0.0,
        method=f"CRR Binomial Tree ({exercise}, {steps} steps)",
        n_paths=steps, runtime_ms=runtime_ms,
        extra={"u": u, "d": d, "p": p},
    )
