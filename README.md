# Asian Option Pricing Engine

[![CI](https://github.com/gautamprasadwork/asian-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/gautamprasadwork/asian-option-pricer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-1E3A5F.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade pricing engine for **arithmetic Asian options** under
Black–Scholes (GBM) dynamics with discrete monitoring, priced by Monte
Carlo simulation with a **geometric-Asian control variate** and validated
against a Cox-Ross-Rubinstein binomial tree.

Built as a final project for *Advanced Corporate Finance* at the Poznań
University of Economics and Business (UEP).

---

## What does it do?

An Asian option's payoff depends on the **average** of the underlying
price over a monitoring window, not its terminal value. That makes it the
standard hedge for any natural exposure to an average price — an Indian
refiner buying crude every day, a European exporter receiving USD every
week, a Polish fund tracking an index. The engine prices these
instruments, computes their Greeks, simulates a dynamic delta-hedging
strategy, calibrates implied volatility from a quoted market price, and
runs full sensitivity & stress analysis — all through an institutional
Streamlit dashboard.

```bash
git clone https://github.com/gautamprasadwork/asian-option-pricer.git
cd asian-option-pricer
pip install -r requirements.txt
streamlit run app.py
```

---

## Methods

| Layer | Method | Purpose |
|-------|--------|---------|
| Closed-form | **Kemna–Vorst (1990)** for the discrete geometric Asian | Analytical benchmark + control-variate mean |
| Primary numerical | **Monte Carlo** with antithetic variates and a **geometric-Asian control variate** | Arithmetic-Asian price; ~two orders of magnitude variance reduction vs plain MC |
| Quasi-MC | Scrambled **Sobol** with **Brownian-bridge** path construction | Faster convergence in moderate dimension |
| Lattice | **Cox-Ross-Rubinstein** binomial tree (vanilla European/American) | Independent validation against Black-Scholes |
| Greeks | **Finite differences with Common Random Numbers** + **Broadie–Glasserman pathwise** Δ | Two independent Δ estimators must agree |
| Calibration | **Brent's method** on the inverse pricing function | Implied volatility from a market quote |
| Risk | Hull-style **delta-hedging simulation** with `(m−k)/m` Asian-Δ proxy | Operational P&L distribution after hedging |

### The headline result — variance reduction

For an at-the-money 1-year Asian call (S₀ = K = 80, σ = 32%, r = 4.5%,
m = 252, N = 50 000 paths):

| Method | Price | Std error | Variance reduction |
|--------|-------|-----------|--------------------|
| Plain Monte Carlo | 6.635 | 0.0460 | 1× |
| MC + Antithetic | 6.691 | 0.0354 | 1.7× |
| **MC + Geometric Control Variate** | **6.645** | **0.0021** | **~480×** |
| Quasi-MC (Sobol + Brownian bridge) | 6.643 | 0.080 ⁽¹⁾ | — |

⁽¹⁾ QMC's "standard error" via the sample-std proxy over-states the true
error; properly measured (randomized scrambled Sobol), QMC converges at
~O(N⁻¹).

---

## Sensitivity & risk analysis

Six interactive visuals — every one driven by the same engine call:

- **Interactive 1-D sweep** (price vs S₀ / K / σ / T / r) with a draggable marker.
- **2-D price-surface heatmap** for any pair of parameters; current operating point overlaid.
- **Tornado chart** ranking parameters by ±-shock impact.
- **Scenario stress table** — the classical ±10% S, ±5pp σ, +30d, ±50bp r ladder.
- **P&L attribution waterfall** — decomposes any scenario shock into Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + Θ·ΔT + ρ·Δr + residual.
- **Delta-hedging P&L distribution** — short the option, hedge at chosen frequency, show how tight the post-hedge P&L gets.

---

## Mathematics

**Discrete geometric Asian — Kemna-Vorst closed form.** With monitoring at
$t_i = i\,T/m$ for $i=1,\dots,m$:

$$
\sigma_g^2 = \sigma^2 \cdot \frac{(m+1)(2m+1)}{6m^2}, \qquad
\nu = (r-q-\tfrac12\sigma^2)\cdot\frac{m+1}{2m}, \qquad
\mu_{kv} = \nu + \tfrac12\sigma_g^2.
$$

$$
C_g = e^{-rT}\!\left[\,S_0 e^{\mu_{kv}T}\, N(d_1) \;-\; K\, N(d_2)\right],\qquad
d_1 = \frac{\ln(S_0/K) + (\mu_{kv}+\tfrac12\sigma_g^2)T}{\sigma_g\sqrt{T}},\;
d_2 = d_1 - \sigma_g\sqrt{T}.
$$

**Control-variate estimator.** With $Y_i$ = discounted arithmetic-Asian
payoff and $X_i$ = discounted geometric-Asian payoff on the *same* path,

$$
\hat{Y}_{\text{cv}} \;=\; \frac{1}{N}\sum_{i=1}^N\!\Big[Y_i - \beta\,(X_i - \mu_X)\Big],
\qquad
\beta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},
$$

where $\mu_X$ is the Kemna-Vorst price. Variance reduces by the factor
$1-\rho^2$; for arithmetic Asians under GBM, $\rho \approx 0.99$.

---

## Validation

Every claim in this README is enforced by a test that runs on every commit:

```bash
$ pytest tests/ -q
.........................                                          [100%]
25 passed in 8.48s
```

| Test class | What it asserts |
|------------|----------------|
| `TestClosedForm` | KV positivity, put-call parity, continuous-limit stability |
| `TestMonteCarloConvergence` | MC-geometric matches KV within 2·SE |
| `TestVarianceReduction` | Antithetic < Plain; CV ≥ 5× tighter than Antithetic; ρ > 0.98 |
| `TestTrees` | CRR → BS within 1 bp at 2 000 steps; put-call parity on the tree |
| `TestGreeks` | Call Δ ∈ [0,1], put Δ ∈ [-1,0], Vega > 0, Theta < 0 (correct sign); FD-Δ and pathwise-Δ agree |
| `TestDiagnostics` | Reality checks fire on deep OTM / extreme vol / bad inputs |
| `TestHedging` | Hedge reduces variance; finer rebalance helps; mean P&L ≈ 0 |
| `TestCalibration` | Round-trip σ → price → σ recovers input within 0.1% |

---

## Architecture

```
asian-option-pricer/
├── engine/                  # All math lives here (zero UI code)
│   ├── contracts.py         # OptionContract dataclass + enums
│   ├── results.py           # PricingResult / GreeksResult containers
│   ├── analytic.py          # Kemna-Vorst, Black-Scholes
│   ├── paths.py             # Vectorised GBM paths (pseudo + Sobol+BB)
│   ├── mc.py                # MC pricers (plain / antithetic / CV / QMC)
│   ├── trees.py             # CRR binomial tree
│   ├── greeks.py            # FD-CRN + pathwise Δ
│   ├── hedging.py           # Dynamic Δ-hedging simulation
│   ├── calibration.py       # Brent's method for implied vol
│   ├── sensitivities.py     # 1D / 2D / tornado / scenario / attribution
│   ├── diagnostics.py       # Reality-check warnings
│   └── presets.py           # Brent / WIG20 / EUR-USD / Nifty defaults
├── tests/                   # pytest suite (25 tests)
├── app.py                   # Streamlit UI (thin)
├── .streamlit/config.toml   # Theme
├── .github/workflows/ci.yml # Multi-version CI
├── requirements.txt
└── README.md
```

The engine has zero Streamlit dependencies — it can be imported and used
from a script, notebook, or another web framework.

---

## Real-world presets

The dashboard ships with four ready-made scenarios, each with realistic
parameters and a concrete buyer narrative:

| Preset | Market | Why this contract exists |
|--------|--------|--------------------------|
| **Brent Crude — Refiner Hedge** | $80/bbl, σ = 32%, T = 1y, m = 252 | Refiner buying ~100 k bbl/month locks in the annual average price |
| **WIG20 — ETF Tracking Overlay** | 2 500 PLN, σ = 18%, T = 6m, m = 126 | Polish fund hedges a 6-month average-linked structured product |
| **EUR/USD — Exporter FX Hedge** | 1.08, σ = 8%, T = 3m, m = 63 | European exporter hedges quarterly USD receivables at the average rate |
| **Nifty 50 — Yield Enhancement** | 24 000 INR, σ = 17%, T = 1y, m = 252 | Indian PMS sells OTM Asian calls on the portfolio for premium income |

---

## Model limitations (stated up-front)

Every model lies. This one lies in these specific ways, and the report
discusses each in writing rather than hiding them:

- Constant volatility — no smile/skew (real markets have one).
- No jumps — GBM is continuous (commodities have news jumps).
- Flat interest-rate curve.
- No transaction costs in the pricer (the hedge sim has an optional bps knob).
- The delta-hedging simulation uses a fast practitioner approximation to
  the Asian Δ at intermediate times (BS Δ on the residual contract,
  scaled by the un-realised fraction of the average). It is conservative.

---

## License

MIT. See [LICENSE](LICENSE).

## Author

**Prabhakar** — MSc Quantitative Finance, Poznań University of Economics
and Business (UEP). Project for *Advanced Corporate Finance* (2026).
