# Asian Option Pricing Engine

[![CI](https://github.com/gautamprasadwork/asian-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/gautamprasadwork/asian-option-pricer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Prices **arithmetic Asian options** under Black–Scholes dynamics using Monte
Carlo with a geometric-Asian control variate, validated against a binomial
tree. Ships with an interactive Streamlit dashboard for pricing, Greeks,
delta-hedging, calibration, and sensitivity analysis.

## Live 
https://asianoptionprice.streamlit.app/

## Run it

```bash
git clone https://github.com/gautamprasadwork/asian-option-pricer.git
cd asian-option-pricer
pip install -r requirements.txt
streamlit run app.py
```

## What's inside

- **Pricing** — Monte Carlo (antithetic + geometric control variate, ~480× variance reduction), Quasi-MC (Sobol + Brownian bridge), and a Kemna–Vorst closed form.
- **Greeks** — finite differences (common random numbers) plus a pathwise delta.
- **Risk** — dynamic delta-hedging simulation, implied-vol calibration, and 1-D/2-D/tornado/scenario sensitivity tools.

All math lives in `engine/` (no UI dependencies); `app.py` is a thin Streamlit layer.

## Tests

```bash
pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Gautam Prasad** — Poznań University of Economics and Business (UEP), 2026.
