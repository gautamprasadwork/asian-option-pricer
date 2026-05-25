"""
Realistic market presets with concrete buyer narratives.

Each preset bundles a fully-specified `OptionContract` with a short
"who buys this and why" use case. Together they ground the dashboard
in real markets instead of arbitrary toy numbers, and give the user
(or a recruiter looking at the project) instant context.

Parameter rationale
-------------------
- **Brent crude** (default). $80/bbl spot is roughly mid-cycle. Implied
  vol of 32% is typical for liquid crude options. Refiner hedging the
  *average* monthly purchase price is the textbook commodity Asian.
- **WIG20**. The Polish blue-chip index. Vol of 18% sits in the middle
  of its historical range; the 6-month maturity matches a typical
  structured-product reset window.
- **EUR/USD**. The most liquid FX pair. Vol of 8% is a normal regime.
  3-month Asian is a standard corporate-treasury hedge for export
  receivables.
- **Nifty 50**. The Indian benchmark. ~17% vol, 1-year tenor is a
  conventional portfolio overlay setup.

These numbers are illustrative defaults — they are not market quotes
and not investment advice. The methodology section makes this explicit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from engine.contracts import OptionContract, OptionType, Averaging, StrikeType


@dataclass(frozen=True)
class Preset:
    """A named, illustrative contract with a one-line use case."""
    key: str
    label: str
    contract: OptionContract
    currency: str
    use_case: str
    notes: str = ""

    def __str__(self) -> str:
        return f"{self.label}: {self.contract.describe()}"


PRESETS: Dict[str, Preset] = {
    "brent_refiner": Preset(
        key="brent_refiner",
        label="Brent Crude — Refiner Hedge",
        contract=OptionContract(
            S0=80.0,
            K=80.0,
            r=0.045,
            sigma=0.32,
            T=1.0,
            option_type=OptionType.CALL,
            averaging=Averaging.ARITHMETIC,
            strike_type=StrikeType.FIXED,
            m=252,
            q=0.0,
        ),
        currency="USD/bbl",
        use_case=(
            "An Indian refiner buys ~100,000 barrels/month of Brent. Locking in "
            "the annual *average* purchase price (not the spot at year-end) "
            "matches its real economic exposure — so it buys a 12-month "
            "fixed-strike arithmetic Asian call as a cap on the average."
        ),
        notes="Asian options are the standard hedge for commodity averages and dominate "
              "the OTC oil derivatives market.",
    ),
    "wig20_etf": Preset(
        key="wig20_etf",
        label="WIG20 — ETF Tracking Overlay",
        contract=OptionContract(
            S0=2500.0,
            K=2500.0,
            r=0.055,
            sigma=0.18,
            T=0.5,
            option_type=OptionType.CALL,
            averaging=Averaging.ARITHMETIC,
            strike_type=StrikeType.FIXED,
            m=126,
            q=0.025,
        ),
        currency="PLN",
        use_case=(
            "A Polish asset manager issuing a 6-month structured product linked "
            "to the average level of WIG20 hedges its short Asian exposure with "
            "a long arithmetic Asian call from the issuing bank."
        ),
        notes="Polish rates (NBP reference ~5.5%) and WIG20 dividend yield ~2.5%.",
    ),
    "eurusd_corp": Preset(
        key="eurusd_corp",
        label="EUR/USD — Exporter FX Hedge",
        contract=OptionContract(
            S0=1.08,
            K=1.08,
            r=0.04,
            sigma=0.08,
            T=0.25,
            option_type=OptionType.PUT,
            averaging=Averaging.ARITHMETIC,
            strike_type=StrikeType.FIXED,
            m=63,
            q=0.035,                # foreign (USD) rate acts like a dividend yield
        ),
        currency="EUR/USD",
        use_case=(
            "A European exporter invoicing in USD receives daily payments over "
            "Q1. Their P&L depends on the *average* EUR/USD rate, so they hedge "
            "with a 3-month arithmetic Asian put — much cheaper than 63 separate "
            "European puts."
        ),
        notes="Garman-Kohlhagen FX adaptation: q = foreign (USD) rate.",
    ),
    "nifty_overlay": Preset(
        key="nifty_overlay",
        label="Nifty 50 — Yield Enhancement",
        contract=OptionContract(
            S0=24000.0,
            K=25000.0,
            r=0.0675,
            sigma=0.17,
            T=1.0,
            option_type=OptionType.CALL,
            averaging=Averaging.ARITHMETIC,
            strike_type=StrikeType.FIXED,
            m=252,
            q=0.014,
        ),
        currency="INR",
        use_case=(
            "An Indian PMS/AIF holds a long Nifty 50 portfolio and sells "
            "out-of-the-money Asian calls (K = 25,000 vs spot 24,000) for "
            "premium income. The averaging makes the option cheaper to write "
            "than a vanilla and softens the call-away risk."
        ),
        notes="Indian repo ~6.75%; Nifty trailing dividend yield ~1.4%.",
    ),
}


# Order in which presets appear in the UI dropdown
PRESET_ORDER = [
    "brent_refiner", "wig20_etf", "eurusd_corp", "nifty_overlay",
]


def default_preset() -> Preset:
    return PRESETS["brent_refiner"]


def get_preset(key: str) -> Preset:
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{key}'. Available: {list(PRESETS)}")
    return PRESETS[key]
