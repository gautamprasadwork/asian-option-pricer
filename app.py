"""
Asian Option Pricer — single-page interactive dashboard.

A focused dashboard for pricing arithmetic Asian call options using Monte
Carlo simulation with a geometric-Asian control variate. Built as a master's
project in Quantitative Finance at UEP Poznań.

The dashboard prioritises clarity, intuition and visual quality over feature
sprawl. Sections (top → bottom):

    1. KPI cards          (price · CI · SE · runtime · variance reduction)
    2. Simulated paths    (40 GBM paths with the average highlighted)
    3. MC convergence     (running estimate ± 95% CI as N grows)
    4. Sensitivity        (price vs σ, price vs S, S × σ heatmap)
    5. Variance reduction (Plain MC vs MC + CV)
    6. Educational notes  (collapsible explainers)

Engine modules in `engine/` do all the math; this file is only orchestration
and presentation.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (  # noqa: E402
    OptionContract, OptionType, Averaging, StrikeType,
    price_mc, price_mc_cv, kemna_vorst_price,
    simulate_gbm_paths, compute_greeks,
)


# Plotly chart helper: hide the modebar everywhere for a cleaner look.
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

def st_plot(fig: "go.Figure"):
    """Render a Plotly figure without the modebar (zoom/pan/download icons)."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ============================================================================
#  Page config
# ============================================================================
st.set_page_config(
    page_title="Asian Option Pricer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Asian Option Pricer — Monte Carlo with geometric-Asian "
                 "control variate. Quantitative Finance project, UEP Poznań."
    },
)


# ============================================================================
#  Dark theme palette  (kept in sync with .streamlit/config.toml)
# ============================================================================
BG       = "#0A0E13"
SURFACE  = "#131820"
CARD     = "#1A1F2A"
BORDER   = "#232B38"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"
ACCENT   = "#58A6FF"          # electric blue
ACCENT_SOFT = "rgba(88,166,255,0.12)"
GREEN    = "#7EE787"
AMBER    = "#F7CA88"
RED      = "#F85149"


# ============================================================================
#  Custom CSS
# ============================================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
      background: {BG};
      color: {TEXT};
  }}
  * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }}
  .num, .num * {{
      font-family: 'JetBrains Mono', monospace !important;
      font-variant-numeric: tabular-nums;
  }}

  /* Strip Streamlit default chrome */
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  .block-container {{ padding-top: 2rem; max-width: 1400px; }}

  /* ---- Hero ---- */
  .hero-title {{
      font-size: 34px; font-weight: 700; letter-spacing: -1.2px;
      margin: 0 0 4px 0; color: {TEXT};
  }}
  .hero-title .accent {{ color: {ACCENT}; }}
  .hero-sub {{
      font-size: 14px; color: {MUTED}; margin-bottom: 28px;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* ---- Section headings ---- */
  .section-title {{
      font-size: 20px; font-weight: 600; color: {TEXT};
      letter-spacing: -0.3px; margin: 18px 0 4px 0;
  }}
  .section-caption {{
      font-size: 13.5px; color: {MUTED}; line-height: 1.6;
      max-width: 880px; margin: 6px 0 16px 0;
  }}
  .section-caption b {{ color: {TEXT}; }}

  /* ---- KPI cards ---- */
  .kpi {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 18px 20px;
      height: 100%;
      transition: border-color 0.18s ease, transform 0.18s ease;
  }}
  .kpi:hover {{
      border-color: rgba(88,166,255,0.4);
      transform: translateY(-1px);
  }}
  .kpi-label {{
      font-size: 10.5px; font-weight: 600; color: {MUTED};
      letter-spacing: 0.7px; text-transform: uppercase; margin-bottom: 10px;
  }}
  .kpi-value {{
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      font-size: 28px; font-weight: 600; color: {TEXT};
      line-height: 1.1;
  }}
  .kpi-sub {{
      font-size: 11.5px; color: {MUTED}; margin-top: 6px;
      font-family: 'JetBrains Mono', monospace;
  }}
  .kpi.accent {{ border-left: 3px solid {ACCENT}; }}
  .kpi.accent .kpi-value {{ color: {ACCENT}; }}
  .kpi.good   {{ border-left: 3px solid {GREEN}; }}
  .kpi.good   .kpi-value {{ color: {GREEN}; }}

  /* ---- Greek cards (compact, sit below KPI row) ---- */
  .greek {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 12px 14px;
      height: 100%;
      transition: border-color 0.18s ease;
  }}
  .greek:hover {{ border-color: rgba(88,166,255,0.35); }}
  .greek-symbol {{
      font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
      color: {ACCENT}; font-family: 'JetBrains Mono', monospace;
      margin-bottom: 4px;
  }}
  .greek-value {{
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      font-size: 18px; font-weight: 600; color: {TEXT};
      line-height: 1.15;
  }}
  .greek-units {{
      font-size: 10.5px; color: {MUTED}; margin-top: 4px;
  }}

  /* ---- Sidebar polish ---- */
  section[data-testid="stSidebar"] {{
      background: {SURFACE};
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3,
  section[data-testid="stSidebar"] .stMarkdown h4,
  section[data-testid="stSidebar"] .stMarkdown h5 {{
      color: {MUTED} !important;
      font-size: 11px !important;
      font-weight: 600 !important;
      letter-spacing: 0.6px !important;
      text-transform: uppercase !important;
      margin: 18px 0 6px 0 !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
      color: {MUTED} !important;
      font-size: 12px !important;
      font-weight: 500 !important;
  }}

  /* ---- Expanders (educational notes) ---- */
  [data-testid="stExpander"] {{
      background: {CARD};
      border: 1px solid {BORDER} !important;
      border-radius: 8px;
      margin-bottom: 8px;
  }}
  [data-testid="stExpander"] summary {{
      font-weight: 500;
      color: {TEXT};
      padding: 10px 16px;
  }}

  /* ---- Dividers ---- */
  hr {{
      border-color: {BORDER} !important;
      margin: 28px 0 !important;
  }}

  /* ---- Tables ---- */
  [data-testid="stDataFrame"] {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 8px;
  }}

  /* ---- Floating right-side jump nav ---- */
  .nav-jump {{
      position: fixed;
      top: 86px;
      right: 22px;
      z-index: 999;
      background: rgba(19, 24, 32, 0.82);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 8px 6px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      font-size: 11px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.35);
  }}
  .nav-jump-label {{
      font-size: 9px; font-weight: 700; color: {MUTED};
      letter-spacing: 1px; text-transform: uppercase;
      padding: 2px 10px 4px 10px;
      border-bottom: 1px solid {BORDER};
      margin-bottom: 4px;
  }}
  .nav-jump a {{
      color: {MUTED};
      text-decoration: none;
      padding: 5px 12px;
      border-radius: 6px;
      transition: color 0.15s, background 0.15s;
      font-weight: 500;
      letter-spacing: 0.2px;
      white-space: nowrap;
  }}
  .nav-jump a:hover {{
      color: {ACCENT};
      background: rgba(88, 166, 255, 0.10);
  }}

  /* Smooth scroll when jumping to anchors */
  html {{ scroll-behavior: smooth; }}
  /* Offset anchor targets so the fixed header doesn't cover them */
  [id^="sec-"] {{ scroll-margin-top: 40px; }}

  @media (max-width: 1200px) {{
      .nav-jump {{ display: none; }}
  }}
</style>
""", unsafe_allow_html=True)


# ---- Floating jump-nav anchored top-right ---------------------------------
st.markdown("""
<div class="nav-jump">
  <div class="nav-jump-label">Jump to</div>
  <a href="#sec-price">Price</a>
  <a href="#sec-paths">Paths</a>
  <a href="#sec-conv">Convergence</a>
  <a href="#sec-sens">Sensitivity</a>
  <a href="#sec-vr">Variance Reduction</a>
</div>
""", unsafe_allow_html=True)


# ============================================================================
#  Plotly dark theme helper
# ============================================================================
def style_dark(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    layout_kwargs = dict(
        height=height,
        margin=dict(l=10, r=10, t=42 if title else 12, b=10),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", size=12, color=MUTED),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=CARD, bordercolor=ACCENT,
            font=dict(family="JetBrains Mono", size=12, color=TEXT),
        ),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title,
            font=dict(size=13, color=TEXT, family="Inter, sans-serif"),
            x=0.0, xanchor="left",
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor=BORDER, linecolor=BORDER, ticks="outside",
        tickcolor=BORDER, zeroline=False, color=MUTED,
        title_font=dict(size=11, color=MUTED),
    )
    fig.update_yaxes(
        gridcolor=BORDER, linecolor=BORDER, ticks="outside",
        tickcolor=BORDER, zeroline=False, color=MUTED,
        title_font=dict(size=11, color=MUTED),
    )
    return fig


# ============================================================================
#  Sidebar: parameters
# ============================================================================
with st.sidebar:
    st.markdown("##### Contract")
    option_type_label = st.radio(
        "Option type", ["Call", "Put"], horizontal=True, index=0,
        help="Call: right to buy at K. Put: right to sell at K.",
    )

    st.markdown("##### Underlying")
    S0 = st.number_input("Spot price S₀", min_value=1.0, max_value=10000.0,
                          value=100.0, step=1.0)
    K  = st.number_input("Strike price K",  min_value=1.0, max_value=10000.0,
                          value=100.0, step=1.0)

    st.markdown("##### Market")
    sigma = st.slider("Volatility σ", min_value=0.01, max_value=1.00,
                      value=0.20, step=0.01, format="%.2f")
    r = st.slider("Risk-free rate r", min_value=0.00, max_value=0.10,
                  value=0.05, step=0.005, format="%.3f")
    T = st.slider("Maturity T (years)", min_value=0.10, max_value=3.00,
                  value=1.00, step=0.05)

    st.markdown("##### Simulation")
    N = st.selectbox("Number of simulations",
                     options=[5_000, 10_000, 25_000, 50_000, 100_000],
                     index=3, format_func=lambda x: f"{x:,}")
    m_per_year = st.selectbox(
        "Averaging frequency",
        options=[12, 52, 252], index=2,
        format_func=lambda x: {12: "Monthly (12 / year)",
                               52: "Weekly  (52 / year)",
                               252: "Daily   (252 / year)"}[x],
        help=("How often the underlying price is sampled to compute the "
              "average that drives the payoff. 252 trading days = 1 year, "
              "so 'daily' over 1y means m = 252 sample dates."),
    )
    use_cv = st.toggle("Control Variate", value=True,
                       help="Apply geometric-Asian control variate to reduce variance")

# Effective number of monitoring dates over [0, T]
m_eff = max(1, int(round(m_per_year * T)))
option_type_value = option_type_label.lower()  # "call" / "put"
is_call = (option_type_value == "call")


# ============================================================================
#  Build contract & price both methods (cached)
# ============================================================================
@st.cache_data(show_spinner=False)
def price_both(S0: float, K: float, r: float, sigma: float, T: float,
               m: int, N: int, option_type: str):
    c = OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=OptionType(option_type),
        averaging=Averaging.ARITHMETIC,
        strike_type=StrikeType.FIXED,
        m=m, q=0.0,
    )
    plain = price_mc(c, n_paths=N, seed=42)
    cv    = price_mc_cv(c, n_paths=N, seed=42)
    return plain, cv, c


@st.cache_data(show_spinner=False)
def compute_greeks_cached(S0: float, K: float, r: float, sigma: float, T: float,
                           m: int, N: int, option_type: str):
    c = OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=OptionType(option_type),
        averaging=Averaging.ARITHMETIC,
        strike_type=StrikeType.FIXED,
        m=m, q=0.0,
    )
    return compute_greeks(c, n_paths=max(N // 2, 10_000), seed=42)


with st.spinner("Running Monte Carlo simulation…"):
    plain_result, cv_result, contract = price_both(
        S0, K, r, sigma, T, m_eff, N, option_type_value,
    )
chosen = cv_result if use_cv else plain_result


# ============================================================================
#  Hero
# ============================================================================
st.markdown(
    '<div class="hero-title">Asian Option <span class="accent">Pricer</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero-sub">Arithmetic Asian {option_type_label} · '
    f'Monte Carlo with Geometric Control Variate · '
    f'{N:,} simulations · {m_eff} monitoring dates</div>',
    unsafe_allow_html=True,
)


# ============================================================================
#  SECTION 1 — KPI cards
# ============================================================================
lo, hi = chosen.ci_95
if use_cv and plain_result.std_error > 0 and cv_result.std_error > 0:
    var_red_ratio = (plain_result.std_error / cv_result.std_error) ** 2
    var_red_pct   = (1.0 - (cv_result.std_error / plain_result.std_error) ** 2) * 100
    vr_value = f"{var_red_ratio:.0f}×"
    vr_sub = f"{var_red_pct:.1f}% less variance vs plain MC"
    vr_class = "good"
else:
    vr_value = "—"
    vr_sub = "Control variate disabled"
    vr_class = ""

cards = [
    ("Option Price", f"{chosen.price:.4f}",
     "Geometric CV" if use_cv else "Plain MC", "accent"),
    ("95% Confidence Interval", f"±{(hi - lo) / 2:.4f}",
     f"[{lo:.4f}, {hi:.4f}]", ""),
    ("Standard Error", f"{chosen.std_error:.5f}",
     f"of the price estimate", ""),
    ("Runtime", f"{chosen.runtime_ms:.0f} ms",
     f"{chosen.n_paths:,} paths", ""),
    ("Variance Reduction", vr_value, vr_sub, vr_class),
]
cols = st.columns(5, gap="small")
for col, (label, value, sub, cls) in zip(cols, cards):
    col.markdown(f"""
    <div class="kpi {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value num">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Greeks strip (Δ Γ ν Θ ρ) ----------
with st.spinner("Computing risk sensitivities (Greeks)…"):
    greeks = compute_greeks_cached(
        S0, K, r, sigma, T, m_eff, N, option_type_value,
    )

greek_rows = [
    ("Δ  Delta", f"{greeks.delta.value:+.4f}", "per €1 of spot"),
    ("Γ  Gamma", f"{greeks.gamma.value:+.5f}", "delta convexity"),
    ("ν  Vega",  f"{greeks.vega.value:+.4f}",  "per 1% of vol"),
    ("Θ  Theta", f"{greeks.theta.value:+.4f}", "per calendar day"),
    ("ρ  Rho",   f"{greeks.rho.value:+.4f}",   "per 1% of rate"),
]
gcols = st.columns(5, gap="small")
for col, (sym, val, units) in zip(gcols, greek_rows):
    col.markdown(f"""
    <div class="greek">
        <div class="greek-symbol">{sym}</div>
        <div class="greek-value num">{val}</div>
        <div class="greek-units">{units}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
#  SECTION 2 — Simulated paths
# ============================================================================
st.markdown('<div class="section-title">Simulated Price Paths</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Monte Carlo generates many possible futures '
    'for the underlying. The Asian option pays based on the <b>average</b> '
    'of each path, not its endpoint — so the entire shape matters.</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def make_viz_paths(S0: float, r: float, sigma: float, T: float, m: int,
                    n_viz: int = 40):
    rng = np.random.default_rng(123)
    paths = simulate_gbm_paths(S0, r, sigma, T, m, n_paths=n_viz,
                                antithetic=False, rng=rng)
    # Prepend S(0) so paths visually start at the spot
    full = np.hstack([np.full((n_viz, 1), S0), paths])
    return full, np.linspace(0, T, m + 1)


viz_paths, times = make_viz_paths(S0, r, sigma, T, m_eff)

fig_paths = go.Figure()
for i in range(viz_paths.shape[0]):
    fig_paths.add_trace(go.Scatter(
        x=times, y=viz_paths[i],
        mode="lines",
        line=dict(color="rgba(139,148,158,0.30)", width=0.9),
        hovertemplate=f"path {i+1}<br>t=%{{x:.3f}}y<br>S=%{{y:.3f}}<extra></extra>",
        showlegend=False,
    ))
# Average path highlighted
avg_path = viz_paths.mean(axis=0)
fig_paths.add_trace(go.Scatter(
    x=times, y=avg_path, mode="lines",
    line=dict(color=ACCENT, width=3),
    name="Average across paths",
    hovertemplate="<b>Average</b><br>t=%{x:.3f}y<br>S̄=%{y:.3f}<extra></extra>",
))
# Strike reference
fig_paths.add_hline(
    y=K, line=dict(color=AMBER, width=1, dash="dash"),
    annotation_text=f"  K = {K:g}",
    annotation_font=dict(color=AMBER, size=11, family="JetBrains Mono"),
    annotation_position="top right",
)
style_dark(fig_paths, height=420)
fig_paths.update_xaxes(title="Time (years)")
fig_paths.update_yaxes(title="Asset price")
st_plot(fig_paths)
st.markdown(
    '<div class="section-caption">Arithmetic Asian options depend on the '
    'average asset price over time rather than only the terminal price. '
    'This smoothes the payoff and reduces sensitivity to single-day price spikes.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================================
#  SECTION 3 — Monte Carlo convergence
# ============================================================================
st.markdown('<div class="section-title">Monte Carlo Convergence</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The estimator stabilises as more paths are '
    'added, with the 95% confidence band shrinking at rate <b>1 / √N</b>. To '
    'halve the standard error, you need 4× more simulations.</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def compute_convergence(S0: float, K: float, r: float, sigma: float, T: float,
                         m: int, max_N: int, is_call: bool):
    """Compute running mean + 95% CI of the plain MC estimator as N grows.

    Cheap because everything is vectorised in numpy — no Python loop over N.
    """
    rng = np.random.default_rng(42)
    dt = T / m
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = rng.standard_normal((max_N, m))
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(log_paths)
    A = paths.mean(axis=1)
    if is_call:
        payoff = np.maximum(A - K, 0.0)
    else:
        payoff = np.maximum(K - A, 0.0)
    Y = payoff * np.exp(-r * T)

    cum   = np.cumsum(Y)
    cumsq = np.cumsum(Y ** 2)
    N_grid = np.arange(1, max_N + 1)
    mean = cum / N_grid
    var = np.maximum(cumsq / N_grid - mean ** 2, 0.0)
    se = np.sqrt(var / N_grid)

    # Subsample log-spaced indices for plotting
    plot_idx = np.unique(np.logspace(2, np.log10(max_N), 120).astype(int))
    return N_grid[plot_idx], mean[plot_idx], se[plot_idx]


with st.spinner("Computing convergence…"):
    Ns_conv, prices_conv, ses_conv = compute_convergence(
        S0, K, r, sigma, T, m_eff, max(N, 50_000), is_call,
    )

upper = prices_conv + 1.96 * ses_conv
lower = prices_conv - 1.96 * ses_conv

fig_conv = go.Figure()
fig_conv.add_trace(go.Scatter(
    x=np.concatenate([Ns_conv, Ns_conv[::-1]]),
    y=np.concatenate([upper, lower[::-1]]),
    fill="toself", fillcolor=ACCENT_SOFT,
    line=dict(color="rgba(0,0,0,0)"),
    name="95% confidence band",
    hoverinfo="skip",
))
fig_conv.add_trace(go.Scatter(
    x=Ns_conv, y=prices_conv,
    mode="lines", line=dict(color=ACCENT, width=2),
    name="Running estimate",
    hovertemplate="N=%{x:,}<br>price=%{y:.4f}<extra></extra>",
))
final_price = float(prices_conv[-1])
fig_conv.add_hline(
    y=final_price, line=dict(color=GREEN, width=1, dash="dash"),
    annotation_text=f"  converged: {final_price:.4f}",
    annotation_position="bottom right",
    annotation_font=dict(color=GREEN, size=11, family="JetBrains Mono"),
)
style_dark(fig_conv, height=380)
fig_conv.update_xaxes(type="log", title="Number of simulations (log scale)")
fig_conv.update_yaxes(title="Estimated price")
st_plot(fig_conv)
st.markdown(
    '<div class="section-caption">As the number of simulations increases, '
    'the Monte Carlo estimate stabilises and becomes more accurate.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================================
#  SECTION 4 — Sensitivity analysis
# ============================================================================
st.markdown('<div class="section-title">Sensitivity Analysis</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">How does the price respond when we vary '
    'one or two inputs? Sensitivity analysis reveals which parameters move '
    'the price the most — the foundation of risk management.</div>',
    unsafe_allow_html=True,
)


def _base_contract(S0: float, K: float, r: float, sigma: float, T: float,
                    m: int, option_type: str) -> OptionContract:
    return OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=OptionType(option_type),
        averaging=Averaging.ARITHMETIC,
        strike_type=StrikeType.FIXED,
        m=m, q=0.0,
    )


@st.cache_data(show_spinner=False)
def sweep_sigma(S0: float, K: float, r: float, T: float, m: int, N: int,
                 option_type: str):
    sigmas = np.linspace(0.05, 0.60, 26)
    base = _base_contract(S0, K, r, 0.20, T, m, option_type)
    prices = [price_mc_cv(base.with_(sigma=float(s)), n_paths=N).price
              for s in sigmas]
    return sigmas, prices


@st.cache_data(show_spinner=False)
def sweep_spot(K: float, r: float, sigma: float, T: float, m: int, N: int,
                option_type: str):
    spots = np.linspace(0.55 * K, 1.45 * K, 26)
    base = _base_contract(K, K, r, sigma, T, m, option_type)
    prices = [price_mc_cv(base.with_(S0=float(s)), n_paths=N).price
              for s in spots]
    return spots, prices


@st.cache_data(show_spinner=False)
def sweep_maturity(S0: float, K: float, r: float, sigma: float, m_per_year: int,
                    N: int, option_type: str):
    Ts = np.linspace(0.10, 3.0, 26)
    prices = []
    for T_i in Ts:
        m_i = max(1, int(round(m_per_year * T_i)))
        c = _base_contract(S0, K, r, sigma, float(T_i), m_i, option_type)
        prices.append(price_mc_cv(c, n_paths=N).price)
    return Ts, prices


@st.cache_data(show_spinner=False)
def sweep_strike(S0: float, r: float, sigma: float, T: float, m: int, N: int,
                  option_type: str):
    Ks = np.linspace(0.55 * S0, 1.45 * S0, 26)
    base = _base_contract(S0, S0, r, sigma, T, m, option_type)
    prices = [price_mc_cv(base.with_(K=float(k)), n_paths=N).price
              for k in Ks]
    return Ks, prices


@st.cache_data(show_spinner=False)
def heatmap_spot_vol(K: float, r: float, T: float, m: int, N: int,
                     option_type: str):
    spots = np.linspace(0.70 * K, 1.30 * K, 14)
    sigmas = np.linspace(0.10, 0.50, 12)
    Z = np.zeros((len(sigmas), len(spots)))
    base = _base_contract(K, K, r, 0.20, T, m, option_type)
    for i, sg in enumerate(sigmas):
        for j, sp in enumerate(spots):
            Z[i, j] = price_mc_cv(
                base.with_(S0=float(sp), sigma=float(sg)),
                n_paths=N,
            ).price
    return spots, sigmas, Z


# Use a smaller N for sensitivity (the SHAPE of the curve matters, not precision)
N_sens = max(N // 4, 5_000)

with st.spinner("Building sensitivity charts…"):
    sigmas_x, prices_vs_sigma = sweep_sigma(S0, K, r, T, m_eff, N_sens, option_type_value)
    spots_x,  prices_vs_spot  = sweep_spot(K, r, sigma, T, m_eff, N_sens, option_type_value)
    Ts_x,     prices_vs_T     = sweep_maturity(S0, K, r, sigma, m_per_year, N_sens, option_type_value)
    Ks_x,     prices_vs_K     = sweep_strike(S0, r, sigma, T, m_eff, N_sens, option_type_value)
    heat_S, heat_sig, heat_Z  = heatmap_spot_vol(K, r, T, m_eff, max(N // 8, 3_000), option_type_value)


def _sweep_chart(xs, ys, current_x, current_y, color, fill_color,
                  title: str, x_label: str, x_format: str = "",
                  ref_lines=None):
    """Build a single sweep panel: filled curve + amber current marker + dashed reference."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate=f"{x_label}=%{{x:{x_format}}}<br>price=%{{y:.4f}}<extra></extra>",
        showlegend=False, name="price",
    ))
    fig.add_trace(go.Scatter(
        x=[current_x], y=[current_y], mode="markers",
        marker=dict(size=12, color=AMBER, line=dict(color="white", width=2)),
        hovertemplate=f"<b>current</b><br>{x_label}=%{{x:{x_format}}}<br>price=%{{y:.4f}}<extra></extra>",
        showlegend=False, name="current",
    ))
    fig.add_vline(x=current_x, line=dict(color=AMBER, width=1, dash="dot"))
    if ref_lines:
        for rx, rlabel in ref_lines:
            fig.add_vline(
                x=rx, line=dict(color=MUTED, width=1, dash="dash"),
                annotation_text=f"  {rlabel}",
                annotation_font=dict(color=MUTED, size=10),
                annotation_position="top right",
            )
    style_dark(fig, height=290, title=title)
    return fig


# --- 2×2 grid of univariate sweeps ------------------------------------------
row1 = st.columns(2, gap="medium")
row2 = st.columns(2, gap="medium")

# A. Price vs Volatility
cur_idx_sig = int(np.argmin(np.abs(np.array(sigmas_x) - sigma)))
fig_a = _sweep_chart(
    sigmas_x, prices_vs_sigma, sigma, prices_vs_sigma[cur_idx_sig],
    ACCENT, ACCENT_SOFT,
    "Price vs Volatility (σ)", "σ", ".0%",
)
fig_a.update_xaxes(title="σ (annualised)", tickformat=".0%")
fig_a.update_yaxes(title="Option price")
with row1[0]:
    st_plot(fig_a)

# B. Price vs Spot
cur_idx_s = int(np.argmin(np.abs(np.array(spots_x) - S0)))
fig_b = _sweep_chart(
    spots_x, prices_vs_spot, S0, prices_vs_spot[cur_idx_s],
    GREEN, "rgba(126,231,135,0.10)",
    "Price vs Spot (S₀)", "S₀", ".2f",
    ref_lines=[(K, f"K={K:g}")],
)
fig_b.update_xaxes(title="Spot price S₀")
fig_b.update_yaxes(title="Option price")
with row1[1]:
    st_plot(fig_b)

# C. Price vs Maturity T
cur_idx_T = int(np.argmin(np.abs(np.array(Ts_x) - T)))
fig_c = _sweep_chart(
    Ts_x, prices_vs_T, T, prices_vs_T[cur_idx_T],
    "#F0B441", "rgba(240,180,65,0.10)",
    "Price vs Maturity (T)", "T", ".2f",
)
fig_c.update_xaxes(title="Maturity T (years)")
fig_c.update_yaxes(title="Option price")
with row2[0]:
    st_plot(fig_c)

# D. Price vs Strike K
cur_idx_K = int(np.argmin(np.abs(np.array(Ks_x) - K)))
fig_d = _sweep_chart(
    Ks_x, prices_vs_K, K, prices_vs_K[cur_idx_K],
    "#C792EA", "rgba(199,146,234,0.10)",
    "Price vs Strike (K)", "K", ".2f",
    ref_lines=[(S0, f"S₀={S0:g}")],
)
fig_d.update_xaxes(title="Strike price K")
fig_d.update_yaxes(title="Option price")
with row2[1]:
    st_plot(fig_d)

# --- Heatmap below (full width) ----------------------------------------------
fig_heat = go.Figure(data=go.Heatmap(
    z=heat_Z, x=heat_S, y=heat_sig,
    colorscale=[
        [0.00, "#0F1822"],
        [0.30, "#1E3A6E"],
        [0.65, ACCENT],
        [1.00, GREEN],
    ],
    colorbar=dict(
        title=dict(text="Price", font=dict(size=11, color=MUTED)),
        thickness=12, len=0.85,
        tickfont=dict(family="JetBrains Mono", size=10, color=MUTED),
    ),
    hovertemplate="S₀=%{x:.2f}<br>σ=%{y:.0%}<br>price=%{z:.4f}<extra></extra>",
    showscale=True, showlegend=False, name="price",
))
fig_heat.add_trace(go.Scatter(
    x=[S0], y=[sigma], mode="markers",
    marker=dict(size=14, color=AMBER, symbol="x",
                line=dict(color="white", width=2)),
    hovertemplate=f"<b>current</b><br>S₀={S0:.2f}<br>σ={sigma:.0%}<extra></extra>",
    showlegend=False, name="current",
))
style_dark(fig_heat, height=380, title="Joint sensitivity: Spot × Volatility heatmap")
fig_heat.update_xaxes(title="Spot price S₀")
fig_heat.update_yaxes(title="Volatility σ", tickformat=".0%")
st_plot(fig_heat)

st.markdown(
    '<div class="section-caption">The four line charts show how the price '
    'responds to one parameter at a time; the heatmap shows the joint effect '
    'of spot and volatility. Read the steepest gradient as the region of '
    'highest risk concentration — where small parameter moves cause the '
    'largest price changes.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================================
#  SECTION 5 — Variance reduction comparison
# ============================================================================
st.markdown('<div class="section-title">Variance Reduction Comparison</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">The geometric-Asian control variate uses '
    'the closed-form Kemna-Vorst (1990) price as a per-path correction — '
    'removing most of the simulation noise without introducing bias.</div>',
    unsafe_allow_html=True,
)

vr_cols = st.columns([3, 4], gap="medium")

# Bar chart of SE
with vr_cols[0]:
    fig_vr = go.Figure()
    fig_vr.add_trace(go.Bar(
        x=["Plain MC", "MC + Control Variate"],
        y=[plain_result.std_error, cv_result.std_error],
        marker=dict(color=[MUTED, ACCENT], line=dict(width=0)),
        text=[f"{plain_result.std_error:.5f}", f"{cv_result.std_error:.5f}"],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color=TEXT),
        hovertemplate="%{x}<br>SE = %{y:.5f}<extra></extra>",
        showlegend=False,
    ))
    style_dark(fig_vr, height=340, title="Standard Error (lower is better)")
    fig_vr.update_yaxes(title="Standard error")
    fig_vr.update_xaxes(title=None)
    st_plot(fig_vr)

# Comparison table
with vr_cols[1]:
    if plain_result.std_error > 0 and cv_result.std_error > 0:
        vr_ratio_str = f"{(plain_result.std_error / cv_result.std_error) ** 2:.0f}×"
    else:
        vr_ratio_str = "—"
    df_vr = pd.DataFrame({
        "Method": ["Plain Monte Carlo", "MC + Control Variate"],
        "Price": [f"{plain_result.price:.4f}", f"{cv_result.price:.4f}"],
        "Std Error": [f"{plain_result.std_error:.5f}", f"{cv_result.std_error:.5f}"],
        "Runtime (ms)": [f"{plain_result.runtime_ms:.0f}", f"{cv_result.runtime_ms:.0f}"],
        "Variance Reduction": ["1.0×", vr_ratio_str],
    })
    st.dataframe(df_vr, hide_index=True, use_container_width=True)

    if plain_result.std_error > 0 and cv_result.std_error > 0:
        equiv_paths = int(N * (plain_result.std_error / cv_result.std_error) ** 2)
        st.markdown(
            f'<div class="section-caption" style="margin-top:8px;">'
            f'To match the CV estimator\'s precision, plain Monte Carlo would '
            f'need approximately <b>{equiv_paths:,} paths</b> '
            f'(vs <b>{N:,}</b> here) — the same accuracy in '
            f'<b>{(plain_result.std_error / cv_result.std_error) ** 2:.0f}×</b> '
            f'less compute.'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ============================================================================
#  SECTION 6 — Educational notes
# ============================================================================
st.markdown('<div class="section-title">Concepts</div>',
            unsafe_allow_html=True)

with st.expander("What is an Asian Option?"):
    st.markdown("""
An **Asian option** is an exotic option whose payoff depends on the **average price**
of the underlying asset over a monitoring window, rather than the price at expiry alone.
For a fixed-strike arithmetic Asian:

> **Call payoff** = max( Ā − K , 0 )    where Ā = (1/m) Σᵢ S(tᵢ)
> **Put  payoff** = max( K − Ā , 0 )

Asian options are widely used in commodity, FX and energy markets. An oil refiner that
buys crude every day cares about the **average** annual price, not the price on
31 December — so hedging the average matches the real economic exposure.

They are also **cheaper** than vanilla European options because averaging dampens
the path's volatility, and they are **harder to manipulate** because no single tick
determines the payoff.
    """)

with st.expander("What is Monte Carlo Simulation?"):
    st.markdown("""
**Monte Carlo simulation** prices an option by:

1. Generating many random future paths for the underlying asset.
2. Computing the option payoff on each path.
3. Averaging the discounted payoffs to estimate the fair price.

For an arithmetic Asian option there is **no closed-form solution** — the sum of
correlated lognormals isn't itself lognormal. Monte Carlo is the standard approach.

The estimator's accuracy improves at rate **1 / √N**, meaning that to **halve** the
standard error you need **4×** more simulations. This slow convergence is Monte
Carlo's main weakness — and the motivation for variance-reduction techniques.
    """)

with st.expander("Why do we use variance reduction?"):
    st.markdown("""
Plain Monte Carlo can require millions of paths to reach tight confidence intervals.
Variance reduction techniques tighten the standard error without introducing bias.

**Antithetic variates** pair each random path with its sign-flipped mirror. Roughly
halves the variance for symmetric payoffs.

**Control variates** exploit a related instrument whose true price is known. For the
arithmetic Asian, the **geometric Asian** has a closed-form solution
(*Kemna & Vorst, 1990*) and is ~99% correlated with the arithmetic Asian. Using
the geometric price as a per-path correction removes most of the simulation noise.

The variance-reduction factor is `(1 − ρ²)` — for ρ ≈ 0.99 that's a **~50× reduction
in variance** in theory, and 100×+ in practice once combined with antithetic variates.
    """)

with st.expander("What are the Greeks (Δ, Γ, ν, Θ, ρ)?"):
    st.markdown("""
The **Greeks** are first-order sensitivities of the option price to each input
parameter. They tell you how the price changes for a tiny move in one variable,
which is exactly what a trader needs to know to hedge.

| Symbol | Name  | What it measures                                  | Sign for a call (typically) |
|--------|-------|---------------------------------------------------|------------------------------|
| **Δ Delta** | Delta | ∂Price / ∂S₀ — change per €1 of spot       | 0 to 1 |
| **Γ Gamma** | Gamma | ∂Δ / ∂S₀ — curvature of the price in S₀    | > 0 |
| **ν Vega**  | Vega  | ∂Price / ∂σ — change per 1% change in vol  | > 0 |
| **Θ Theta** | Theta | ∂Price / ∂t — time decay per calendar day  | < 0 (option loses value over time) |
| **ρ Rho**   | Rho   | ∂Price / ∂r — change per 1% change in rate | > 0 for call, < 0 for put |

For Asian options, **all Greeks are smaller in absolute value** than for the
corresponding European vanilla option — the averaging dampens sensitivity to any
single moment in time. This is one of the key economic reasons Asian options
are popular as hedging instruments.

In this engine, Greeks are computed by **finite differences with common random numbers**
(same Brownian paths at the base and bumped parameters, so the noise cancels).
Delta is additionally cross-checked against an independent **pathwise estimator**
(Broadie & Glasserman, 1996) — they must agree, which is one of the validation tests.
    """)

with st.expander("Why 252 monitoring dates? What does m mean?"):
    st.markdown("""
**m** is the **number of monitoring dates** — the times at which the underlying
price is sampled to build the running average that drives the payoff. The choice
matters and is set by the contract specification, not by the model.

| Frequency | m / year | Typical use |
|-----------|----------|-------------|
| Monthly | 12 | Some FX Asian options, employee stock options |
| Weekly  | 52 | Mid-frequency commodity contracts |
| Daily   | **252** | The standard for liquid commodity (crude oil, jet fuel) and many index Asians |

**Why 252?** Markets have approximately **252 trading days per year** (the calendar
has 365 days but ~110 weekends + holidays are removed). So over a 1-year contract
with daily monitoring, the average is built from 252 daily price observations.

**Why m matters for our model:**
1. **Higher m → finer average.** With m = 252 the discrete arithmetic average is
   very close to the continuous-time average.
2. **The Kemna-Vorst formula depends on m explicitly** — the formula uses
   σ_g² = σ²·(m+1)(2m+1)/(6m²) and ν = (r-σ²/2)·(m+1)/(2m). As m → ∞ these
   converge to σ²/3 and (r-σ²/2)/2, the well-known continuous-monitoring limits.
3. **More sampling dates ⇒ slightly more compute** but a more realistic price
   for liquid markets.

For your report, you can simply say: *"The contract specifies daily monitoring
over the maturity period (m = 252 × T sampling dates), matching the standard
convention for OTC commodity Asian options."*
    """)


# ============================================================================
#  Footer
# ============================================================================
st.markdown("""
<div style="text-align:center; color:#586069; font-size:11px;
            padding: 30px 0 10px 0; letter-spacing: 0.3px;
            font-family: 'JetBrains Mono', monospace;">
  Asian Option Pricer · Monte Carlo with Geometric Control Variate ·
  Quantitative Finance Project · UEP Poznań 2026
</div>
""", unsafe_allow_html=True)
