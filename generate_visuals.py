"""
Generate the five report visuals (PNG) used in REPORT.md.

Run from the project root:
    python generate_visuals.py

Outputs are written to ./visuals/. The default contract matches the dashboard
defaults so the report numbers and visuals tell the same story.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from engine import (
    OptionContract, OptionType, Averaging, StrikeType,
    simulate_gbm_paths, price_mc, price_mc_cv,
)
from engine.mc import _compute_payoffs   # internal helper, used here just for the convergence chart  # noqa


# --------------------------------------------------------------------------- #
#  Palette (light, print-friendly)
# --------------------------------------------------------------------------- #
NAVY    = "#1E3A5F"
ACCENT  = "#3B82F6"
AMBER   = "#D97706"
GREEN   = "#16A34A"
RED     = "#DC2626"
BG      = "#FFFFFF"
GRID    = "#E2E8F0"
INK     = "#0F172A"
MUTED   = "#64748B"
LIGHT   = "#F8FAFC"


mpl.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.facecolor":      BG,
    "figure.facecolor":    BG,
    "savefig.facecolor":   BG,
    "axes.edgecolor":      GRID,
    "axes.linewidth":      1.0,
    "axes.grid":           True,
    "grid.color":          GRID,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.labelcolor":     MUTED,
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "axes.titlecolor":     INK,
    "axes.titlelocation":  "left",
    "axes.titlepad":       12,
    "xtick.color":         MUTED,
    "ytick.color":         MUTED,
    "legend.frameon":      False,
    "legend.fontsize":     9,
})


# Default contract (matches the dashboard defaults)
S0, K, r, sigma, T, m_per_year = 100.0, 100.0, 0.05, 0.20, 1.0, 252
m_eff = int(round(m_per_year * T))
N = 50_000

contract = OptionContract(
    S0=S0, K=K, r=r, sigma=sigma, T=T,
    option_type=OptionType.CALL,
    averaging=Averaging.ARITHMETIC,
    strike_type=StrikeType.FIXED,
    m=m_eff, q=0.0,
)


os.makedirs("visuals", exist_ok=True)


# --------------------------------------------------------------------------- #
#  1. Simulated paths
# --------------------------------------------------------------------------- #
def fig_paths():
    rng = np.random.default_rng(123)
    paths = simulate_gbm_paths(S0, r, sigma, T, m_eff, n_paths=40,
                                antithetic=False, rng=rng)
    full = np.hstack([np.full((40, 1), S0), paths])
    times = np.linspace(0, T, m_eff + 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i in range(40):
        ax.plot(times, full[i], color="#CBD5E1", linewidth=0.7, alpha=0.85)
    avg_path = full.mean(axis=0)
    ax.plot(times, avg_path, color=NAVY, linewidth=2.6, label="Average across paths")
    ax.axhline(K, color=AMBER, linewidth=1.4, linestyle="--", label=f"Strike K = {K:g}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Asset price")
    ax.set_title("Simulated Price Paths (40 sample paths)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("visuals/paths.png", dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
#  2. Convergence
# --------------------------------------------------------------------------- #
def fig_convergence():
    rng = np.random.default_rng(42)
    max_N = 100_000
    dt = T / m_eff
    drift = (r - 0.5 * sigma**2) * dt
    diff  = sigma * np.sqrt(dt)
    Z = rng.standard_normal((max_N, m_eff))
    log_ret = drift + diff * Z
    log_paths = np.cumsum(log_ret, axis=1)
    P = S0 * np.exp(log_paths)
    A = P.mean(axis=1)
    payoff = np.maximum(A - K, 0.0) * np.exp(-r * T)

    cum   = np.cumsum(payoff)
    cumsq = np.cumsum(payoff ** 2)
    Ng = np.arange(1, max_N + 1)
    mean = cum / Ng
    var  = np.maximum(cumsq / Ng - mean ** 2, 0.0)
    se   = np.sqrt(var / Ng)

    idx = np.unique(np.logspace(2, np.log10(max_N), 130).astype(int))
    idx = idx[idx < max_N]   # keep within array bounds
    Ns, m_arr, s_arr = Ng[idx], mean[idx], se[idx]

    fig, ax = plt.subplots(figsize=(9, 4.0))
    upper = m_arr + 1.96 * s_arr
    lower = m_arr - 1.96 * s_arr
    ax.fill_between(Ns, lower, upper, color=ACCENT, alpha=0.18,
                    label="95% confidence band")
    ax.plot(Ns, m_arr, color=ACCENT, linewidth=2.0, label="Running estimate")
    ax.axhline(m_arr[-1], color=GREEN, linewidth=1.3, linestyle="--",
                label=f"Converged: {m_arr[-1]:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Estimated price")
    ax.set_title("Monte Carlo convergence")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("visuals/convergence.png", dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
#  3. 2x2 sensitivity grid (σ, S, T, K)
# --------------------------------------------------------------------------- #
def fig_sensitivity_grid():
    def sweep(param_name, values, base=contract):
        prices = []
        for v in values:
            kwargs = {param_name: float(v)}
            c = base.with_(**kwargs)
            prices.append(price_mc_cv(c, n_paths=12_000, seed=42).price)
        return prices

    sigmas = np.linspace(0.05, 0.60, 22)
    spots  = np.linspace(0.55 * K, 1.45 * K, 22)
    Ts     = np.linspace(0.10, 3.0, 22)
    Ks     = np.linspace(0.55 * S0, 1.45 * S0, 22)

    print("  Sweeping σ ..."); ys_sig  = sweep("sigma", sigmas)
    print("  Sweeping S ..."); ys_S    = sweep("S0",    spots)
    print("  Sweeping T ..."); ys_T    = []
    for T_i in Ts:
        m_i = max(1, int(round(m_per_year * T_i)))
        c_i = contract.with_(T=float(T_i), m=m_i)
        ys_T.append(price_mc_cv(c_i, n_paths=12_000, seed=42).price)
    print("  Sweeping K ..."); ys_K    = sweep("K",     Ks)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0))
    panels = [
        (axes[0, 0], "Price vs Volatility (σ)", sigmas, ys_sig, sigma,
         "Volatility σ", "{:.0%}"),
        (axes[0, 1], "Price vs Spot (S₀)",      spots,  ys_S,   S0,
         "Spot S₀", "{:.0f}"),
        (axes[1, 0], "Price vs Maturity (T)",   Ts,     ys_T,   T,
         "Maturity T (years)", "{:.1f}"),
        (axes[1, 1], "Price vs Strike (K)",     Ks,     ys_K,   K,
         "Strike K", "{:.0f}"),
    ]
    for ax, title, xs, ys, cur, xlab, fmt in panels:
        ax.fill_between(xs, ys, color=NAVY, alpha=0.10)
        ax.plot(xs, ys, color=NAVY, linewidth=2.0)
        cur_idx = int(np.argmin(np.abs(np.array(xs) - cur)))
        ax.scatter([cur], [ys[cur_idx]], s=70, color=AMBER, zorder=5,
                    edgecolor="white", linewidth=1.5,
                    label=f"current ({fmt.format(cur)})")
        ax.axvline(cur, color=AMBER, linewidth=0.8, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Option price")
        ax.legend(loc="upper left", fontsize=8)
        if "σ" in xlab:
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    plt.tight_layout()
    plt.savefig("visuals/sensitivity_grid.png", dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
#  4. Spot × Vol heatmap
# --------------------------------------------------------------------------- #
def fig_heatmap():
    spots = np.linspace(0.70 * K, 1.30 * K, 13)
    sigmas = np.linspace(0.10, 0.50, 11)
    Z = np.zeros((len(sigmas), len(spots)))
    print("  Building heatmap ...")
    for i, sg in enumerate(sigmas):
        for j, sp in enumerate(spots):
            c = contract.with_(S0=float(sp), sigma=float(sg))
            Z[i, j] = price_mc_cv(c, n_paths=4_000, seed=42).price

    fig, ax = plt.subplots(figsize=(9, 5.0))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "navy_to_green",
        [LIGHT, "#BFDBFE", ACCENT, GREEN],
    )
    im = ax.imshow(
        Z, origin="lower", aspect="auto", cmap=cmap,
        extent=[spots.min(), spots.max(), sigmas.min(), sigmas.max()],
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Option price", color=MUTED, fontsize=10)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(GRID)

    ax.scatter([S0], [sigma], marker="x", s=140, color=AMBER,
                linewidths=3.0, label=f"Current (S₀={S0:g}, σ={sigma:.0%})")
    ax.set_xlabel("Spot S₀")
    ax.set_ylabel("Volatility σ")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Price as a function of Spot and Volatility")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("visuals/heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
#  5. Variance reduction comparison
# --------------------------------------------------------------------------- #
def fig_variance_reduction():
    print("  Pricing for VR chart ...")
    plain = price_mc(contract, n_paths=N, seed=42)
    cv    = price_mc_cv(contract, n_paths=N, seed=42)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Plain Monte Carlo", "MC + Geometric CV"]
    values = [plain.std_error, cv.std_error]
    colors = [MUTED, ACCENT]

    bars = ax.bar(labels, values, color=colors, width=0.55,
                   edgecolor="white", linewidth=1.5)
    for bar, v, lab in zip(bars, values, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.03,
                 f"SE = {v:.5f}",
                 ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Standard error (lower is better)")
    ax.set_title("Standard error: Plain Monte Carlo vs Control Variate")
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    plt.savefig("visuals/variance_reduction.png", dpi=130, bbox_inches="tight")
    plt.close()

    return plain, cv


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Generating report visuals (this takes ~30 s) ...")
    fig_paths()
    print("  paths.png OK")
    fig_convergence()
    print("  convergence.png OK")
    fig_sensitivity_grid()
    print("  sensitivity_grid.png OK")
    fig_heatmap()
    print("  heatmap.png OK")
    plain, cv = fig_variance_reduction()
    print("  variance_reduction.png OK")
    print()
    print(f"All visuals saved in ./visuals/")
    print(f"Headline numbers (used in report):")
    print(f"  Plain MC : price={plain.price:.4f}  SE={plain.std_error:.5f}  runtime={plain.runtime_ms:.0f} ms")
    print(f"  MC + CV  : price={cv.price:.4f}  SE={cv.std_error:.5f}  runtime={cv.runtime_ms:.0f} ms")
    print(f"  Variance reduction: {(plain.std_error/cv.std_error)**2:.0f}x")
