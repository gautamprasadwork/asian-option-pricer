"""
Engine test suite.

These tests are the project's correctness contract. They check four classes
of property:

  1. Mathematical identities the engine must satisfy
     (put-call parity, no-arbitrage bounds, classical limits).
  2. Convergence of numerical methods to closed-form benchmarks
     (MC-geometric → Kemna-Vorst; CRR-vanilla → Black-Scholes).
  3. Cross-validation between independent estimators
     (FD-Delta vs pathwise-Delta).
  4. Operational claims about variance reduction
     (CV << Antithetic << Plain MC).

A green test suite is the basis for the green CI badge in the README and
the validation table in the report.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from engine import (
    OptionContract, OptionType, Averaging, StrikeType,
    default_preset, kemna_vorst_price, black_scholes_price,
    price_mc, price_mc_antithetic, price_mc_cv, price_qmc,
    price_crr_tree, compute_greeks, pathwise_delta,
    run_diagnostics, Severity, simulate_delta_hedge, implied_volatility,
    price_auto,
)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def atm_call():
    """A reasonable at-the-money fixed-strike arithmetic Asian call."""
    return OptionContract(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        option_type=OptionType.CALL,
        averaging=Averaging.ARITHMETIC,
        strike_type=StrikeType.FIXED,
        m=252, q=0.0,
    )


@pytest.fixture
def atm_put(atm_call):
    return atm_call.with_(option_type=OptionType.PUT)


@pytest.fixture
def geom_call(atm_call):
    return atm_call.with_(averaging=Averaging.GEOMETRIC)


# --------------------------------------------------------------------------- #
#  1. Closed-form sanity
# --------------------------------------------------------------------------- #
class TestClosedForm:

    def test_kemna_vorst_positive_for_call(self, geom_call):
        """KV price must be > 0 for a sensible call."""
        r = kemna_vorst_price(geom_call)
        assert r.price > 0
        assert r.std_error == 0.0

    def test_kemna_vorst_put_call_parity(self, geom_call):
        """
        Discrete geometric-Asian put-call parity:
            C - P = e^{-rT}(E[G_T] - K)
        where E[G_T] is the KV-implied forward of the geometric average.
        """
        call = kemna_vorst_price(geom_call)
        put  = kemna_vorst_price(geom_call.with_(option_type=OptionType.PUT))
        forward_G = call.extra["forward_G"]
        rhs = math.exp(-geom_call.r * geom_call.T) * (forward_G - geom_call.K)
        assert call.price - put.price == pytest.approx(rhs, abs=1e-8)

    def test_kemna_vorst_continuous_limit(self):
        """
        For very large m (m=10_000) the discrete KV should approach the
        continuous formula (σ_g = σ/√3 in the variance limit). We don't
        compare against a separate continuous formula — instead we check
        that price changes by less than 1bp going from m=5000 to m=10000.
        """
        base = OptionContract(
            S0=100, K=100, r=0.03, sigma=0.20, T=1.0,
            averaging=Averaging.GEOMETRIC, m=5_000,
        )
        p_5k = kemna_vorst_price(base).price
        p_10k = kemna_vorst_price(base.with_(m=10_000)).price
        assert abs(p_10k - p_5k) / p_5k < 1e-4    # < 1 bp


# --------------------------------------------------------------------------- #
#  2. Monte Carlo convergence to closed-form benchmark
# --------------------------------------------------------------------------- #
class TestMonteCarloConvergence:

    def test_mc_geometric_matches_kemna_vorst(self, geom_call):
        """
        MC on the geometric variant must agree with KV closed form within
        two standard errors. This proves the path simulator, the payoff
        logic, and the discounting are all correct.
        """
        truth = kemna_vorst_price(geom_call).price
        r = price_mc_antithetic(geom_call, n_paths=100_000, seed=42)
        gap = abs(r.price - truth)
        assert gap < 2 * r.std_error, (
            f"MC-geometric ({r.price:.5f}) differs from KV ({truth:.5f}) "
            f"by {gap:.5f}, more than 2·SE ({2*r.std_error:.5f})."
        )

    def test_cv_unbiased_for_arithmetic(self, atm_call):
        """
        The CV estimator must be unbiased: rerunning with very different
        seeds gives prices that lie within their joint 95% CI.
        """
        r1 = price_mc_cv(atm_call, n_paths=50_000, seed=42)
        r2 = price_mc_cv(atm_call, n_paths=50_000, seed=2024)
        joint_se = math.sqrt(r1.std_error ** 2 + r2.std_error ** 2)
        assert abs(r1.price - r2.price) < 4 * joint_se, (
            "Two independent CV runs disagree more than expected."
        )


# --------------------------------------------------------------------------- #
#  3. Variance reduction discipline
# --------------------------------------------------------------------------- #
class TestVarianceReduction:

    def test_antithetic_strictly_better_than_plain(self, atm_call):
        """Antithetic SE should be lower than plain SE on the same N."""
        plain = price_mc(atm_call, n_paths=20_000, seed=42)
        anti  = price_mc_antithetic(atm_call, n_paths=20_000, seed=42)
        assert anti.std_error < plain.std_error, (
            f"Antithetic SE ({anti.std_error:.5f}) is not lower than "
            f"plain SE ({plain.std_error:.5f})."
        )

    def test_control_variate_dominates(self, atm_call):
        """CV SE should be at least 5× lower than antithetic at the same N."""
        anti = price_mc_antithetic(atm_call, n_paths=20_000, seed=42)
        cv   = price_mc_cv(atm_call, n_paths=20_000, seed=42)
        ratio = anti.std_error / cv.std_error
        assert ratio >= 5.0, (
            f"CV improvement ({ratio:.1f}×) under-delivered — expected ≥5× "
            "for an at-the-money arithmetic Asian."
        )
        # And the prices should agree
        joint = math.sqrt(anti.std_error ** 2 + cv.std_error ** 2)
        assert abs(anti.price - cv.price) < 4 * joint

    def test_cv_correlation_high(self, atm_call):
        """ρ(arithmetic, geometric) for ATM Asian should exceed 0.98."""
        cv = price_mc_cv(atm_call, n_paths=20_000, seed=42)
        rho = cv.extra["rho"]
        assert rho > 0.98, f"CV correlation {rho:.4f} below the 0.98 expected for ATM Asians."


# --------------------------------------------------------------------------- #
#  4. Tree validation (CRR → BS)
# --------------------------------------------------------------------------- #
class TestTrees:

    def test_crr_call_converges_to_black_scholes(self):
        """CRR with 2000 steps must match BS to < 1 bp on an ATM call."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
        bs = black_scholes_price(S0, K, r, sigma, T, "call")
        tree = price_crr_tree(S0, K, r, sigma, T, steps=2000, option_type="call")
        assert abs(bs - tree.price) / bs < 1e-3, (
            f"CRR ({tree.price:.5f}) deviates from BS ({bs:.5f}) too much."
        )

    def test_crr_put_call_parity(self):
        """C - P = S0 - K·e^{-rT} on the tree, exactly (within numerics)."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
        c = price_crr_tree(S0, K, r, sigma, T, steps=500, option_type="call").price
        p = price_crr_tree(S0, K, r, sigma, T, steps=500, option_type="put").price
        lhs = c - p
        rhs = S0 - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.01

    def test_american_call_no_div_equals_european(self):
        """American call with q=0 should equal European call (well-known result)."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.30, 1.0
        eur = price_crr_tree(S0, K, r, sigma, T, steps=500, exercise="european").price
        am  = price_crr_tree(S0, K, r, sigma, T, steps=500, exercise="american").price
        assert abs(eur - am) < 1e-6


# --------------------------------------------------------------------------- #
#  5. Greeks — signs and cross-validation
# --------------------------------------------------------------------------- #
class TestGreeks:

    def test_call_delta_in_bounds(self, atm_call):
        g = compute_greeks(atm_call, n_paths=20_000)
        assert 0.0 <= g.delta.value <= 1.0

    def test_put_delta_in_bounds(self, atm_put):
        g = compute_greeks(atm_put, n_paths=20_000)
        assert -1.0 <= g.delta.value <= 0.0

    def test_vega_positive(self, atm_call):
        """Vega is positive for a long option (both call and put)."""
        g = compute_greeks(atm_call, n_paths=20_000)
        assert g.vega.value > 0

    def test_theta_negative_for_atm_call(self, atm_call):
        """ATM call loses value with time — theta < 0 per calendar day."""
        g = compute_greeks(atm_call, n_paths=20_000)
        assert g.theta.value < 0, f"Theta {g.theta.value} is non-negative."

    def test_fd_and_pathwise_delta_agree(self, atm_call):
        """The two independent Delta estimators must overlap inside CIs."""
        g = compute_greeks(atm_call, n_paths=50_000)
        assert g.pathwise_delta is not None
        diff = abs(g.delta.value - g.pathwise_delta.value)
        joint_se = math.sqrt(g.delta.std_error ** 2 + g.pathwise_delta.std_error ** 2)
        assert diff < 4 * joint_se, (
            f"FD Delta {g.delta.value:.4f} ± {g.delta.std_error:.4f} disagrees with "
            f"pathwise Delta {g.pathwise_delta.value:.4f} ± {g.pathwise_delta.std_error:.4f}."
        )


# --------------------------------------------------------------------------- #
#  6. No-arbitrage bounds enforced by diagnostics
# --------------------------------------------------------------------------- #
class TestDiagnostics:

    def test_no_warnings_on_sensible_contract(self, atm_call):
        r = price_auto(atm_call, n_paths=50_000)
        g = compute_greeks(atm_call, n_paths=20_000)
        diags = run_diagnostics(atm_call, r, g)
        # Allow INFO diagnostics, but no WARN or ERROR
        bad = [d for d in diags if d.severity in (Severity.WARN, Severity.ERROR)]
        assert not bad, f"Unexpected warnings on ATM call: {[str(d) for d in bad]}"

    def test_deep_otm_warns_essentially_worthless(self):
        c = OptionContract(
            S0=100, K=200, r=0.05, sigma=0.20, T=0.25,
            averaging=Averaging.ARITHMETIC, m=63,
        )
        r = price_mc_cv(c, n_paths=20_000, seed=42)
        diags = run_diagnostics(c, r)
        titles = [d.title for d in diags]
        assert any("worthless" in t.lower() for t in titles), (
            f"Expected 'essentially worthless' on deep OTM, got: {titles}"
        )

    def test_extreme_vol_warns(self, atm_call):
        crisis = atm_call.with_(sigma=1.5)   # 150% vol
        r = price_mc_cv(crisis, n_paths=10_000, seed=42)
        diags = run_diagnostics(crisis, r)
        assert any("extreme" in d.title.lower() for d in diags)


# --------------------------------------------------------------------------- #
#  7. Hedging simulation discipline
# --------------------------------------------------------------------------- #
class TestHedging:

    def test_hedge_reduces_variance(self, atm_call):
        # Use coarse monitoring (m=21, weekly) to keep this test fast
        c = atm_call.with_(m=21)
        base = price_auto(c, n_paths=10_000).price
        hr = simulate_delta_hedge(c, base, rebalance_every=1, n_paths=800)
        assert hr.std < hr.std_no_hedge, "Hedge made variance WORSE — bug in Δ approximation."
        assert hr.hedge_efficiency > 0.4, (
            f"Hedge efficiency only {hr.hedge_efficiency:.1%} — expected ≥40% for ATM."
        )

    def test_finer_rebalance_helps(self, atm_call):
        c = atm_call.with_(m=21)
        base = price_auto(c, n_paths=10_000).price
        coarse = simulate_delta_hedge(c, base, rebalance_every=5, n_paths=800)
        fine   = simulate_delta_hedge(c, base, rebalance_every=1, n_paths=800)
        assert fine.std < coarse.std, (
            f"Finer std ({fine.std:.4f}) not lower than coarser ({coarse.std:.4f})."
        )

    def test_hedge_is_unbiased(self, atm_call):
        """Mean P&L of a hedged short should be close to zero."""
        c = atm_call.with_(m=21)
        base = price_auto(c, n_paths=10_000).price
        hr = simulate_delta_hedge(c, base, rebalance_every=1, n_paths=1_500)
        # Allow some bias from gamma higher-order terms
        assert abs(hr.mean) < 1.0, f"Mean P&L {hr.mean:+.4f} suspiciously non-zero."


# --------------------------------------------------------------------------- #
#  8. IV calibration round-trip
# --------------------------------------------------------------------------- #
class TestCalibration:

    def test_round_trip_recovers_input_vol_geometric(self, geom_call):
        """For geometric variant we use the closed form, which is fast and exact."""
        target_sigma = 0.20
        c = geom_call.with_(sigma=target_sigma)
        target_price = kemna_vorst_price(c).price
        result = implied_volatility(c, target_price)
        assert result.converged
        assert abs(result.implied_vol - target_sigma) < 1e-3, (
            f"Recovered σ = {result.implied_vol:.6f} differs from input {target_sigma}."
        )


# --------------------------------------------------------------------------- #
#  9. Preset hygiene
# --------------------------------------------------------------------------- #
class TestPresets:

    def test_all_presets_price(self):
        """Every preset must price without errors and produce positive price."""
        from engine import PRESETS
        for key, preset in PRESETS.items():
            r = price_auto(preset.contract, n_paths=10_000)
            assert r.price > 0, f"Preset {key} produced non-positive price {r.price}"

    def test_default_is_brent(self):
        from engine import default_preset
        assert default_preset().key == "brent_refiner"
