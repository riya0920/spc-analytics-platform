"""Tests for the parts of DATA-2 that everything else rests on."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import capability  # noqa: E402
import charts  # noqa: E402
import constants as k  # noqa: E402
import gauge_rr  # noqa: E402
import generate  # noqa: E402
import rules  # noqa: E402

# Montgomery, Introduction to Statistical Quality Control, Appendix VI.
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
      9: 2.970, 10: 3.078}
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373,
      9: 0.337, 10: 0.308}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864,
      9: 1.816, 10: 1.777}
C4 = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594,
      8: 0.9650, 9: 0.9693, 10: 0.9727}


@pytest.mark.parametrize("n", sorted(D2))
def test_d2_matches_published(n):
    assert k.d2(n) == pytest.approx(D2[n], abs=5e-4)


@pytest.mark.parametrize("n", sorted(C4))
def test_c4_matches_published(n):
    assert k.c4(n) == pytest.approx(C4[n], abs=5e-5)


@pytest.mark.parametrize("n", sorted(A2))
def test_a2_and_d4_match_published(n):
    assert k.A2(n) == pytest.approx(A2[n], abs=1e-3)
    assert k.D4(n) == pytest.approx(D4[n], abs=1e-3)


def test_within_subgroup_limits_are_tighter_than_overall_sigma():
    """The core claim of the project, as a test rather than an assertion."""
    ch = generate.Characteristic(
        name="t_shift", target=10.0, sigma_process=0.1, sigma_gauge=0.0,
        lsl=9.5, usl=10.5, n_subgroups=200,
        disturbances=[generate.Disturbance("shift", 100, 200, 3.0, "big shift")])
    meas, _ = generate.simulate(ch)
    xchart, _ = charts.xbar_r(meas, baseline=slice(0, 80))
    _, ucl_n, lcl_n = charts.naive_limits(meas["value"].to_numpy())
    correct_width = xchart.ucl - xchart.lcl
    naive_width = (ucl_n - lcl_n) / np.sqrt(ch.subgroup_size)
    assert naive_width > correct_width


def test_rule1_fires_only_beyond_three_sigma():
    z = np.array([0.0, 2.9, -2.9, 3.1, -3.1])
    r = rules.apply_rules(z, (1,))[1]
    assert list(np.flatnonzero(r)) == [3, 4]


def test_rule4_needs_eight_consecutive_on_one_side():
    z = np.concatenate([np.full(7, 0.5), [-0.1], np.full(8, 0.5)])
    r = rules.apply_rules(z, (4,))[4]
    assert not r[:8].any()          # 7 then a crossing: no violation
    assert r[15]                    # 8 consecutive after the crossing


def test_batched_rules_match_single_chart():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 120))
    batch = rules.apply_rules(x)
    for i in range(6):
        single = rules.apply_rules(x[i])
        for rule in range(1, 9):
            assert np.array_equal(batch[rule][i], single[rule])


def test_ewma_and_cusum_batch_match_single():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 200))
    e, c = rules.ewma(x, mu0=0, sigma=1), rules.cusum(x, mu0=0, sigma=1)
    for i in range(4):
        assert np.allclose(e["z"][i], rules.ewma(x[i], mu0=0, sigma=1)["z"])
        assert np.allclose(c["c_plus"][i], rules.cusum(x[i], mu0=0, sigma=1)["c_plus"])


def test_capability_refuses_on_unstable_process():
    v = np.concatenate([np.random.default_rng(1).normal(10, 0.1, 100),
                        np.random.default_rng(2).normal(11, 0.1, 100)])
    with pytest.raises(capability.NotInControl):
        capability.assess(v, 0.1, 9.5, 10.5, in_control=False,
                          out_of_control_points=12)


def test_cpk_exceeds_ppk_when_the_process_drifts():
    rng = np.random.default_rng(7)
    # Tight within subgroups, drifting between them.
    parts = [rng.normal(10 + 0.4 * i, 0.05, 25) for i in range(8)]
    v = np.concatenate(parts)
    res = capability.normal_capability(v, sigma_within=0.05, lsl=8.0, usl=13.0)
    assert res["Cpk"] > res["Ppk"]
    assert res["Cpk_over_Ppk"] > 1.5


def test_gauge_rr_recovers_known_variance_components():
    """Averaged over seeds, the ANOVA estimates should be unbiased."""
    ev, pv = [], []
    for s in range(25):
        d = gauge_rr.simulate_study(sigma_part=1.0, sigma_repeat=0.05,
                                    sigma_operator=0.03, sigma_interaction=0.01,
                                    seed=s)
        r = gauge_rr.anova_grr(d)
        ev.append(r["EV_repeatability"])
        pv.append(r["PV_part"])
    assert np.mean(ev) == pytest.approx(0.05, abs=0.012)
    assert np.mean(pv) == pytest.approx(1.0, abs=0.12)


def test_gauge_rr_flags_an_instrument_noisier_than_the_process():
    d = gauge_rr.simulate_study(sigma_part=0.04, sigma_repeat=0.075,
                                sigma_operator=0.03, sigma_interaction=0.015)
    r = gauge_rr.anova_grr(d)
    assert r["verdict_AIAG"] == "UNACCEPTABLE"
    assert r["ndc"] < 2.0
