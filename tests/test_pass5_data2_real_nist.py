"""Pass 5: NIST's real measurement data, and NIST's answers as the reference.

The point of these tests is that they can fail for a reason other than "the code
changed": three of them compare this project's arithmetic to numbers published by
somebody else, so they also fail if the arithmetic drifts toward something that
merely agrees with itself.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


RS = _load_module("rspc", ROOT / "run_real_spc.py")
FN = _load_module("fnist", ROOT / "fetch_nist.py")

NPZ = ROOT / "data" / "NIST" / "nist.npz"
RESULT = ROOT / "out" / "real_spc.json"
HAVE = NPZ.exists()


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def test_the_parser_finds_data_by_shape_not_by_a_hard_coded_skip():
    """NIST's own reader says SKIP 50. A hard-coded skip breaks silently the
    day the prose header gains a line."""
    text = ("This is the data file MPC62.DAT.\n"
            "Source: 25 days of measurements\n"
            "Col 1: a\nCol 2: b\nCol 3: c\n"
            "1 2 3\n4 5 6\n")
    a = FN._parse(text, 3, "toy")
    assert a.shape == (2, 3)
    assert a[0].tolist() == [1.0, 2.0, 3.0]


def test_a_prose_line_that_starts_with_a_number_is_not_data():
    text = "2 columns are described below\n1 2\n3 4\n"
    a = FN._parse(text, 2, "toy")
    assert a.shape == (2, 2)


def test_a_file_with_no_data_rows_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        FN._parse("just prose here\nand more prose\n", 4, "toy")


def test_the_published_values_are_recorded_verbatim():
    """These are NIST's numbers. If they are ever edited to match something this
    project computed, the reference stops being a reference."""
    p = FN.NIST_PUBLISHED
    assert p["s1_repeatability"] == 0.06139
    assert p["s2_level2"] == 0.02680
    assert p["s_chart_ucl"] == 0.09238
    assert p["J_repetitions"] == 6 and p["K_days"] == 25
    assert "e-Handbook" in p["source"]


# ---------------------------------------------------------------------------
# reproducing NIST
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_this_project_reproduces_nists_variance_decomposition():
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    assert n["s1"] == pytest.approx(0.06139, abs=5e-5)
    assert n["s2"] == pytest.approx(0.02680, abs=5e-5)
    assert n["s1_df"] == 125 and n["s2_df"] == 24


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_pooling_standard_deviations_is_not_averaging_them():
    """The classic way to get repeatability slightly and confidently wrong."""
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    assert n["s1"] > n["s1_if_you_average_sds_instead"]
    assert n["s1_if_you_average_sds_instead"] != pytest.approx(0.06139, abs=5e-5)


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_ucl_discrepancy_is_in_the_reference_and_changes_nothing():
    """NIST's printed UCL does not follow from NIST's printed s1 and F. It is
    reported rather than absorbed by a wider tolerance, and what makes that safe
    is that both limits flag the same days."""
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    assert n["ucl"] != pytest.approx(n["ucl_nist_states"], abs=1e-4)
    assert n["f_implied_by_nists_ucl"] == pytest.approx(2.2644, abs=1e-3)
    assert n["days_exceeding_ucl"] == n["days_exceeding_nists_ucl"] == 2
    assert n["same_days_either_way"] is True


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_agreement_check_is_not_a_tolerance_that_passes_anything():
    """A check that accepts everything is not a check. Perturb a computed value
    by 1% and the comparison must fail."""
    d = RS._load()
    pub = dict(d["published"])
    pub["s1_repeatability"] = 0.06139 * 1.01
    n = RS.reproduce_nist(d["MPC62"], pub)
    s1_row = next(r for r in n["checks"] if "s1" in r["quantity"])
    assert s1_row["agrees"] is False


# ---------------------------------------------------------------------------
# the scale trap
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_three_sigmas_are_different_and_ordered():
    """s1 is the SD of one measurement; the plotted values are means of six.
    Using s1 as sigma_within against them mixes scales and looks fine."""
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    o = RS.own_machinery(d["MPC62"], n)
    assert o["sigma_daily_mean"] < o["sigma_within"] < o["sigma_single_measurement"]
    assert o["sigma_single_measurement"] == pytest.approx(
        np.hypot(n["s1"], n["s2"]))


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_wrong_scale_gives_a_different_and_plausible_answer():
    """Which is why it is dangerous: it does not error, it flatters."""
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    o = RS.own_machinery(d["MPC62"], n)
    right, wrong = o["capability"], o["capability_at_wrong_scale"]
    assert not right.get("refused") and not wrong.get("refused")
    assert wrong["Cpk"] > right["Cpk"]
    assert 0.0 < right["Cpk"] < 10.0 and 0.0 < wrong["Cpk"] < 10.0


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_stability_gate_is_given_the_real_answer():
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    o = RS.own_machinery(d["MPC62"], n)
    assert o["in_control"] is True
    assert o["points_beyond_limits"] == 0
    assert o["capability"]["refused"] is False


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_real_check_standard_data_is_normal():
    """The synthetic study needed a transformation to pass this. Real data
    should not need one, and if it suddenly does the generator has drifted."""
    d = RS._load()
    n = RS.reproduce_nist(d["MPC62"], d["published"])
    o = RS.own_machinery(d["MPC62"], n)
    assert o["normality"]["normal_at_5pct"] is True


# ---------------------------------------------------------------------------
# the gauge study
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_an_unbalanced_gauge_table_is_truncated_rather_than_pushed_through():
    """A balanced expected-mean-squares decomposition on unequal cells gives
    variance components that are simply wrong, silently."""
    d = RS._load()
    g = RS.gauge_study(d["MPC61"])
    if g["balanced"]:
        pytest.skip("this copy of MPC61 is balanced")
    assert "truncated_to_reps" in g
    assert g["truncated_to_reps"] == min(g["cell_sizes"])


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_the_grr_headline_and_the_probe_bias_disagree():
    """The finding. %GRR is a ratio to PART variation, so a real systematic
    offset between probes can hide inside an excellent-looking number — and
    NIST's own conclusion is that these probes are not equivalent."""
    d = RS._load()
    g = RS.gauge_study(d["MPC61"])
    assert g["grr"]["pct_GRR_of_TV"] < 10.0, "AIAG would call this acceptable"
    assert g["probe_bias_anova"]["significant_at_5pct"] is True
    assert g["probe_bias_anova"]["p"] < 1e-4


@pytest.mark.skipif(not HAVE, reason="run fetch_nist.py first")
def test_probe_bias_is_measured_on_wafer_centred_values():
    """Otherwise wafer-to-wafer variation masquerades as probe bias and the
    ANOVA finds an effect on any dataset at all."""
    src = (ROOT / "run_real_spc.py").read_text(encoding="utf-8")
    assert 'groupby("part")["value"].transform("mean")' in src


# ---------------------------------------------------------------------------
# the written report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULT.exists(), reason="run run_real_spc.py first")
def test_the_report_states_the_discrepancy_rather_than_hiding_it():
    doc = (ROOT / "docs" / "REAL_SPC.md").read_text(encoding="utf-8")
    assert "discrepancy in the reference" in doc
    assert "0.09238" in doc
    assert "scale trap" in doc
    assert "hides a real bias" in doc
