"""Tests for the attribute charts and the chart-selection consequences."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import attribute as A  # noqa: E402


def test_p_chart_limits_vary_with_subgroup_size():
    """Drawing one average limit makes small subgroups look in control and large
    ones look out of control, purely as an artifact of n."""
    # p ~ 0.1 so the LOWER limit stays above zero at every n. With a small p the
    # LCL clamps at 0 for the small subgroups, which truncates their width and
    # hides the very scaling this test is checking.
    n = np.array([100, 400, 900, 1600])
    d = (0.1 * n).astype(int)
    ch = A.p_chart(d, n)
    widths = ch["ucl"] - ch["lcl"]
    assert ch["variable_limits"]
    assert (ch["lcl"] > 0).all(), "clamped LCL would invalidate the ratio below"
    # Standard error scales as 1/sqrt(n), so a 16x size ratio is a 4x width ratio.
    assert widths[0] == pytest.approx(widths[-1] * 4.0, rel=0.02)


def test_p_chart_baseline_excludes_the_disturbed_period():
    """The bug this covers: computing pbar over ALL data drags the centre line
    toward the disturbance and then flags the healthy points on the other side.
    It produced 16 false alarms in 140 in-control points."""
    sim = A.simulate_p(n_points=200, p0=0.03, shift_at=140, shift_to=0.075)
    contaminated = A.p_chart(sim["defectives"], sim["sizes"])
    clean = A.p_chart(sim["defectives"], sim["sizes"], baseline=slice(0, 120))
    assert clean["center"] < contaminated["center"], (
        "a baseline-scoped centre line must not absorb the shift")
    pre_clean = int(A.violations(clean)[:140].sum())
    pre_dirty = int(A.violations(contaminated)[:140].sum())
    assert pre_clean < pre_dirty
    assert pre_clean <= 2, pre_clean


def test_every_attribute_chart_detects_its_planted_shift():
    base = slice(0, 120)
    p = A.simulate_p()
    assert A.violations(A.p_chart(p["defectives"], p["sizes"], base))[p["shift_at"]:].any()
    c = A.simulate_c()
    assert A.violations(A.c_chart(c["counts"], base))[c["shift_at"]:].any()
    u = A.simulate_u()
    assert A.violations(A.u_chart(u["counts"], u["areas"], base))[u["shift_at"]:].any()


def test_c_chart_on_varying_area_over_alarms_against_u_chart():
    """The cost of picking the wrong chart: a c chart assumes constant opportunity,
    so it flags points whose only sin is a larger inspection area."""
    u = A.simulate_u(area_range=(0.5, 4.0))
    pen = A.wrong_chart_penalty(u["counts"], u["areas"])
    assert pen["c_chart_violations"] > pen["u_chart_violations"]


def test_c_and_u_agree_when_the_area_is_constant():
    """The two charts must coincide in the regime where the c chart's assumption
    actually holds -- otherwise the comparison above proves nothing."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(8.0, 200)
    areas = np.ones(200)
    base = slice(0, 120)
    vc = A.violations(A.c_chart(counts, base))
    vu = A.violations(A.u_chart(counts, areas, base))
    assert int(vc.sum()) == pytest.approx(int(vu.sum()), abs=2)


def test_lower_limits_are_never_negative():
    counts = np.array([0, 1, 0, 2, 1])
    assert (A.c_chart(counts)["lcl"] >= 0).all()
    assert (A.p_chart(np.array([0, 1]), np.array([50, 50]))["lcl"] >= 0).all()
