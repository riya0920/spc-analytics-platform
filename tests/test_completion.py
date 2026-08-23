"""Tests for the third-pass modules."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import limits as LIM  # noqa: E402


def _limset(sigma=1.0):
    return LIM.LimitSet(center=10.0, ucl=13.0, lcl=7.0, sigma_within=sigma,
                        n_subgroups=100, established_at="2026-01-01T00:00:00Z",
                        reason="initial phase I study")


# ---------------------------------------------------------------------------
# phase I
# ---------------------------------------------------------------------------

def test_phase_one_removes_outliers_and_refits():
    rng = np.random.default_rng(0)
    means = rng.normal(10, 0.1, 60)
    ranges = np.abs(rng.normal(0.25, 0.05, 60))
    means[7] = 14.0                      # an unmistakable excursion
    out = LIM.phase_one(means, ranges, n=5)
    assert 7 in sum((r["removed"] for r in out["removed_rounds"]), [])
    assert out["n_kept"] < out["n_total"]
    assert out["converged"]


def test_phase_one_is_iterative_not_single_pass():
    """A single pass leaves the disturbance inflating the limits it sits in."""
    rng = np.random.default_rng(1)
    means = rng.normal(10, 0.1, 80)
    ranges = np.abs(rng.normal(0.25, 0.05, 80))
    means[10:14] = [13.0, 12.4, 12.0, 11.6]
    out = LIM.phase_one(means, ranges, n=5)
    assert out["iterations"] >= 2, "one pass cannot peel a graded excursion"


def test_phase_one_refuses_to_call_a_shredded_study_usable():
    rng = np.random.default_rng(2)
    means = rng.normal(10, 3.0, 60)      # wildly unstable
    ranges = np.abs(rng.normal(0.25, 0.05, 60))
    out = LIM.phase_one(means, ranges, n=5, max_iterations=8)
    if out["fraction_removed"] >= 0.20:
        assert not out["usable"]


# ---------------------------------------------------------------------------
# limit revision
# ---------------------------------------------------------------------------

def test_alarming_is_never_a_reason_to_revise_limits():
    """The anti-trigger, and the reason this module exists."""
    out = LIM.propose_revision(_limset(), np.random.default_rng(0).normal(10, 2, 500),
                               reason="TOO_MANY_ALARMS", subgroup_size=5)
    assert not out["approved"]
    assert "never revised because the chart is alarming" in out["why"]


def test_a_brief_improvement_is_refused_for_duration():
    rng = np.random.default_rng(3)
    out = LIM.propose_revision(_limset(), rng.normal(10, 0.3, 8 * 5),
                               reason="SUSTAINED_IMPROVEMENT", subgroup_size=5)
    assert not out["approved"] and "subgroups" in out["why"]


def test_worse_variation_is_refused_even_with_the_right_reason():
    rng = np.random.default_rng(4)
    out = LIM.propose_revision(_limset(sigma=0.5), rng.normal(10, 2.0, 60 * 5),
                               reason="SUSTAINED_IMPROVEMENT", subgroup_size=5)
    assert not out["approved"] and "did not decrease" in out["why"]


def test_a_real_sustained_improvement_is_approved_and_audited():
    rng = np.random.default_rng(5)
    cur = _limset(sigma=1.0)
    vals = rng.normal(10, 0.4, 60 * 5)
    dec = LIM.propose_revision(cur, vals, reason="SUSTAINED_IMPROVEMENT",
                               subgroup_size=5)
    assert dec["approved"] and dec["p_value"] < 0.05
    new = LIM.apply_revision(cur, dec, vals, 5, "SUSTAINED_IMPROVEMENT")
    assert new.revision == 1
    assert new.sigma_within < cur.sigma_within
    assert new.history and new.history[0]["reason"] == cur.reason


def test_applying_an_unapproved_revision_raises():
    with pytest.raises(ValueError):
        LIM.apply_revision(_limset(), {"approved": False, "why": "no"},
                           np.zeros(100), 5, "PROCESS_CHANGE")


def test_rolling_limits_cannot_see_a_drift_they_walk_with():
    """The concrete form of 'a chart that adapts to the process is not a chart'."""
    rng = np.random.default_rng(6)
    v = rng.normal(10, 0.1, 300 * 5)
    out = LIM.rolling_limits_demo(v, window=25, drift_per_point=0.01,
                                  subgroup_size=5)
    assert out["frozen_limit_alarms"] > out["rolling_limit_alarms"]


# ---------------------------------------------------------------------------
# disposition workflow
# ---------------------------------------------------------------------------

def test_closing_without_a_cause_is_refused():
    """The rule that makes the assignable-cause Pareto exist at all."""
    q = LIM.DispositionQueue()
    e = q.raise_event(12, "rule1_beyond_3sigma", 13.4)
    q.assign(e, "eng-1")
    with pytest.raises(ValueError, match="assignable cause"):
        q.close(e, "continue")


def test_the_happy_path_closes_and_feeds_the_pareto():
    q = LIM.DispositionQueue()
    for i, cause in enumerate(["tool wear", "tool wear", "setup error"]):
        e = q.raise_event(i, "rule2_run_of_9", 11.0)
        q.assign(e, "eng-1")
        q.record_cause(e, cause, "check tool life")
        q.close(e, "flag")
    s = q.summary()
    assert s["by_state"]["CLOSED"] == 3
    assert s["cause_pareto"][0] == ("tool wear", 2)


def test_state_machine_rejects_out_of_order_transitions():
    q = LIM.DispositionQueue()
    e = q.raise_event(1, "rule1_beyond_3sigma", 20.0)
    q.assign(e, "eng-2")
    q.record_cause(e, "gauge drift", "re-measure")
    q.close(e, "quarantine")
    with pytest.raises(ValueError):
        q.assign(e, "eng-3")


def test_every_rule_has_an_action_plan_with_a_product_disposition():
    for rule in LIM.OOC_ACTIONS:
        plan = LIM.action_plan(rule)
        assert plan["actions"] and plan["product_disposition"] and plan["urgency"]


def test_rule1_quarantines_and_says_to_re_measure_first():
    """A gauge fault and a process fault look identical on the chart."""
    plan = LIM.action_plan("rule1_beyond_3sigma")
    assert plan["product_disposition"] == "quarantine"
    assert any("re-measure" in a.lower() for a in plan["actions"])


def test_unknown_rule_gets_a_conservative_default():
    plan = LIM.action_plan("rule_that_does_not_exist")
    assert plan["product_disposition"].startswith("continue")
