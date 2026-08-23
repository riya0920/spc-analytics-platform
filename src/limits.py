"""Phase I / phase II, limit revision, and the OOC disposition workflow.

===========================================================================
PHASE I vs PHASE II -- the distinction the earlier passes used and never named
===========================================================================

They are different statistical problems and people run them with the same code,
which is where most of the trouble comes from.

  PHASE I   RETROSPECTIVE. You have a batch of history and you are asking "was
            this process stable, and if so what are its limits?" You expect to
            find and remove out-of-control points, then refit. The false-alarm
            rate is not the thing being controlled -- you are *studying* the data.

  PHASE II  PROSPECTIVE. Limits are fixed, new points arrive, and each is
            judged against them. Now ARL0 matters, because every false alarm
            costs an investigation, and the limits must NOT move in response to
            the data they are judging.

The classic error is running phase II with limits recomputed on a rolling window.
It feels adaptive and it is self-defeating: a slow drift walks the limits along
with it and the chart never alarms. **A control chart that adapts to the process
is not a control chart** -- it is a smoother.

===========================================================================
WHEN LIMITS MAY BE REVISED
===========================================================================

Only for a REASON, and the reason is never "the chart is alarming a lot". Three
legitimate triggers, and the point of writing them down is that a policy nobody
wrote down becomes "whenever it is inconvenient":

  1. a deliberate, documented process change (new tool, new material, new
     method) -- limits from before the change describe a process that no longer
     exists
  2. the process genuinely IMPROVED and the improvement is sustained -- keeping
     wide limits means a chart that can no longer detect the process degrading
     back to where it was
  3. the original phase-I study was inadequate -- too few subgroups, or it
     included a disturbance nobody removed

And one anti-trigger, which is the whole reason this module exists: **limits are
never widened because the chart is alarming.** That is the process telling you
something and the response being to turn down the volume.

The improvement case is the interesting one, because it needs evidence that the
improvement is REAL and SUSTAINED rather than a lucky fortnight. A sigma-ratio
F-test plus a minimum duration is the cheap version of that, and both halves are
required: significance without duration finds noise, duration without
significance finds nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# phase I
# ---------------------------------------------------------------------------

def phase_one(subgroup_means: np.ndarray, subgroup_ranges: np.ndarray,
              n: int, *, max_iterations: int = 5,
              d2: float | None = None, A2: float | None = None,
              D3: float = 0.0, D4: float = 2.114) -> dict:
    """Iteratively remove out-of-control subgroups and refit, as phase I requires.

    The iteration is the part that is usually skipped. A single pass computes
    limits from data that includes the very disturbances the limits are supposed
    to exclude, so the limits come out too wide and then fail to flag those same
    points -- the disturbance hides itself by inflating the limits it sits in.

    Bounded at `max_iterations`, and it reports whether it converged. An
    unbounded loop on a genuinely unstable process removes points until almost
    nothing is left, and a "stable" process defined by having deleted 40% of its
    history is not a finding.
    """
    from constants import A2 as A2_fn, D3 as D3_fn, D4 as D4_fn, d2 as d2_fn
    A2 = A2 if A2 is not None else A2_fn(n)
    D3 = D3_fn(n)
    D4 = D4_fn(n)
    d2 = d2 if d2 is not None else d2_fn(n)

    keep = np.ones(len(subgroup_means), dtype=bool)
    removed_rounds = []
    converged = False
    for it in range(max_iterations):
        xbar = float(subgroup_means[keep].mean())
        rbar = float(subgroup_ranges[keep].mean())
        ucl_x, lcl_x = xbar + A2 * rbar, xbar - A2 * rbar
        ucl_r, lcl_r = D4 * rbar, D3 * rbar
        bad = ((subgroup_means > ucl_x) | (subgroup_means < lcl_x)
               | (subgroup_ranges > ucl_r) | (subgroup_ranges < lcl_r)) & keep
        if not bad.any():
            converged = True
            break
        removed_rounds.append({"iteration": it + 1,
                               "removed": np.flatnonzero(bad).tolist()})
        keep &= ~bad

    xbar = float(subgroup_means[keep].mean())
    rbar = float(subgroup_ranges[keep].mean())
    return {
        "converged": converged,
        "iterations": len(removed_rounds),
        "n_kept": int(keep.sum()), "n_total": int(len(keep)),
        "fraction_removed": 1 - keep.mean(),
        "removed_rounds": removed_rounds,
        "center": xbar, "rbar": rbar,
        "sigma_within": rbar / d2,
        "ucl_x": xbar + A2 * rbar, "lcl_x": xbar - A2 * rbar,
        "ucl_r": D4 * rbar, "lcl_r": D3 * rbar,
        # A phase-I study that had to delete a fifth of its own data has not
        # established stability; it has established that something happened.
        "usable": converged and (1 - keep.mean()) < 0.20,
    }


# ---------------------------------------------------------------------------
# limit revision
# ---------------------------------------------------------------------------

@dataclass
class LimitSet:
    """Frozen limits plus the audit trail of how they came to be."""
    center: float
    ucl: float
    lcl: float
    sigma_within: float
    n_subgroups: int
    established_at: str
    reason: str
    revision: int = 0
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"center": self.center, "ucl": self.ucl, "lcl": self.lcl,
                "sigma_within": self.sigma_within,
                "n_subgroups": self.n_subgroups, "revision": self.revision,
                "established_at": self.established_at, "reason": self.reason,
                "history": self.history}


VALID_REASONS = {
    "PROCESS_CHANGE": "a documented change to tool, material or method",
    "SUSTAINED_IMPROVEMENT": "variation genuinely reduced, and it held",
    "INADEQUATE_PHASE_I": "the original study was too small or contaminated",
}


def propose_revision(current: LimitSet, new_values: np.ndarray, *,
                     reason: str, subgroup_size: int,
                     min_subgroups: int = 25, alpha: float = 0.05) -> dict:
    """Decide whether limits may be revised. Refusal is the normal answer.

    For SUSTAINED_IMPROVEMENT both tests must pass, and requiring both is the
    design:

      significance  an F-test on the variance ratio, so a lucky fortnight does
                    not qualify
      duration      at least `min_subgroups`, so a real but brief improvement
                    does not either

    Either alone is easy to satisfy by accident. Together they are roughly the
    evidence a quality engineer would ask for before signing.
    """
    if reason not in VALID_REASONS:
        return {"approved": False,
                "why": (f"{reason!r} is not a valid trigger. Limits are never "
                        "revised because the chart is alarming -- that is the "
                        "process talking and the response being to turn down "
                        "the volume."),
                "valid_reasons": VALID_REASONS}

    v = np.asarray(new_values, dtype=float)
    n_sub = len(v) // subgroup_size
    if n_sub < min_subgroups:
        return {"approved": False,
                "why": f"only {n_sub} subgroups; {min_subgroups} required",
                "n_subgroups": n_sub}

    groups = v[: n_sub * subgroup_size].reshape(n_sub, subgroup_size)
    from constants import d2 as d2_fn
    rbar = float(np.ptp(groups, axis=1).mean())
    sigma_new = rbar / d2_fn(subgroup_size)

    out = {"n_subgroups": n_sub, "sigma_old": current.sigma_within,
           "sigma_new": sigma_new,
           "ratio": sigma_new / max(current.sigma_within, 1e-12)}

    if reason == "SUSTAINED_IMPROVEMENT":
        df = n_sub * (subgroup_size - 1)
        f = (current.sigma_within ** 2) / max(sigma_new ** 2, 1e-24)
        p = 1 - stats.f.cdf(f, df, df)
        out.update({"f_statistic": float(f), "p_value": float(p)})
        if sigma_new >= current.sigma_within:
            out.update({"approved": False,
                        "why": "variation did not decrease"})
            return out
        if p >= alpha:
            out.update({"approved": False,
                        "why": (f"reduction is not significant (p={p:.3f}); a "
                                "lucky fortnight is not an improvement")})
            return out

    out["approved"] = True
    out["why"] = f"{reason}: {VALID_REASONS[reason]}"
    return out


def apply_revision(current: LimitSet, decision: dict, new_values: np.ndarray,
                   subgroup_size: int, reason: str) -> LimitSet:
    if not decision.get("approved"):
        raise ValueError(f"revision not approved: {decision.get('why')}")
    from constants import A2 as A2_fn, d2 as d2_fn
    v = np.asarray(new_values, dtype=float)
    n_sub = len(v) // subgroup_size
    g = v[: n_sub * subgroup_size].reshape(n_sub, subgroup_size)
    xbar = float(g.mean())
    rbar = float(np.ptp(g, axis=1).mean())
    A2 = A2_fn(subgroup_size)
    hist = list(current.history) + [{
        "revision": current.revision, "center": current.center,
        "ucl": current.ucl, "lcl": current.lcl, "reason": current.reason,
        "retired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }]
    return LimitSet(center=xbar, ucl=xbar + A2 * rbar, lcl=xbar - A2 * rbar,
                    sigma_within=rbar / d2_fn(subgroup_size),
                    n_subgroups=n_sub,
                    established_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    reason=reason, revision=current.revision + 1, history=hist)


def rolling_limits_demo(values: np.ndarray, window: int, drift_per_point: float,
                        subgroup_size: int = 5) -> dict:
    """Show that recomputed limits cannot see a drift they are walking with.

    This is the concrete version of "a control chart that adapts to the process
    is not a control chart". A slow drift is injected, then judged twice: once
    against limits frozen in phase I, and once against limits recomputed on a
    trailing window.
    """
    from constants import A2 as A2_fn
    v = np.asarray(values, dtype=float).copy()
    n_sub = len(v) // subgroup_size
    g = v[: n_sub * subgroup_size].reshape(n_sub, subgroup_size)
    drift = np.arange(n_sub) * drift_per_point
    g = g + drift[:, None]
    means = g.mean(axis=1)
    ranges = np.ptp(g, axis=1)
    A2 = A2_fn(subgroup_size)

    base = slice(0, max(window, 25))
    cx, cr = float(means[base].mean()), float(ranges[base].mean())
    frozen_hits = int(((means > cx + A2 * cr) | (means < cx - A2 * cr)).sum())

    rolling_hits = 0
    for i in range(window, n_sub):
        w = slice(i - window, i)
        c, r = float(means[w].mean()), float(ranges[w].mean())
        if means[i] > c + A2 * r or means[i] < c - A2 * r:
            rolling_hits += 1
    return {"n_subgroups": n_sub, "window": window,
            "total_drift_in_sigma": float(drift[-1] / max(cr / 2.326, 1e-9)),
            "frozen_limit_alarms": frozen_hits,
            "rolling_limit_alarms": rolling_hits}


# ---------------------------------------------------------------------------
# OOC disposition
# ---------------------------------------------------------------------------

OOC_ACTIONS = {
    "rule1_beyond_3sigma": {
        "urgency": "stop and contain",
        "likely_causes": ["tool breakage", "material change", "setup error",
                          "measurement error"],
        "actions": ["quarantine product back to the last known-good subgroup",
                    "re-measure the subgroup before acting -- a gauge fault "
                    "looks exactly like a process fault",
                    "check for a setup or material change at that timestamp"],
        "product_disposition": "quarantine",
    },
    "rule2_run_of_9": {
        "urgency": "investigate this shift",
        "likely_causes": ["tool wear", "gradual temperature change",
                          "a new operator's method"],
        "actions": ["check tool life against the run", "check ambient trend",
                    "compare with the other shift"],
        "product_disposition": "continue, flag for review",
    },
    "rule3_trend_of_6": {
        "urgency": "investigate this shift",
        "likely_causes": ["progressive tool wear", "fixture loosening",
                          "reagent depletion"],
        "actions": ["project the trend to the spec limit and estimate hours "
                    "remaining", "schedule a tool change before it arrives"],
        "product_disposition": "continue, plan intervention",
    },
    "rule5_2of3_beyond_2sigma": {
        "urgency": "watch",
        "likely_causes": ["a shift beginning", "an intermittent disturbance"],
        "actions": ["increase sampling frequency until it resolves"],
        "product_disposition": "continue",
    },
}


@dataclass
class OOCEvent:
    subgroup: int
    rule: str
    value: float
    raised_at: str
    state: str = "OPEN"
    assignee: str | None = None
    cause: str | None = None
    action: str | None = None
    disposition: str | None = None
    closed_at: str | None = None


class DispositionQueue:
    """OOC events with a state machine, because a violation is a work item.

    The spec's phrase is that violations must be "investigated and
    dispositioned, not just displayed". The state machine is the difference:
    OPEN -> ASSIGNED -> CAUSE_FOUND -> CLOSED, and closing REQUIRES a cause and a
    product disposition.

    That requirement is the whole mechanism. Without it the queue silently
    becomes a list of things somebody clicked away, and the most valuable output
    of SPC -- a Pareto of assignable causes, which is what tells you where to
    spend engineering time -- is never produced, because nobody was ever forced
    to name one.
    """

    def __init__(self) -> None:
        self.events: list[OOCEvent] = []

    def raise_event(self, subgroup: int, rule: str, value: float) -> OOCEvent:
        e = OOCEvent(subgroup=subgroup, rule=rule, value=value,
                     raised_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.events.append(e)
        return e

    def assign(self, e: OOCEvent, who: str) -> OOCEvent:
        if e.state != "OPEN":
            raise ValueError(f"cannot assign an event in state {e.state}")
        e.state, e.assignee = "ASSIGNED", who
        return e

    def record_cause(self, e: OOCEvent, cause: str, action: str) -> OOCEvent:
        if e.state not in ("ASSIGNED", "OPEN"):
            raise ValueError(f"cannot record a cause in state {e.state}")
        e.state, e.cause, e.action = "CAUSE_FOUND", cause, action
        return e

    def close(self, e: OOCEvent, disposition: str) -> OOCEvent:
        if not e.cause:
            raise ValueError(
                "cannot close without an assignable cause -- an OOC queue that "
                "can be emptied without naming causes produces no Pareto, and "
                "the Pareto is the point")
        e.state, e.disposition = "CLOSED", disposition
        e.closed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return e

    def cause_pareto(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for e in self.events:
            if e.cause:
                counts[e.cause] = counts.get(e.cause, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def summary(self) -> dict:
        by_state: dict[str, int] = {}
        for e in self.events:
            by_state[e.state] = by_state.get(e.state, 0) + 1
        return {"n": len(self.events), "by_state": by_state,
                "open": by_state.get("OPEN", 0) + by_state.get("ASSIGNED", 0),
                "cause_pareto": self.cause_pareto()}


def action_plan(rule: str) -> dict:
    return OOC_ACTIONS.get(rule, {
        "urgency": "investigate", "likely_causes": ["unknown"],
        "actions": ["characterise the signal before acting"],
        "product_disposition": "continue, flag for review"})
