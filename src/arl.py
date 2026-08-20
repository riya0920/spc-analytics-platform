"""Average Run Length by simulation — the professional-grade way to compare charts.

ARL0 = expected number of points until a FALSE alarm on an in-control process.
       Want it long. A 3-sigma Shewhart chart on normal data has ARL0 = 1/0.0027
       = 370.4 by construction, and that is the number every other detector here
       is CALIBRATED to match, so the comparison means something.
ARL1 = expected number of points until detection of a real shift. Want it short.

Reporting ARL1 without ARL0 is how bake-offs get won: any chart detects faster if
you let it alarm more. The pair is the result; either alone is marketing.

The second number computed here is the one that gets skipped everywhere: what
STACKING RULES does to ARL0. Each added rule is another test on the same data, so
the in-control alarm rate compounds. The measured inflation is in RESULTS.md.

All simulations are batched (reps x max_n matrices) — the loop-per-replication
version was ~200x slower and made the calibration below impractical, which is
probably why so few portfolios contain one.
"""
from __future__ import annotations

import numpy as np

from rules import any_violation, cusum, ewma

MAX_N = 3000  # run-length censoring point


def _first_signal(viol: np.ndarray, max_n: int) -> np.ndarray:
    """First True index per row (1-based), or max_n if a row never signals."""
    any_ = viol.any(axis=-1)
    first = viol.argmax(axis=-1) + 1
    return np.where(any_, first, max_n)


def _mean_rl(viol: np.ndarray, max_n: int) -> float:
    rl = _first_signal(viol, max_n)
    censored = float(np.mean(rl >= max_n))
    if censored > 0.02:
        # Honest failure mode: with heavy censoring the mean is biased low and
        # should not be quoted as an ARL.
        return float("nan")
    return float(rl.mean())


def arl_shewhart(shift: float, which=(1,), reps: int = 2000, seed: int = 7,
                 max_n: int = MAX_N) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, max_n)) + shift
    return _mean_rl(any_violation(x, which), max_n)


def arl_ewma(shift: float, lam: float = 0.2, L: float = 2.86, reps: int = 2000,
             seed: int = 7, max_n: int = MAX_N) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, max_n)) + shift
    return _mean_rl(ewma(x, lam=lam, L=L, mu0=0.0, sigma=1.0)["violation"], max_n)


def arl_cusum(shift: float, k: float = 0.5, h: float = 4.77, reps: int = 2000,
              seed: int = 7, max_n: int = MAX_N) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, max_n)) + shift
    return _mean_rl(cusum(x, k=k, h=h, mu0=0.0, sigma=1.0)["violation"], max_n)


def calibrate(kind: str, target_arl0: float = 370.4, lo: float = 2.0, hi: float = 9.0,
              reps: int = 4000, tol: float = 0.03, iters: int = 14) -> float:
    """Bisect the control parameter until MEASURED ARL0 hits the target.

    Why calibrate rather than quote the textbook constant: the published pairing
    (lambda=0.2, L=2.962 -> ARL0=370) is derived for the ASYMPTOTIC EWMA limits.
    This implementation uses the exact time-varying limits, which are tighter over
    the first few dozen points and therefore alarm more often; measured ARL0 at
    L=2.962 comes out near 490, not 370 (measured: 487.5 at reps=6000). Quoting the constant anyway would have
    put EWMA and Shewhart on different false-alarm budgets and made the head-to-
    head meaningless. So the parameter is measured into place, and the calibrated
    value is reported next to the textbook one.
    """
    f = {"ewma": lambda p: arl_ewma(0.0, L=p, reps=reps),
         "cusum": lambda p: arl_cusum(0.0, h=p, reps=reps)}[kind]
    best = (float("inf"), (lo + hi) / 2)
    for _ in range(iters):
        mid = (lo + hi) / 2
        got = f(mid)
        if not np.isfinite(got):  # censored: parameter too loose
            hi = mid
            continue
        err = abs(got - target_arl0) / target_arl0
        if err < best[0]:
            best = (err, mid)
        if err < tol:
            return mid
        if got < target_arl0:
            lo = mid
        else:
            hi = mid
    return best[1]


def arl_table(shifts=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0), reps: int = 4000,
              ewma_L: float = 2.86, cusum_h: float = 4.77) -> list[dict]:
    rows = []
    for s in shifts:
        rows.append({
            "shift_sigma": s,
            "shewhart_rule1": arl_shewhart(s, which=(1,), reps=reps),
            "shewhart_rules1_4": arl_shewhart(s, which=(1, 2, 3, 4), reps=reps),
            "shewhart_all8": arl_shewhart(s, which=tuple(range(1, 9)), reps=reps),
            "ewma": arl_ewma(s, L=ewma_L, reps=reps),
            "cusum": arl_cusum(s, h=cusum_h, reps=reps),
        })
    return rows


def per_rule_performance(reps: int = 4000, n: int = 200, seed: int = 11,
                         shift: float = 0.0) -> list[dict]:
    """Per-rule firing probability over an n-point chart.

    Expressed as 'probability this rule fires at least once over a 200-point
    chart', which is the form a quality engineer actually experiences: not a
    per-point alpha, but how often the rule interrupts them in a month of charting.
    At shift=0 this is the false-alarm rate; at shift>0 it is the detection rate.
    """
    from rules import RULE_DOCS, apply_rules

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, n)) + shift
    res = apply_rules(x)
    rows = [{"rule": str(r), "description": RULE_DOCS[r],
             "p_fires": float(res[r].any(axis=-1).mean())} for r in sorted(res)]
    stacked = np.zeros((reps, n), dtype=bool)
    for v in res.values():
        stacked |= v
    rows.append({"rule": "1-8 stacked", "description": "all eight rules together",
                 "p_fires": float(stacked.any(axis=-1).mean())})
    rows.append({"rule": "1-4 stacked", "description": "classic Western Electric only",
                 "p_fires": float(np.logical_or.reduce([res[r] for r in (1, 2, 3, 4)]).any(axis=-1).mean())})
    return rows
