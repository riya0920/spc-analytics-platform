"""Attribute control charts: p, np, c, u -- and why choosing between them matters.

The first build shipped `p_chart` as a stub and exercised none of them. That is a
real gap, because attribute data is what most plants actually have: a count of
defectives off an inspection station is free, while a measured dimension needs a
gauge, an operator and a study.

THE CHART SELECTION TREE, which is the actual skill:

    Counting DEFECTIVE UNITS (a part is good or bad)     -> binomial
        subgroup size varies    -> p chart   (fraction defective)
        subgroup size constant  -> np chart  (number defective)

    Counting DEFECTS (one unit can have several)         -> Poisson
        area of opportunity varies   -> u chart  (defects per unit)
        area of opportunity constant -> c chart  (count of defects)

Getting this wrong is not cosmetic. A c chart on data where the inspection area
varies has limits computed for the wrong opportunity, so it alarms whenever the
batch size changes and the operator learns that the chart tracks production
volume. The distinction between a DEFECTIVE (a unit) and a DEFECT (a flaw, of
which a unit may have several) is the one that decides the whole branch, and it is
the question to ask first.

VARIABLE LIMITS. For p and u charts with varying subgroup size the limits are
computed PER POINT, because the standard error depends on n. Drawing a single
average limit -- which is what a spreadsheet does -- makes small subgroups look
in control and large ones look out of control, purely as an artifact of n.
"""
from __future__ import annotations

import numpy as np


def p_chart(defectives: np.ndarray, sizes: np.ndarray,
            baseline: slice | None = None) -> dict:
    """Fraction defective, VARIABLE limits (one pair per point).

    `baseline` is the PHASE I window the centre line is estimated from. Without
    it, pbar is computed over data that includes the out-of-control period, which
    drags the centre line toward the disturbance and then flags the healthy
    points on the other side of it. That is not a subtle effect -- it produced 16
    false alarms in 140 in-control points on the first run of this experiment,
    and it is the identical mistake to computing control limits from overall
    sigma, committed on the centre line instead of the spread.
    """
    d = np.asarray(defectives, dtype=float)
    n = np.asarray(sizes, dtype=float)
    p = d / n
    db, nb = (d[baseline], n[baseline]) if baseline is not None else (d, n)
    pbar = float(db.sum() / nb.sum())
    se = np.sqrt(pbar * (1 - pbar) / n)
    return {
        "chart": "p", "stat": p, "center": pbar,
        "ucl": pbar + 3 * se, "lcl": np.maximum(0.0, pbar - 3 * se),
        "variable_limits": True, "sizes": n,
    }


def np_chart(defectives: np.ndarray, n: float,
             baseline: slice | None = None) -> dict:
    """Number defective, CONSTANT subgroup size."""
    d = np.asarray(defectives, dtype=float)
    db = d[baseline] if baseline is not None else d
    pbar = float(db.mean() / n)
    centre = n * pbar
    se = np.sqrt(n * pbar * (1 - pbar))
    return {
        "chart": "np", "stat": d, "center": centre,
        "ucl": np.full(len(d), centre + 3 * se),
        "lcl": np.full(len(d), max(0.0, centre - 3 * se)),
        "variable_limits": False,
    }


def c_chart(counts: np.ndarray, baseline: slice | None = None) -> dict:
    """Count of defects, CONSTANT area of opportunity. Poisson: variance = mean."""
    c = np.asarray(counts, dtype=float)
    cbar = float((c[baseline] if baseline is not None else c).mean())
    se = np.sqrt(cbar)
    return {
        "chart": "c", "stat": c, "center": cbar,
        "ucl": np.full(len(c), cbar + 3 * se),
        "lcl": np.full(len(c), max(0.0, cbar - 3 * se)),
        "variable_limits": False,
    }


def u_chart(counts: np.ndarray, areas: np.ndarray,
            baseline: slice | None = None) -> dict:
    """Defects per unit of opportunity, VARIABLE area."""
    c = np.asarray(counts, dtype=float)
    a = np.asarray(areas, dtype=float)
    u = c / a
    cb, ab = (c[baseline], a[baseline]) if baseline is not None else (c, a)
    ubar = float(cb.sum() / ab.sum())
    se = np.sqrt(ubar / a)
    return {
        "chart": "u", "stat": u, "center": ubar,
        "ucl": ubar + 3 * se, "lcl": np.maximum(0.0, ubar - 3 * se),
        "variable_limits": True, "areas": a,
    }


def violations(chart: dict) -> np.ndarray:
    return (chart["stat"] > chart["ucl"]) | (chart["stat"] < chart["lcl"])


def wrong_chart_penalty(counts: np.ndarray, areas: np.ndarray) -> dict:
    """What using a c chart on varying-area data actually costs.

    The c chart assumes a constant area of opportunity, so it computes one limit
    for every point. When the area varies, that limit is right only for the mean
    area -- and every point with a larger inspection area sits above it for a
    reason that has nothing to do with the process.
    """
    correct = u_chart(counts, areas, baseline=slice(0, 120))
    wrong = c_chart(counts, baseline=slice(0, 120))
    return {
        "u_chart_violations": int(violations(correct).sum()),
        "c_chart_violations": int(violations(wrong).sum()),
        "n_points": len(counts),
        "area_min": float(np.min(areas)), "area_max": float(np.max(areas)),
    }


# --------------------------------------------------------------------------
# generators with planted disturbances
# --------------------------------------------------------------------------

def simulate_p(n_points: int = 200, p0: float = 0.03, shift_at: int = 140,
               shift_to: float = 0.075, size_range=(400, 1200),
               seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(size_range[0], size_range[1], n_points)
    p = np.where(np.arange(n_points) >= shift_at, shift_to, p0)
    d = rng.binomial(sizes, p)
    return {"defectives": d, "sizes": sizes, "shift_at": shift_at,
            "p0": p0, "shift_to": shift_to}


def simulate_c(n_points: int = 200, c0: float = 6.0, shift_at: int = 150,
               shift_to: float = 13.0, seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    lam = np.where(np.arange(n_points) >= shift_at, shift_to, c0)
    return {"counts": rng.poisson(lam), "shift_at": shift_at,
            "c0": c0, "shift_to": shift_to}


def simulate_u(n_points: int = 200, u0: float = 2.5, shift_at: int = 150,
               shift_to: float = 4.5, area_range=(0.5, 4.0), seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    areas = rng.uniform(area_range[0], area_range[1], n_points)
    u = np.where(np.arange(n_points) >= shift_at, shift_to, u0)
    return {"counts": rng.poisson(u * areas), "areas": areas,
            "shift_at": shift_at, "u0": u0, "shift_to": shift_to}
