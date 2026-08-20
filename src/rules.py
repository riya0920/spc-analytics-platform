"""The runs-rules engine.

Naming, precisely, because the names are checkable: rules 1-4 are the classic
Western Electric zone tests from the *Statistical Quality Control Handbook*
(Western Electric Co., 1956). Rules 5-8 as implemented here are the additional
tests popularised by Lloyd Nelson (*Journal of Quality Technology*, 1984) and
shipped as "Western Electric rules" by most software. Calling all eight "Western
Electric" is the industry's habit; this module keeps the distinction in the
docstring so nobody is misled about provenance.

Every rule returns a boolean array over the plotted points. The engine is
validated in run_spc.py against planted patterns for BOTH detection rate and
false-alarm rate, because a rule set is a hypothesis test and reporting only its
power is half a result.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

RULE_DOCS = {
    1: "1 point beyond 3 sigma (WE)",
    2: "2 of 3 consecutive points beyond 2 sigma, same side (WE)",
    3: "4 of 5 consecutive points beyond 1 sigma, same side (WE)",
    4: "8 consecutive points on one side of the centre line (WE)",
    5: "6 consecutive points steadily increasing or decreasing (Nelson)",
    6: "14 consecutive points alternating up and down (Nelson)",
    7: "15 consecutive points within 1 sigma, either side (Nelson: stratification)",
    8: "8 consecutive points beyond 1 sigma, either side (Nelson: mixture)",
}


def zones(stat: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Signed distance from centre in sigma units of the PLOTTED statistic."""
    return (np.asarray(stat, dtype=float) - center) / sigma


def _window_count(flags: np.ndarray, window: int, need: int) -> np.ndarray:
    """True at index i if >= `need` of the `window` points ENDING at i are flagged.

    Vectorised over a trailing axis so the ARL simulator can run thousands of
    charts at once; the scalar version it replaced is kept honest by
    tests/test_rules.py, which checks both against hand-worked patterns.
    """
    f = flags.astype(np.int32)
    c = np.cumsum(f, axis=-1)
    n = flags.shape[-1]
    s = c.copy()
    s[..., window:] = c[..., window:] - c[..., :-window]
    out = (s >= need) & flags
    out[..., : window - 1] = False
    return out


def _run_of(flags: np.ndarray, length: int) -> np.ndarray:
    """True at index i if the run of consecutive True ending at i is >= length."""
    n = flags.shape[-1]
    idx = np.arange(n)
    last_false = np.maximum.accumulate(np.where(~flags, idx, -1), axis=-1)
    return (idx - last_false) >= length


def apply_rules(z: np.ndarray, which=(1, 2, 3, 4, 5, 6, 7, 8)) -> dict[int, np.ndarray]:
    """Rule violations for a chart (1-D) or a batch of charts (2-D, last axis time)."""
    z = np.asarray(z, dtype=float)
    res: dict[int, np.ndarray] = {}
    above, below = z > 0, z < 0

    if 1 in which:
        res[1] = np.abs(z) > 3
    if 2 in which:
        res[2] = _window_count(above & (z > 2), 3, 2) | _window_count(below & (z < -2), 3, 2)
    if 3 in which:
        res[3] = _window_count(above & (z > 1), 5, 4) | _window_count(below & (z < -1), 5, 4)
    if 4 in which:
        res[4] = _run_of(above, 8) | _run_of(below, 8)
    if 5 in which:
        d = np.diff(z, axis=-1, prepend=z[..., :1])
        res[5] = _run_of(d > 0, 6) | _run_of(d < 0, 6)
    if 6 in which:
        d = np.diff(z, axis=-1, prepend=z[..., :1])
        alt = np.zeros(z.shape, dtype=bool)
        alt[..., 2:] = (d[..., 2:] * d[..., 1:-1]) < 0
        res[6] = _run_of(alt, 13)
    if 7 in which:
        res[7] = _run_of(np.abs(z) < 1, 15)
    if 8 in which:
        res[8] = _run_of(np.abs(z) > 1, 8)
    return res


def any_violation(z: np.ndarray, which=(1, 2, 3, 4, 5, 6, 7, 8)) -> np.ndarray:
    r = apply_rules(z, which)
    out = np.zeros(np.shape(z), dtype=bool)
    for v in r.values():
        out |= v
    return out


# --------------------------------------------------------------------------
# EWMA and CUSUM: the small-shift detectors.
# --------------------------------------------------------------------------

def ewma(x: np.ndarray, lam: float = 0.2, L: float = 2.86,
         mu0: float | None = None, sigma: float | None = None) -> dict:
    """Exponentially weighted moving average. Accepts one chart or a batch.

    Limits are TIME VARYING (they widen from the start-up value towards the
    asymptote). Using the asymptotic limits from point 1 -- which is what the
    textbook L constants assume -- makes the chart insensitive exactly when a
    start-up problem would show. The consequence is that the published
    (lam=0.2, L=2.962 -> ARL0=370) pairing does not hold for this implementation,
    so arl.calibrate() measures L into place instead. See arl.py.
    """
    x = np.asarray(x, dtype=float)
    mu0 = float(x.mean()) if mu0 is None else mu0
    sigma = float(x.std(ddof=1)) if sigma is None else sigma
    n = x.shape[-1]
    z = lfilter([lam], [1.0, -(1.0 - lam)], x - mu0, axis=-1) + mu0
    i = np.arange(1, n + 1)
    half = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    return {"z": z, "ucl": mu0 + half, "lcl": mu0 - half, "center": mu0,
            "violation": (z > mu0 + half) | (z < mu0 - half)}


def cusum(x: np.ndarray, k: float = 0.5, h: float = 4.77,
          mu0: float | None = None, sigma: float | None = None) -> dict:
    """Tabular CUSUM. Accepts one chart or a batch (last axis is time).

    k = 0.5 targets a 1-sigma shift (the reference value is half the shift you
    most want to catch). h = 4.77 is the textbook decision interval for
    ARL0 ~ 370; as with EWMA it is re-measured rather than assumed.
    """
    x = np.asarray(x, dtype=float)
    mu0 = float(x.mean()) if mu0 is None else mu0
    sigma = float(x.std(ddof=1)) if sigma is None else sigma
    zs = (x - mu0) / sigma
    cp = np.zeros_like(zs)
    cn = np.zeros_like(zs)
    p = np.zeros(zs.shape[:-1])
    m = np.zeros(zs.shape[:-1])
    for t in range(zs.shape[-1]):
        p = np.maximum(0.0, p + zs[..., t] - k)
        m = np.maximum(0.0, m - zs[..., t] - k)
        cp[..., t] = p
        cn[..., t] = m
    return {"c_plus": cp, "c_minus": cn, "h": h, "violation": (cp > h) | (cn > h)}
