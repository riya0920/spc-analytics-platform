"""Process capability, with the two disciplines that make it mean anything.

DISCIPLINE 1 — Cp/Cpk and Pp/Ppk are different questions, not synonyms.
    Cp / Cpk use SHORT-TERM sigma, estimated within subgroups (Rbar/d2). They
        answer: how capable is this process *at its best*, over the span of a
        subgroup, with no between-subgroup drift included.
    Pp / Ppk use LONG-TERM sigma, the overall standard deviation of everything.
        They answer: what did the customer actually receive, drift and all.
    Cpk >> Ppk therefore has a specific meaning: the process is inherently
    capable but is not staying put. The action is to chase the shifts (setup,
    tooling, material lots, shift changes), not to reduce the spread. Reporting
    only Cpk hides the drift; reporting only Ppk hides the opportunity.

DISCIPLINE 2 — capability on an out-of-control process is not a number.
    Cpk extrapolates a tail probability from a mean and a sigma. If the process
    is not stable, there is no single mean and no single sigma, so the estimate
    is of a distribution that does not exist. `assess()` REFUSES rather than
    returning a number with a caveat, because the caveat gets deleted on the way
    to the customer and the number does not.

Normality is checked, not assumed: Cpk's link from a ratio to a PPM defect rate
runs entirely through the normal tail. On a skewed characteristic the percentile
method (ISO 21747 / Clements-style) is used instead and labelled as such.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class NotInControl(Exception):
    """Raised when capability is requested on an unstable process."""


def _ppm(z: float) -> float:
    return float(stats.norm.sf(z) * 1e6)


def normal_capability(values: np.ndarray, sigma_within: float, lsl: float, usl: float) -> dict:
    v = np.asarray(values, dtype=float)
    mu = float(v.mean())
    sigma_overall = float(v.std(ddof=1))
    out = {
        "mu": mu,
        "sigma_within": sigma_within,
        "sigma_overall": sigma_overall,
        "Cp": (usl - lsl) / (6 * sigma_within),
        "Cpk": min((usl - mu) / (3 * sigma_within), (mu - lsl) / (3 * sigma_within)),
        "Pp": (usl - lsl) / (6 * sigma_overall),
        "Ppk": min((usl - mu) / (3 * sigma_overall), (mu - lsl) / (3 * sigma_overall)),
        "method": "normal",
    }
    out["Cpk_over_Ppk"] = out["Cpk"] / out["Ppk"] if out["Ppk"] else float("nan")
    out["expected_ppm_long_term"] = _ppm((usl - mu) / sigma_overall) + _ppm((mu - lsl) / sigma_overall)
    out["observed_ppm"] = float(np.mean((v < lsl) | (v > usl)) * 1e6)
    return out


def percentile_capability(values: np.ndarray, lsl: float, usl: float) -> dict:
    """Non-normal fallback: replace 6-sigma with the 0.135%-99.865% span.

    That span is exactly what +/- 3 sigma means for a normal distribution, so the
    index keeps its interpretation (fraction of the tolerance the process uses)
    without borrowing the normal tail. Percentiles are taken from the empirical
    distribution here; with fewer than a few hundred points a fitted distribution
    would be the better estimator, and that is not built.
    """
    v = np.asarray(values, dtype=float)
    p00135, p50, p99865 = np.percentile(v, [0.135, 50, 99.865])
    span = p99865 - p00135
    return {
        "mu": float(v.mean()),
        "median": float(p50),
        "p0.135": float(p00135),
        "p99.865": float(p99865),
        "Pp_percentile": (usl - lsl) / span,
        "Ppk_percentile": min((usl - p50) / (p99865 - p50), (p50 - lsl) / (p50 - p00135)),
        "observed_ppm": float(np.mean((v < lsl) | (v > usl)) * 1e6),
        "method": "percentile (ISO 21747 style)",
    }


def normality(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    ad = stats.anderson(v, dist="norm")
    crit_5pct = float(ad.critical_values[list(ad.significance_level).index(5.0)])
    return {
        "anderson_darling_stat": float(ad.statistic),
        "critical_value_5pct": crit_5pct,
        "normal_at_5pct": bool(ad.statistic < crit_5pct),
        "skew": float(stats.skew(v)),
        "kurtosis_excess": float(stats.kurtosis(v)),
    }


def assess(values: np.ndarray, sigma_within: float, lsl: float, usl: float,
           in_control: bool, out_of_control_points: int = 0) -> dict:
    """The gate. Stability first, distribution second, index third."""
    if not in_control:
        raise NotInControl(
            f"{out_of_control_points} out-of-control points on the baseline chart. "
            "Capability describes a stable process; on an unstable one Cpk is an "
            "estimate of a distribution that does not exist. Stabilise first."
        )
    norm = normality(values)
    if norm["normal_at_5pct"]:
        res = normal_capability(values, sigma_within, lsl, usl)
    else:
        res = percentile_capability(values, lsl, usl)
        res["note"] = (
            "Anderson-Darling rejects normality at 5%, so the normal-theory Cpk "
            "would convert a ratio into a PPM figure through a tail that is not "
            "there. Percentile method used instead."
        )
    res["normality"] = norm
    return res
