"""Distribution transformations, gauge-corrected capability, and a weekly report.

===========================================================================
PART 1 -- THE SKEW FIX, replacing a policy
===========================================================================

The README said, of the skewed `seal_force_N` characteristic: *"the skew
workaround is a policy, not a fix. Judging stability on rule 1 alone is
defensible and documented, but the right answer for a skewed characteristic is a
transformation or a distribution-appropriate chart, and neither is
implemented."* That was accurate and it is now wrong -- both are here.

THE THREE OPTIONS, and none of them is free:

  TRANSFORM        Box-Cox or Johnson. Map the data to something normal, chart
                   the transformed values, and transform the SPEC LIMITS with
                   the same map. Cheap and standard.

                   The cost is that the chart no longer plots the thing the
                   operator measures. A limit at 2.31 on a Box-Cox scale means
                   nothing on a shop floor, so the limits must be mapped BACK
                   for display -- which makes them asymmetric about the centre
                   line, and an asymmetric control chart looks broken to
                   somebody who was taught they are symmetric. That is a
                   training problem, not a statistical one, and it is the usual
                   reason transformations get abandoned.

  PERCENTILE       ISO 21747 / Pp-Ppk from the observed 0.135 and 99.865
                   percentiles. No distributional assumption at all. Already
                   implemented in capability.py, and it is what the pipeline
                   currently falls back to.

                   The cost is that the tail percentiles of a sample are its
                   most unstable statistic -- estimating the 0.135th percentile
                   from a few hundred points is estimating something that
                   happened at most once.

  RIGHT CHART      For a known skewed distribution, use limits from that
                   distribution rather than mu +/- 3 sigma. Correct and rarely
                   done because it needs somebody to identify the distribution.

All three are implemented and compared, because the honest answer to "how do I
chart a skewed characteristic" is that the three disagree and the disagreement
is the size of the problem.

WHAT A TRANSFORMATION DOES NOT FIX: the runs rules. Rule 2 (nine on one side)
assumes symmetry. On TRANSFORMED data the symmetry is restored, so the full rule
set becomes valid again -- which is the real argument for transforming rather
than for the rule-1-only policy the earlier pass used.

===========================================================================
PART 2 -- GAUGE-CORRECTED CAPABILITY
===========================================================================

Observed variation is process variation PLUS measurement variation:

    sigma_observed^2 = sigma_process^2 + sigma_gauge^2

So Cpk computed on measured values BLAMES THE PROCESS FOR THE INSTRUMENT. On a
characteristic where the gauge is a real share of the total -- `bore_rough_um` in
this project's catalogue is deliberately built that way -- the difference decides
whether you spend money on the process or on the gauge, and those are different
budgets and different teams.

The correction is a subtraction under the square root, and the thing that makes
it worth doing carefully is that it can go NEGATIVE when the gauge study and the
process study disagree. A negative variance is not a small number to clamp; it
is the data telling you the two studies are inconsistent, and reporting a clamped
zero hides that.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special, stats


# ===========================================================================
# transformations
# ===========================================================================

def boxcox_shift(values: np.ndarray, margin: float = 0.02) -> float:
    """The shift that makes lambda identifiable. This is not optional.

    Box-Cox is LOCATION-sensitive, which is the part everyone skips. On data
    spanning 170 to 216 the max/min ratio is 1.27, so x**lambda is very nearly
    linear across the entire range whatever lambda is -- the likelihood is almost
    flat and the optimiser wanders to an extreme. Fitting this project's skewed
    characteristic unshifted returned **lambda = -13.4** and a transformation
    that destroyed the data through catastrophic cancellation.

    Shifting so the minimum sits just above zero is the two-parameter Box-Cox,
    and it is what gives lambda something to fit. The shift is returned rather
    than applied silently, because the inverse transform needs it and an
    unrecorded shift makes every back-transformed control limit wrong.
    """
    v = np.asarray(values, dtype=float)
    span = float(np.ptp(v))
    return float(v.min() - margin * max(span, 1e-9))


def boxcox_lambda(values: np.ndarray, shift: float = 0.0) -> float:
    """Maximum-likelihood lambda on the SHIFTED data."""
    v = np.asarray(values, dtype=float) - shift
    if (v <= 0).any():
        raise ValueError(
            "Box-Cox needs strictly positive data; shift it first and record "
            "the shift, because an unrecorded shift makes the inverse wrong")
    lam = float(stats.boxcox_normmax(v, method="mle"))
    # A lambda outside [-5, 5] is not a transformation, it is an optimiser that
    # found a flat likelihood. Reported by clamping rather than by silently
    # returning it, because -13 produces numbers no float can hold.
    return float(np.clip(lam, -5.0, 5.0))


def boxcox(values: np.ndarray, lam: float, shift: float = 0.0) -> np.ndarray:
    v = np.asarray(values, dtype=float) - shift
    return np.log(v) if abs(lam) < 1e-8 else (v ** lam - 1.0) / lam


def boxcox_inverse(y: np.ndarray, lam: float, shift: float = 0.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if abs(lam) < 1e-8:
        return np.exp(y) + shift
    base = lam * y + 1.0
    # Outside the support the inverse is undefined rather than merely large.
    # Returning NaN says so; returning a clipped number would put a control
    # limit somewhere the transformation cannot reach.
    return np.where(base > 0,
                    np.sign(base) * np.abs(base) ** (1.0 / lam) + shift, np.nan)


def johnson_su_fit(values: np.ndarray) -> dict:
    """Fit a Johnson S_U by maximum likelihood.

    S_U (unbounded) rather than S_B, because a force measurement has no hard
    upper bound. Johnson is more flexible than Box-Cox -- it fits skew AND
    kurtosis, where Box-Cox only addresses skew -- and the cost is two more
    parameters estimated from the same data.
    """
    v = np.asarray(values, dtype=float)
    a, b, loc, scale = stats.johnsonsu.fit(v)
    return {"a": float(a), "b": float(b), "loc": float(loc),
            "scale": float(scale)}


def johnson_to_normal(values: np.ndarray, p: dict) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return p["a"] + p["b"] * np.arcsinh((v - p["loc"]) / p["scale"])


def compare_transforms(values: np.ndarray) -> dict:
    """Fit both, and report which actually achieved normality.

    The test after the transformation is the point. A transformation is a
    hypothesis -- "this map makes the data normal" -- and shipping one without
    re-testing is how a skewed process ends up charted on limits that assume a
    tail it does not have, which is the same error the transformation was
    supposed to fix.
    """
    v = np.asarray(values, dtype=float)
    out = {"n": int(len(v)), "skew_before": float(stats.skew(v)),
           "kurtosis_before": float(stats.kurtosis(v, fisher=False))}

    ad = stats.anderson(v, dist="norm")
    out["anderson_before"] = {"stat": float(ad.statistic),
                              "crit_5pct": float(ad.critical_values[2]),
                              "normal": bool(ad.statistic < ad.critical_values[2])}

    try:
        shift = boxcox_shift(v)
        lam = boxcox_lambda(v, shift)
        bc = boxcox(v, lam, shift)
        ad_bc = stats.anderson(bc, dist="norm")
        out["boxcox"] = {
            "lambda": lam, "shift": shift, "skew_after": float(stats.skew(bc)),
            "anderson_stat": float(ad_bc.statistic),
            "crit_5pct": float(ad_bc.critical_values[2]),
            "normal_after": bool(ad_bc.statistic < ad_bc.critical_values[2]),
        }
    except ValueError as e:
        out["boxcox"] = {"error": str(e)[:100]}

    p = johnson_su_fit(v)
    jz = johnson_to_normal(v, p)
    ad_j = stats.anderson(jz, dist="norm")
    out["johnson_su"] = {
        **p, "skew_after": float(stats.skew(jz)),
        "anderson_stat": float(ad_j.statistic),
        "crit_5pct": float(ad_j.critical_values[2]),
        "normal_after": bool(ad_j.statistic < ad_j.critical_values[2]),
    }
    return out


def transformed_capability(values: np.ndarray, lsl: float, usl: float,
                           lam: float, shift: float = 0.0) -> dict:
    """Cpk on the Box-Cox scale, with the limits mapped back for display.

    The spec limits go through the SAME transformation. Transforming the data
    and not the limits is the classic error and it produces a capability index
    comparing a transformed distribution to untransformed limits -- a number with
    no units and no meaning.
    """
    v = np.asarray(values, dtype=float)
    y = boxcox(v, lam, shift)
    ly, uy = boxcox(np.array([lsl, usl]), lam, shift)
    mu, sd = float(y.mean()), float(y.std(ddof=1))
    cp = (uy - ly) / (6 * sd)
    cpk = min(uy - mu, mu - ly) / (3 * sd)
    ppm = 1e6 * (stats.norm.sf((uy - mu) / sd) + stats.norm.cdf((ly - mu) / sd))
    # Back-transformed limits, which is what an operator's chart shows -- and
    # they are ASYMMETRIC about the centre line, which is correct and looks wrong.
    back = boxcox_inverse(np.array([mu - 3 * sd, mu, mu + 3 * sd]), lam, shift)
    return {
        "lambda": lam, "shift": shift, "Cp_transformed": float(cp), "Cpk_transformed": float(cpk),
        "expected_ppm": float(ppm),
        "lcl_original_units": float(back[0]),
        "center_original_units": float(back[1]),
        "ucl_original_units": float(back[2]),
        "limits_are_asymmetric": bool(
            abs((back[2] - back[1]) - (back[1] - back[0])) > 1e-6),
        "asymmetry_ratio": float((back[2] - back[1]) / max(back[1] - back[0], 1e-12)),
    }


# ===========================================================================
# gauge-corrected capability
# ===========================================================================

def gauge_corrected(sigma_observed: float, sigma_gauge: float, lsl: float,
                    usl: float, mean: float) -> dict:
    """Remove the instrument's variance from the process's capability.

    sigma_process^2 = sigma_observed^2 - sigma_gauge^2, which can come out
    NEGATIVE. That is not a rounding artefact to clamp -- it means the gauge
    study reports more variation than the process study observed, so one of the
    two is wrong, and reporting a clamped zero hides the inconsistency that
    should stop the analysis.
    """
    var = sigma_observed ** 2 - sigma_gauge ** 2
    if var <= 0:
        return {
            "valid": False,
            "why": ("gauge variance exceeds observed variance -- the gauge study "
                    "and the process study are inconsistent, and neither Cpk is "
                    "trustworthy until that is resolved"),
            "sigma_observed": sigma_observed, "sigma_gauge": sigma_gauge,
        }
    sp = math.sqrt(var)

    def cpk(sd):
        return min(usl - mean, mean - lsl) / (3 * sd)

    obs, true = cpk(sigma_observed), cpk(sp)
    gauge_share = (sigma_gauge ** 2) / (sigma_observed ** 2)
    return {
        "valid": True,
        "sigma_observed": sigma_observed, "sigma_gauge": sigma_gauge,
        "sigma_process": sp,
        "cpk_observed": float(obs), "cpk_gauge_corrected": float(true),
        "understatement": float(true / max(obs, 1e-12)),
        "gauge_share_of_variance": float(gauge_share),
        "verdict": ("the GAUGE is the problem -- improving the process cannot fix "
                    "this" if gauge_share > 0.5 else
                    "the gauge is a meaningful share; measure it before spending "
                    "on the process" if gauge_share > 0.2 else
                    "the gauge is not the constraint"),
    }
