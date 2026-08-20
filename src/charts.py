"""Control charts. The correctness that matters is in one place: WHERE SIGMA COMES FROM.

Control limits are estimated from WITHIN-subgroup variation (Rbar/d2 or sbar/c4),
never from the standard deviation of the whole series.

Why this is not a stylistic preference: a control chart asks "is the variation
between subgroups larger than the variation within them?". Overall sigma contains
both. If the process has shifted, the shift inflates overall sigma, which widens
the limits, which hides the shift. The chart is at its blindest exactly when the
process is at its worst. That single substitution converts a control chart into a
descriptive band plot, and it is the most common defect in industrial dashboards.

`cavity_mix_mm` in the generator exists to make this concrete: two cavities pooled
into every subgroup means within-subgroup variation is INFLATED by the cavity
offset, so limits computed correctly are wide and the chart looks calm. That is
the chart telling the truth about a badly chosen rational subgroup, and the fix is
to re-subgroup by cavity, not to change the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from constants import A2, A3, B3, B4, D3, D4, c4, d2


@dataclass
class Chart:
    name: str
    stat: np.ndarray
    center: float
    ucl: float
    lcl: float
    sigma_hat: float | None = None
    zone_width: float | None = None  # 1 sigma of the PLOTTED statistic

    def beyond_limits(self) -> np.ndarray:
        return (self.stat > self.ucl) | (self.stat < self.lcl)


def _subgroup_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.pivot(index="subgroup", columns="part", values="value").to_numpy()


def xbar_r(df: pd.DataFrame, baseline: slice | None = None) -> tuple[Chart, Chart]:
    """X-bar and R charts with limits from Rbar/d2.

    `baseline` selects the subgroups used to ESTIMATE the limits. Limits are
    normally frozen on a stable baseline period and then applied forward -
    re-estimating them on every new point lets a drifting process drag its own
    limits along behind it and alarm on nothing.
    """
    x = _subgroup_matrix(df)
    n = x.shape[1]
    b = x[baseline] if baseline is not None else x
    xbar = x.mean(axis=1)
    r = x.max(axis=1) - x.min(axis=1)
    xbb = float(b.mean())
    rbar = float((b.max(axis=1) - b.min(axis=1)).mean())
    sigma_hat = rbar / d2(n)

    xchart = Chart(
        name="X-bar (R-based)", stat=xbar, center=xbb,
        ucl=xbb + A2(n) * rbar, lcl=xbb - A2(n) * rbar,
        sigma_hat=sigma_hat, zone_width=sigma_hat / np.sqrt(n),
    )
    rchart = Chart(
        name="R", stat=r, center=rbar, ucl=D4(n) * rbar, lcl=D3(n) * rbar,
        sigma_hat=sigma_hat,
    )
    return xchart, rchart


def xbar_s(df: pd.DataFrame, baseline: slice | None = None) -> tuple[Chart, Chart]:
    """X-bar and s charts with limits from sbar/c4. Preferred for n >= 10."""
    x = _subgroup_matrix(df)
    n = x.shape[1]
    b = x[baseline] if baseline is not None else x
    xbar = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    xbb = float(b.mean())
    sbar = float(b.std(axis=1, ddof=1).mean())
    sigma_hat = sbar / c4(n)
    xchart = Chart(
        name="X-bar (s-based)", stat=xbar, center=xbb,
        ucl=xbb + A3(n) * sbar, lcl=xbb - A3(n) * sbar,
        sigma_hat=sigma_hat, zone_width=sigma_hat / np.sqrt(n),
    )
    schart = Chart(name="s", stat=s, center=sbar, ucl=B4(n) * sbar, lcl=B3(n) * sbar,
                   sigma_hat=sigma_hat)
    return xchart, schart


def i_mr(values: np.ndarray, baseline: slice | None = None) -> tuple[Chart, Chart]:
    """Individuals and moving-range charts.

    Sigma comes from MRbar/d2(2) - the average of successive differences - not
    from the SD of the individuals, for the same reason as above: successive
    differences are the closest thing to within-subgroup variation available when
    the subgroup size is one.
    """
    v = np.asarray(values, dtype=float)
    b = v[baseline] if baseline is not None else v
    mr = np.abs(np.diff(v))
    mrb = float(np.abs(np.diff(b)).mean())
    sigma_hat = mrb / d2(2)
    center = float(b.mean())
    ichart = Chart(name="I", stat=v, center=center,
                   ucl=center + 3 * sigma_hat, lcl=center - 3 * sigma_hat,
                   sigma_hat=sigma_hat, zone_width=sigma_hat)
    mrchart = Chart(name="MR", stat=np.concatenate([[np.nan], mr]), center=mrb,
                    ucl=D4(2) * mrb, lcl=D3(2) * mrb, sigma_hat=sigma_hat)
    return ichart, mrchart


def p_chart(defectives: np.ndarray, sizes: np.ndarray) -> Chart:
    """Attribute chart for fraction nonconforming; limits vary with subgroup size."""
    d = np.asarray(defectives, dtype=float)
    n = np.asarray(sizes, dtype=float)
    p = d / n
    pbar = float(d.sum() / n.sum())
    half = 3.0 * np.sqrt(pbar * (1 - pbar) / n)
    return Chart(name="p", stat=p, center=pbar,
                 ucl=float(np.mean(pbar + half)), lcl=float(max(0.0, np.mean(pbar - half))),
                 sigma_hat=float(np.mean(np.sqrt(pbar * (1 - pbar) / n))))


def naive_limits(values: np.ndarray) -> tuple[float, float, float]:
    """The wrong chart, implemented on purpose so the error can be MEASURED.

    Limits at mean +/- 3 * overall SD. Used only in the comparison table in
    RESULTS.md, where the point is how many real signals it swallows.
    """
    v = np.asarray(values, dtype=float)
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    return mu, mu + 3 * sd, mu - 3 * sd
