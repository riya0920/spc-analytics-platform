"""Control-chart constants, derived rather than pasted.

d2 and c4 are the unbiasing constants that make within-subgroup variation an
unbiased estimate of sigma. Every SPC textbook prints them in a table; deriving
them is a two-line integral and it means the code cannot silently disagree with
the table it was copied from.

    d2(n) = E[range of n iid N(0,1)]
          = integral over R of [1 - Phi(x)^n - (1-Phi(x))^n] dx

    c4(n) = E[s] / sigma for n iid normals
          = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)

d3(n) = SD[range of n iid N(0,1)] has no comparably clean closed form, so it is
tabulated here from the standard published values (Montgomery, *Introduction to
Statistical Quality Control*, Appendix VI) and the source is named rather than
laundered.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special, stats

# Montgomery Appendix VI, n = 2..10.
D3_TABLE = {
    2: 0.8525, 3: 0.8884, 4: 0.8798, 5: 0.8641, 6: 0.8480,
    7: 0.8332, 8: 0.8198, 9: 0.8078, 10: 0.7971,
}


def d2(n: int) -> float:
    """Expected range of n standard normals, by numerical integration."""

    def f(x: float) -> float:
        p = stats.norm.cdf(x)
        return 1.0 - p**n - (1.0 - p) ** n

    val, _ = integrate.quad(f, -12, 12, limit=400)
    return float(val)


def c4(n: int) -> float:
    """E[s]/sigma. Computed through log-gamma so large n does not overflow."""
    return float(np.sqrt(2.0 / (n - 1)) * np.exp(special.gammaln(n / 2.0) - special.gammaln((n - 1) / 2.0)))


def d3(n: int) -> float:
    return D3_TABLE[n]


def A2(n: int) -> float:
    """X-bar chart limit factor for R-based sigma: LCL/UCL = Xbarbar -/+ A2*Rbar."""
    return 3.0 / (d2(n) * np.sqrt(n))


def A3(n: int) -> float:
    """X-bar chart limit factor for s-based sigma."""
    return 3.0 / (c4(n) * np.sqrt(n))


def D3(n: int) -> float:
    return max(0.0, 1.0 - 3.0 * d3(n) / d2(n))


def D4(n: int) -> float:
    return 1.0 + 3.0 * d3(n) / d2(n)


def B3(n: int) -> float:
    return max(0.0, 1.0 - 3.0 / c4(n) * np.sqrt(1.0 - c4(n) ** 2))


def B4(n: int) -> float:
    return 1.0 + 3.0 / c4(n) * np.sqrt(1.0 - c4(n) ** 2)


def E2(n: int) -> float:
    """I-chart limit factor from the moving range: 3/d2(2) when MR uses pairs."""
    return 3.0 / d2(2)
