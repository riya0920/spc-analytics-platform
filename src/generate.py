"""Simulated measurement streams with planted disturbances and full ground truth.

Every disturbance below is recorded in a truth table, so "the chart detected it"
is a scoreable claim rather than a screenshot. The disturbance catalogue is chosen
to cover the failure modes that separate the chart types:

  sustained mean shift   -- Shewhart is fine at 3 sigma, poor at 0.5 sigma;
                            EWMA/CUSUM exist for exactly this
  trend                  -- tool wear; caught by the runs rules, not by rule 1
  cyclic                 -- shift-to-shift or ambient; often mistaken for a shift
  variance increase      -- the R chart's job, invisible on X-bar until it is bad
  bimodal / mixed lots   -- the case that destroys overall-sigma limits, which is
                            why it is in here specifically

Measurement error is layered on top of process variation because the two are
routinely confused on a plant floor, and one characteristic here is built so the
"process problem" is really a gauge problem (see gauge_rr.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import zlib

import numpy as np
import pandas as pd

SEED = 20260818


@dataclass
class Disturbance:
    kind: str
    start_subgroup: int
    end_subgroup: int
    magnitude_sigma: float
    note: str = ""


@dataclass
class Characteristic:
    name: str
    target: float
    sigma_process: float
    sigma_gauge: float
    lsl: float
    usl: float
    subgroup_size: int = 5
    n_subgroups: int = 200
    disturbances: list[Disturbance] = field(default_factory=list)
    skew: float = 0.0


def simulate(ch: Characteristic, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (measurements, truth).

    measurements: one row per part, columns subgroup / part / value
    truth:        one row per disturbance, with the subgroup window it occupies
    """
    # crc32, NOT hash(). Python salts str hashing per process (PEP 456), so
    # `hash(ch.name)` returns a different value on every interpreter start --
    # which made this generator produce DIFFERENT DATA on every run and silently
    # broke the reproducibility every report in this project claims. Every
    # number published before this fix came from a dataset a re-run could not
    # reproduce.
    rng = np.random.default_rng(seed + zlib.crc32(ch.name.encode()) % 10_000)
    n, m = ch.subgroup_size, ch.n_subgroups
    base = rng.standard_normal((m, n))
    if ch.skew > 0:
        # Lognormal-ish skew, rescaled to keep sigma_process meaningful.
        base = np.exp(ch.skew * base)
        base = (base - base.mean()) / base.std()

    value = ch.target + ch.sigma_process * base
    sub_index = np.arange(m)

    for d in ch.disturbances:
        m_win = (sub_index >= d.start_subgroup) & (sub_index < d.end_subgroup)
        idx = np.where(m_win)[0]
        if d.kind == "shift":
            value[idx] += d.magnitude_sigma * ch.sigma_process
        elif d.kind == "trend":
            ramp = np.linspace(0, d.magnitude_sigma, len(idx))
            value[idx] += (ramp * ch.sigma_process)[:, None]
        elif d.kind == "cycle":
            period = max(4, (d.end_subgroup - d.start_subgroup) // 4)
            wave = d.magnitude_sigma * np.sin(2 * np.pi * np.arange(len(idx)) / period)
            value[idx] += (wave * ch.sigma_process)[:, None]
        elif d.kind == "spread":
            value[idx] = ch.target + (value[idx] - ch.target) * d.magnitude_sigma
        elif d.kind == "bimodal":
            # Two cavities / two lots running simultaneously: half the parts in
            # each subgroup come from a stream offset by +/- magnitude.
            half = n // 2
            value[np.ix_(idx, np.arange(half))] += d.magnitude_sigma * ch.sigma_process
            value[np.ix_(idx, np.arange(half, n))] -= d.magnitude_sigma * ch.sigma_process
        else:
            raise ValueError(f"unknown disturbance kind {d.kind!r}")

    measured = value + ch.sigma_gauge * rng.standard_normal((m, n))

    rows = []
    for i in range(m):
        for j in range(n):
            rows.append({"subgroup": i, "part": j, "value": measured[i, j]})
    meas = pd.DataFrame(rows)
    truth = pd.DataFrame([{
        "characteristic": ch.name, "kind": d.kind, "start_subgroup": d.start_subgroup,
        "end_subgroup": d.end_subgroup, "magnitude_sigma": d.magnitude_sigma, "note": d.note,
    } for d in ch.disturbances])
    return meas, truth


def catalogue() -> list[Characteristic]:
    """The characteristics the platform is exercised on."""
    return [
        Characteristic(
            name="shaft_dia_mm", target=25.00, sigma_process=0.010, sigma_gauge=0.002,
            lsl=24.95, usl=25.05, n_subgroups=300,
            disturbances=[
                Disturbance("shift", 100, 140, 0.5, "small sustained shift - EWMA/CUSUM territory"),
                Disturbance("shift", 200, 230, 2.0, "large shift - rule 1 should own this"),
                Disturbance("trend", 250, 300, 3.0, "tool wear ramp"),
            ],
        ),
        Characteristic(
            name="wall_thk_mm", target=2.000, sigma_process=0.020, sigma_gauge=0.004,
            lsl=1.940, usl=2.060, n_subgroups=250,
            disturbances=[
                Disturbance("spread", 120, 170, 2.5, "variance doubling - R chart, not X-bar"),
                Disturbance("cycle", 190, 250, 1.2, "shift-to-shift cycling"),
            ],
        ),
        Characteristic(
            name="cavity_mix_mm", target=12.000, sigma_process=0.015, sigma_gauge=0.003,
            lsl=11.94, usl=12.06, n_subgroups=200,
            disturbances=[
                Disturbance("bimodal", 0, 200, 1.5,
                            "two cavities pooled into one subgroup for the whole run"),
            ],
        ),
        Characteristic(
            name="seal_force_N", target=180.0, sigma_process=6.0, sigma_gauge=1.2,
            lsl=160.0, usl=205.0, n_subgroups=200, skew=0.55,
            disturbances=[],
        ),
        Characteristic(
            # Process is tight; the gauge is not. Any capability computed on the
            # measured values blames the process for the instrument's noise.
            name="bore_rough_um", target=1.60, sigma_process=0.040, sigma_gauge=0.075,
            lsl=1.40, usl=1.80, n_subgroups=200, disturbances=[],
        ),
    ]
