"""ANOVA gauge R&R — measurement systems analysis.

The question SPC cannot answer on its own: when a characteristic looks variable,
how much of that variation is the PROCESS and how much is the INSTRUMENT reading
it? A control chart cannot tell the difference; it charts what the gauge says.

Crossed study: every operator measures every part, several times.
    Repeatability (EV, equipment variation)  -- same operator, same part, spread
    Reproducibility (AV, appraiser variation)-- operator-to-operator differences
    GRR = sqrt(EV^2 + AV^2)
    PV  -- part-to-part variation, the signal you wanted
    TV  = sqrt(GRR^2 + PV^2)

Reported as %GRR = GRR/TV, with the AIAG MSA-4 acceptance bands:
    < 10%      acceptable
    10% - 30%  conditionally acceptable, depending on application and cost
    > 30%      unacceptable
and ndc = 1.41 * PV/GRR (number of distinct categories), where the same manual's
rule of thumb is ndc >= 5. ndc is the more intuitive statement: it is how many
distinct groups of parts the measurement system can actually tell apart. An ndc
of 2 means the gauge sorts parts into "big" and "small" and nothing else.

ANOVA method rather than average-and-range: the ANOVA method estimates the
operator x part INTERACTION (one operator measuring large parts differently),
which the range method structurally cannot see.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def simulate_study(n_parts: int = 10, n_operators: int = 3, n_repeats: int = 3,
                   sigma_part: float = 1.0, sigma_repeat: float = 0.15,
                   sigma_operator: float = 0.10, sigma_interaction: float = 0.05,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    part_true = rng.normal(0, sigma_part, n_parts)
    op_bias = rng.normal(0, sigma_operator, n_operators)
    inter = rng.normal(0, sigma_interaction, (n_parts, n_operators))
    rows = []
    for p, o, r in itertools.product(range(n_parts), range(n_operators), range(n_repeats)):
        rows.append({
            "part": p, "operator": o, "repeat": r,
            "value": part_true[p] + op_bias[o] + inter[p, o] + rng.normal(0, sigma_repeat),
        })
    return pd.DataFrame(rows)


def anova_grr(df: pd.DataFrame, tolerance: float | None = None) -> dict:
    """Balanced crossed ANOVA gauge R&R.

    Variance components come from expected mean squares for the balanced
    two-factor model with replication:
        MS_E   = sigma_repeat^2
        MS_PO  = sigma_repeat^2 + r*sigma_interaction^2
        MS_O   = sigma_repeat^2 + r*sigma_interaction^2 + p*r*sigma_operator^2
        MS_P   = sigma_repeat^2 + r*sigma_interaction^2 + o*r*sigma_part^2
    Negative variance estimates are truncated to zero, which is standard practice
    and also a signal that the interaction term is not supported by the data --
    AIAG's guidance is to drop it and re-fit, which is what happens here.
    """
    p = df["part"].nunique()
    o = df["operator"].nunique()
    r = len(df) // (p * o)
    grand = df["value"].mean()

    part_means = df.groupby("part")["value"].mean()
    op_means = df.groupby("operator")["value"].mean()
    cell_means = df.groupby(["part", "operator"])["value"].mean()

    ss_part = o * r * ((part_means - grand) ** 2).sum()
    ss_op = p * r * ((op_means - grand) ** 2).sum()
    cell = cell_means.reset_index()
    cell["exp"] = cell.apply(
        lambda row: part_means[row["part"]] + op_means[row["operator"]] - grand, axis=1
    )
    ss_inter = r * ((cell["value"] - cell["exp"]) ** 2).sum()
    merged = df.merge(cell_means.rename("cell_mean"), on=["part", "operator"])
    ss_err = ((merged["value"] - merged["cell_mean"]) ** 2).sum()

    df_part, df_op, df_inter, df_err = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    ms_part, ms_op = ss_part / df_part, ss_op / df_op
    ms_inter = ss_inter / df_inter if df_inter else 0.0
    ms_err = ss_err / df_err

    f_inter = ms_inter / ms_err if ms_err > 0 else 0.0
    from scipy import stats as _st
    p_inter = float(_st.f.sf(f_inter, df_inter, df_err)) if df_inter else 1.0
    interaction_significant = p_inter < 0.25  # AIAG uses alpha = 0.25 for this pooling decision

    if interaction_significant:
        v_repeat = ms_err
        v_inter = max(0.0, (ms_inter - ms_err) / r)
        v_op = max(0.0, (ms_op - ms_inter) / (p * r))
        v_part = max(0.0, (ms_part - ms_inter) / (o * r))
    else:
        # Pool the interaction into error and re-fit, per AIAG MSA-4.
        ms_pooled = (ss_inter + ss_err) / (df_inter + df_err)
        v_repeat = ms_pooled
        v_inter = 0.0
        v_op = max(0.0, (ms_op - ms_pooled) / (p * r))
        v_part = max(0.0, (ms_part - ms_pooled) / (o * r))

    ev = np.sqrt(v_repeat)
    av = np.sqrt(v_op + v_inter)
    grr = np.sqrt(ev**2 + av**2)
    pv = np.sqrt(v_part)
    tv = np.sqrt(grr**2 + pv**2)
    pct = 100 * grr / tv if tv > 0 else float("nan")
    ndc = 1.41 * pv / grr if grr > 0 else float("inf")

    verdict = "acceptable" if pct < 10 else ("conditional" if pct < 30 else "UNACCEPTABLE")
    out = {
        "parts": p, "operators": o, "repeats": r,
        "EV_repeatability": float(ev), "AV_reproducibility": float(av),
        "GRR": float(grr), "PV_part": float(pv), "TV_total": float(tv),
        "pct_GRR_of_TV": float(pct), "ndc": float(ndc),
        "interaction_p": p_inter,
        "interaction_kept": bool(interaction_significant),
        "verdict_AIAG": verdict,
        "pct_of_process_variation_that_is_gauge": float(100 * grr**2 / tv**2),
    }
    if tolerance:
        out["pct_GRR_of_tolerance"] = float(100 * 6 * grr / tolerance)
    return out
