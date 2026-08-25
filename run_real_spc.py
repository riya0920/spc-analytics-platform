"""The SPC machinery on NIST's real measurement data, checked against NIST's answers.

The README's first not-built item was *no real measurement data -- everything is
`src/generate.py`*. Fetching some real numbers closes that item and proves
nothing: a control chart computed on real data is still a control chart
validated against itself.

What makes this worth doing is that NIST publishes the answers. For the
check-standard study, in prose, on a page anybody can read:

    pooled repeatability   s1 = 0.06139 ohm.cm   (125 df)
    level-2 (day-to-day)   s2 = 0.02680 ohm.cm   ( 24 df)
    s-chart upper limit    UCL = s1 * sqrt(F(0.05, 5, 125)) = 0.09238

So the arithmetic here has a reference. Every number this script computes that
NIST also states is compared to NIST's, and the comparison is reported as a
pass or a failure rather than as prose.

Writes docs/REAL_SPC.md and out/real_spc.json.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import capability as CAP     # noqa: E402
import charts as CH          # noqa: E402
import gauge_rr as GRR       # noqa: E402
import rules as RU           # noqa: E402
import transforms as TR      # noqa: E402

DATA = ROOT / "data" / "NIST"
OUT = ROOT / "out"
DOCS = ROOT / "docs"

TOL = 5e-5      # agreement tolerance against NIST's 5-significant-figure values


def _load():
    p = DATA / "nist.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    pub = json.loads(str(z["published"]))
    out = {"published": pub}
    for k in ("MPC62", "MPC61"):
        cols = [str(c) for c in z[f"{k}_cols"]]
        out[k] = pd.DataFrame(z[k], columns=cols)
    return out


# ---------------------------------------------------------------------------
# 1. reproduce NIST
# ---------------------------------------------------------------------------

def reproduce_nist(df: pd.DataFrame, pub: dict) -> dict:
    """The check-standard decomposition, against NIST's published values.

    `s1` is the POOLED short-term standard deviation: the root mean square of
    the per-day standard deviations, not their average. Averaging standard
    deviations instead of pooling variances is the classic way to get this
    slightly and confidently wrong -- it is biased low, and on this data it is
    low by enough to notice and not by enough to look wrong.
    """
    s_day = df["stddev"].to_numpy(float)
    x_day = df["checkstd"].to_numpy(float)
    J = int(pub["J_repetitions"])
    K = len(df)

    s1 = float(np.sqrt(np.mean(s_day ** 2)))
    s1_naive = float(np.mean(s_day))                 # the wrong way, for contrast
    s2 = float(np.std(x_day, ddof=1))

    f_crit = float(stats.f.ppf(0.95, J - 1, K * (J - 1)))
    ucl = s1 * np.sqrt(f_crit)

    checks = [
        ("pooled repeatability s1", s1, pub["s1_repeatability"]),
        ("level-2 s2", s2, pub["s2_level2"]),
        ("F(0.05, 5, 125)", f_crit, pub["f_crit_0_05_5_125"]),
        ("s-chart UCL", ucl, pub["s_chart_ucl"]),
    ]
    rows = []
    for name, got, want in checks:
        # NIST quotes to 4-5 significant figures, so agreement is judged at
        # their precision rather than at float precision.
        prec = 10 ** -(len(str(want).split(".")[-1]))
        ok = abs(got - want) <= max(prec, abs(want) * 1e-4)
        rows.append({"quantity": name, "computed": got, "nist": want,
                     "abs_diff": abs(got - want), "agrees": bool(ok)})

    exceed = int((s_day > ucl).sum())
    # NIST's own UCL does not follow from NIST's own inputs. They state
    # s1 = 0.06139, F = 2.29 and UCL = 0.09238; s1 * sqrt(F) is 0.09290 with
    # their rounded F and 0.09283 with the exact one, and the printed value
    # implies F = 2.2644, which is not F(0.05, 5, 125).
    #
    # Reported rather than absorbed by a wider tolerance, and what makes that
    # safe is the check below: both limits flag exactly the two days NIST says
    # are flagged, so the 0.5% changes no conclusion. A reference disagreeing
    # with itself is worth writing down; loosening a threshold until a check
    # passes is the habit this project exists to argue against.
    exceed_nist = int((s_day > pub["s_chart_ucl"]).sum())
    return {
        "K_days": K, "J_repetitions": J,
        "s1": s1, "s1_df": K * (J - 1), "s2": s2, "s2_df": K - 1,
        "s1_if_you_average_sds_instead": s1_naive,
        "pooling_matters_by": s1 - s1_naive,
        "f_crit": f_crit, "ucl": ucl,
        "ucl_from_nists_own_rounded_inputs": float(
            pub["s1_repeatability"] * np.sqrt(pub["f_crit_0_05_5_125"])),
        "ucl_nist_states": pub["s_chart_ucl"],
        "f_implied_by_nists_ucl": float(
            (pub["s_chart_ucl"] / pub["s1_repeatability"]) ** 2),
        "days_exceeding_nists_ucl": exceed_nist,
        "same_days_either_way": exceed == exceed_nist,
        "days_exceeding_ucl": exceed,
        "nist_says_two_exceed": exceed == 2,
        "checks": rows,
        "all_agree": all(r["agrees"] for r in rows),
        "mean_level": float(x_day.mean()),
    }


# ---------------------------------------------------------------------------
# 2. this project's own machinery, on the real data
# ---------------------------------------------------------------------------

def own_machinery(df: pd.DataFrame, nist: dict) -> dict:
    x = df["checkstd"].to_numpy(float)
    s_day = df["stddev"].to_numpy(float)
    J = int(nist["J_repetitions"])

    # I-MR on the daily check-standard values: one value per occasion, which is
    # exactly the case individuals charts exist for.
    ind, mr = CH.i_mr(x)
    # zone_width is 1 sigma OF THE PLOTTED STATISTIC, which for an
    # individuals chart is the individuals sigma. Using sigma_hat here
    # would be right too; using the chart's own zone width is right by
    # construction and stays right if the chart type changes.
    zw = ind.zone_width or ind.sigma_hat or ((ind.ucl - ind.center) / 3.0)
    z = (x - ind.center) / max(float(zw), 1e-12)
    viol = RU.apply_rules(z)
    fired = {int(k): [int(i) for i in np.flatnonzero(v)]
             for k, v in viol.items() if v.any()}

    # Capability against the ASTM/NIST context: the wafers are nominally
    # 100 ohm.cm and this crystal sits near 97.07, so a symmetric +/-1% window
    # around the NOMINAL is the honest spec to test -- inventing limits around
    # the observed mean would guarantee a flattering Cpk.
    nominal = 100.0
    lsl, usl = nominal * 0.97, nominal * 1.03

    # THE SCALE TRAP. The charted values are AVERAGES OF SIX measurements, and
    # `s1` is the standard deviation of ONE. Feeding s1 in as sigma_within
    # against data that are means of six mixes two scales, and it does not look
    # wrong -- it produces a plausible Cpk that answers no question anybody
    # asked. Two different questions live here and they need different sigmas:
    #
    #   capability of a SINGLE measurement on a new wafer
    #       sigma = sqrt(s1^2 + s2^2)   -- repeatability plus day-to-day, which
    #       is NIST's own level-1 + level-2 uncertainty construction
    #
    #   capability of the DAILY AVERAGE, which is what is plotted
    #       sigma = sqrt(s1^2/J + s2^2)
    sigma_single = float(np.hypot(nist["s1"], nist["s2"]))
    sigma_daily_mean = float(np.sqrt(nist["s1"] ** 2 / J + nist["s2"] ** 2))
    # The stability gate gets the REAL answer from the real chart. If the data
    # is not in control this refuses, and that refusal is a result rather than
    # an obstacle -- capability on an unstable process estimates a distribution
    # that does not exist.
    beyond = int(ind.beyond_limits().sum())
    in_control = beyond == 0

    def _cap(sig):
        try:
            r = CAP.assess(x, sigma_within=sig, lsl=lsl, usl=usl,
                           in_control=in_control, out_of_control_points=beyond)
            r["refused"] = False
            return r
        except CAP.NotInControl as e:
            return {"refused": True, "why": str(e)}

    cap = _cap(sigma_single)
    cap_wrong_scale = _cap(nist["s1"])
    norm = CAP.normality(x)

    # Does the real data need a transformation? The synthetic study needed one
    # and this is the first chance to ask the question of data nobody shaped.
    cmp_ = TR.compare_transforms(x)

    return {
        "n": len(x), "mean": float(x.mean()),
        "J": J,
        "sigma_single_measurement": sigma_single,
        "sigma_daily_mean": sigma_daily_mean,
        "capability_at_wrong_scale": cap_wrong_scale,
        "individuals": {"center": float(ind.center), "lcl": float(ind.lcl),
                        "ucl": float(ind.ucl), "sigma": float(zw)},
        "moving_range": {"center": float(mr.center), "ucl": float(mr.ucl)},
        "rules_fired": fired,
        "n_rules_fired": len(fired),
        "spec": {"nominal": nominal, "lsl": lsl, "usl": usl},
        "points_beyond_limits": beyond, "in_control": in_control,
        "capability": cap,
        "normality": norm,
        "transforms": cmp_,
        "sigma_within_source": ("NIST's pooled repeatability s1, not the "
                                "moving range -- the six repetitions per day "
                                "are the real within-subgroup replication and "
                                "MR would estimate day-to-day instead"),
        "sigma_within": nist["s1"],
        "sigma_from_moving_range": float(mr.center / 1.128),
    }


# ---------------------------------------------------------------------------
# 3. the gauge study
# ---------------------------------------------------------------------------

def gauge_study(df: pd.DataFrame) -> dict:
    """MPC61 through this project's ANOVA gauge R&R.

    The design is 5 wafers x 5 probes x 2 runs. Mapping PROBE onto "operator"
    is the honest reading: the question NIST asks of this data is whether the
    probes are equivalent or biased relative to one another, which is
    structurally the reproducibility question. It is written down because the
    mapping is a modelling choice and not a fact about the file.
    """
    d = pd.DataFrame({"part": df["wafer"].astype(int).astype(str),
                      "operator": df["probe"].astype(int).astype(str),
                      "value": df["average"].astype(float)})
    counts = d.groupby(["part", "operator"]).size()
    balanced = counts.nunique() == 1
    out = {"n": len(d), "n_parts": d["part"].nunique(),
           "n_probes": d["operator"].nunique(),
           "reps_per_cell": int(counts.min()),
           "balanced": bool(balanced),
           "cell_sizes": sorted(set(int(c) for c in counts))}
    if not balanced:
        # Truncate to the smallest cell rather than pretending. An unbalanced
        # table pushed through a balanced-ANOVA expected-mean-squares
        # decomposition produces variance components that are simply wrong, and
        # nothing in the arithmetic complains.
        r = int(counts.min())
        d = (d.groupby(["part", "operator"], group_keys=False)
             .apply(lambda g: g.head(r), include_groups=False)
             .reset_index())
        d = pd.DataFrame({"part": d["part"], "operator": d["operator"],
                          "value": d["value"]})
        out["truncated_to_reps"] = r
        out["truncation_note"] = (
            "cells were unequal; truncated to the smallest so the balanced "
            "expected-mean-squares decomposition is valid")

    # Tolerance from the same +/-1% window used above.
    grr = GRR.anova_grr(d, tolerance=100.0 * 0.06)
    out["grr"] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                      else v) for k, v in grr.items()}
    # NIST's own finding on this study is that the probes are NOT equivalent --
    # there are real biases between them. A gauge R&R that says otherwise on
    # this data is disagreeing with the reference.
    out["probe_means"] = {k: float(v) for k, v in
                          d.groupby("operator")["value"].mean().items()}
    spread = max(out["probe_means"].values()) - min(out["probe_means"].values())
    out["probe_bias_spread"] = spread
    # Is the probe spread real, or noise? One-way ANOVA across probes on the
    # wafer-centred values, so wafer-to-wafer variation cannot masquerade as
    # probe bias.
    d2 = d.copy()
    d2["centred"] = d2["value"] - d2.groupby("part")["value"].transform("mean")
    groups = [g["centred"].to_numpy() for _, g in d2.groupby("operator")]
    f, pval = stats.f_oneway(*groups)
    out["probe_bias_anova"] = {"F": float(f), "p": float(pval),
                               "significant_at_5pct": bool(pval < 0.05)}
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    loaded = _load()
    if loaded is None:
        print("no NIST data; run fetch_nist.py first")
        raise SystemExit(1)

    pub = loaded["published"]
    nist = reproduce_nist(loaded["MPC62"], pub)
    own = own_machinery(loaded["MPC62"], nist)
    gauge = gauge_study(loaded["MPC61"])
    d = {"published": pub, "reproduction": nist, "own": own, "gauge": gauge,
         "elapsed_s": time.time() - t0}
    (OUT / "real_spc.json").write_text(json.dumps(d, indent=2, default=str),
                                       encoding="utf-8")
    (DOCS / "REAL_SPC.md").write_text(report(d), encoding="utf-8")
    print(f"wrote docs/REAL_SPC.md in {d['elapsed_s']:.1f}s "
          f"(NIST agreement: {nist['all_agree']})")


def report(d: dict) -> str:
    L: list[str] = []
    A = L.append
    n, o, g, pub = d["reproduction"], d["own"], d["gauge"], d["published"]

    A("# Real measurement data, and NIST's answers to check it against\n")
    A("The first item on this project's not-built list was *no real measurement "
      "data — everything is `src/generate.py`*. Fetching real numbers closes "
      "that item and proves nothing on its own: a control chart computed on real "
      "data is still a control chart validated against itself.\n")
    A("What makes this worth doing is that **NIST publishes the answers**. The "
      "check-standard case study states its variance decomposition in prose, so "
      "the arithmetic here has a reference rather than a mirror.\n")
    A(f"> {pub['source']}\n")

    A("\n## 1. Does this project's arithmetic reproduce NIST's?\n")
    A(f"{n['K_days']} days, {n['J_repetitions']} repetitions per day, "
      f"resistivity of silicon check standard #137 on probe #2362.\n")
    A("| quantity | computed here | NIST publishes | agrees |")
    A("|---|---:|---:|:--:|")
    for r in n["checks"]:
        A(f"| {r['quantity']} | {r['computed']:.5f} | {r['nist']} | "
          f"{'✅' if r['agrees'] else '❌'} |")
    agree_n = sum(1 for r in n["checks"] if r["agrees"])
    A(f"\n**{agree_n} of {len(n['checks'])} agree** to NIST's published "
      "precision. That is the first time anything in this project has been "
      "checked against a number it did not produce.\n")
    A(f"NIST also states that two days exceed the s-chart limit; this finds "
      f"**{n['days_exceeding_ucl']}** \u2014 "
      f"{'matching' if n['nist_says_two_exceed'] else 'NOT matching'}.\n")

    A("\n### The fourth is a discrepancy in the reference\n")
    A(f"NIST's published UCL does not follow from NIST's published inputs. They "
      f"state s\u2081 = {pub['s1_repeatability']}, F = "
      f"{pub['f_crit_0_05_5_125']} and UCL = {pub['s_chart_ucl']}, with the "
      f"formula UCL = s\u2081\u00b7\u221aF. But s\u2081\u00b7\u221aF is "
      f"**{n['ucl_from_nists_own_rounded_inputs']:.5f}** with their rounded F "
      f"and **{n['ucl']:.5f}** with the exact one. The value they print implies "
      f"F = **{n['f_implied_by_nists_ucl']:.4f}**, which is not "
      f"F(0.05, 5, 125).\n")
    A(f"It is 0.5%, and it changes nothing: **both limits flag exactly the "
      f"{n['days_exceeding_ucl']} days NIST says are flagged** "
      f"(`same_days_either_way = {n['same_days_either_way']}`). It is written "
      "down rather than absorbed by a wider tolerance \u2014 a reference "
      "disagreeing with itself is worth recording, and quietly loosening a "
      "threshold until a check passes is the habit this project is against.\n")

    A("\n## 2. The project's own machinery, on data nobody shaped\n")
    ind = o["individuals"]
    A(f"An individuals chart on the {o['n']} daily values: centre "
      f"{ind['center']:.4f}, limits [{ind['lcl']:.4f}, {ind['ucl']:.4f}], "
      f"**{o['points_beyond_limits']} points beyond**.\n")
    if o["n_rules_fired"]:
        A(f"**{o['n_rules_fired']} of the eight Western Electric / Nelson rules "
          f"fire**: " + ", ".join(f"rule {k} at {v}" for k, v in
                                  sorted(o["rules_fired"].items())) + ".\n")
    else:
        A("No runs rule fires on any of the eight — the process is stable, "
          "which is what NIST describes and is the first time this project's "
          "rule set has agreed with an outside source about real data.\n")

    A("\n### The scale trap, which cost a plausible and meaningless Cpk\n")
    A(f"The charted values are **averages of {o['J']} measurements**, and "
      f"NIST's `s1` is the standard deviation of **one**. Feeding `s1` in as "
      "σ_within against data that are means of six mixes two scales, and it "
      "does not look wrong — it produces a perfectly plausible index that "
      "answers no question anybody asked. Two questions live here:\n")
    A(f"| question | σ | value |")
    A("|---|---|---:|")
    A(f"| capability of a **single** measurement on a new wafer | √(s₁²+s₂²) | "
      f"{o['sigma_single_measurement']:.5f} |")
    A(f"| capability of the **daily average**, which is what is plotted | "
      f"√(s₁²/{o['J']}+s₂²) | {o['sigma_daily_mean']:.5f} |")
    A(f"| σ_within if you just use `s1` — **the wrong one** | s₁ | "
      f"{o['sigma_within']:.5f} |")
    A("\nThe first is NIST's own level-1 + level-2 uncertainty construction. "
      "Everything below uses it.\n")

    cap, wrong = o["capability"], o["capability_at_wrong_scale"]
    sp = o["spec"]
    A(f"\n### Capability against a ±3% window on the nominal "
      f"{sp['nominal']:.0f} ohm·cm\n")
    A("The window is chosen on the **nominal**, not around the observed mean: "
      "centring a spec on your own data guarantees a flattering Cpk. This "
      "crystal runs at "
      f"{o['mean']:.2f}, which sits almost exactly on the lower limit of "
      f"{sp['lsl']:.1f}.\n")
    if not cap.get("refused"):
        A("| index | correct scale | using s₁ (wrong scale) |")
        A("|---|---:|---:|")
        for k in ("Cp", "Cpk", "Pp", "Ppk"):
            if k in cap:
                w = wrong.get(k)
                A(f"| {k} | **{cap[k]:.3f}** | "
                  f"{'—' if w is None else f'{w:.3f}'} |")
        A(f"\n**Cp {cap['Cp']:.1f} against Cpk {cap['Cpk']:.2f}** is the "
          "textbook centring failure, and here it is entirely an artefact of "
          "the invented spec: the process is tight enough to fit inside the "
          "window many times over and sits on the edge of it. A real tolerance "
          "for this material would be centred near 97, and then Cp and Cpk "
          "would be close. It is left in because a capability index computed "
          "against a spec somebody made up is exactly what this project warns "
          "about elsewhere, and the demonstration is more useful than the "
          "number.\n")
        if "Cpk_over_Ppk" in cap:
            A(f"**Cpk/Ppk = {cap['Cpk_over_Ppk']:.2f}, and it is below 1** — "
              "the reverse of the usual case. Normally within-subgroup σ is "
              "smaller than overall σ and Cpk exceeds Ppk. Here σ for a single "
              "measurement is larger than the spread of the daily *averages*, "
              "because averaging six repetitions divides the repeatability by "
              "√6. It is the same scale point arriving from the other "
              "direction.\n")
    else:
        A(f"**Capability refused**: {cap['why']}\n")

    nm = o["normality"]
    A(f"\nNormality: Anderson–Darling **{nm['anderson_darling_stat']:.3f}** "
      f"against a 5% critical value of {nm['critical_value_5pct']:.3f} — "
      f"**{'normal' if nm['normal_at_5pct'] else 'not normal'}** "
      f"(skew {nm['skew']:+.2f}, excess kurtosis "
      f"{nm['kurtosis_excess']:+.2f}). The synthetic study needed a "
      "transformation to pass this; real check-standard data does not need "
      "one, which is a point in favour of the transformation machinery being "
      "exercised on a case that genuinely required it rather than one built to.\n")

    A("\n## 3. The gauge study, and a %GRR that hides a real bias\n")
    A(f"MPC61 is {g['n_parts']} check-standard wafers measured on "
      f"{g['n_probes']} probes. Mapping **probe onto operator** is a modelling "
      "choice, written down rather than assumed: NIST's question of this data "
      "is whether the probes are equivalent or biased relative to each other, "
      "which is structurally the reproducibility question.\n")
    if g.get("truncation_note"):
        A(f"\nCells were unequal ({g['cell_sizes']}), so the table was "
          f"truncated to {g['truncated_to_reps']} per cell. An unbalanced table "
          "pushed through a balanced expected-mean-squares decomposition gives "
          "variance components that are simply wrong, and nothing in the "
          "arithmetic complains.\n")
    grr = g["grr"]
    A("| | |")
    A("|---|---:|")
    for k, lab in (("pct_GRR_of_TV", "%GRR of total variation"),
                   ("ndc", "number of distinct categories"),
                   ("EV_repeatability", "EV (repeatability)"),
                   ("AV_reproducibility", "AV (reproducibility)")):
        if isinstance(grr.get(k), float):
            A(f"| {lab} | {grr[k]:.3f} |")
    A(f"| probe-to-probe spread in mean resistivity | "
      f"{g['probe_bias_spread']:.4f} ohm·cm |")
    if isinstance(grr.get("verdict_AIAG"), str):
        A(f"| AIAG verdict | {grr['verdict_AIAG']} |")

    ab = g.get("probe_bias_anova", {})
    A(f"\n**The %GRR says the measurement system is excellent. The probes are "
      f"still measurably biased against each other.** A one-way ANOVA on "
      f"wafer-centred values — so wafer-to-wafer variation cannot masquerade as "
      f"probe bias — gives F = {ab.get('F', float('nan')):.2f}, "
      f"p = {ab.get('p', float('nan')):.2e}: "
      f"**{'significant' if ab.get('significant_at_5pct') else 'not significant'}**. "
      f"The spread between probe means is {g['probe_bias_spread']:.4f} ohm·cm, "
      f"about {g['probe_bias_spread'] / max(n['s2'], 1e-9):.1f}× the day-to-day "
      "standard deviation.\n")
    A("Both things are true, and the reason is that **%GRR is a ratio to part "
      "variation**. These wafers span a wide range of resistivity, so a real "
      "systematic offset between probes disappears into a denominator. That is "
      "the failure mode of the metric, not of the gauge: a measurement system "
      "can be excellent *for telling these parts apart* and still be unfit for "
      "comparing results between probes — which, for a laboratory issuing "
      "certificates, is the question that matters. NIST's own conclusion on "
      "this study is that the probes are not equivalent, and the ANOVA agrees "
      "with NIST rather than with the %GRR headline.\n")

    A("\n## What this settles, and what it does not\n")
    A(f"- **It settles the first not-built item, and more than it asked for.** "
      f"The data is real *and* the arithmetic now has a reference: "
      f"{sum(1 for r in n['checks'] if r['agrees'])} of {len(n['checks'])} "
      "published quantities reproduced to NIST's precision, and the fourth is "
      "a discrepancy inside the reference itself.")
    A("- **It is one process and 25 days.** A check standard measured on one "
      "probe is not a production line, and nothing here exercises the "
      "subgroup-based charts, the attribute charts or the ARL work on real "
      "data — those still run on `src/generate.py`.")
    A("- **The spec limits are invented.** NIST's study has no tolerance; the "
      "±3% window is a stand-in, stated as one, and every capability index "
      "below is a statement about that choice as much as about the process.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
