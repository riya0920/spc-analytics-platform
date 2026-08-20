"""DATA-2 end-to-end: charts, rules validation, ARL bake-off, capability, gauge R&R.

    python run_spc.py

Writes docs/RESULTS.md and out/results.json. Every table is measured here; the
narrative sentences are formatted from the measured numbers.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import arl  # noqa: E402
import capability  # noqa: E402
import charts  # noqa: E402
import constants  # noqa: E402
import gauge_rr  # noqa: E402
import generate  # noqa: E402
import rules  # noqa: E402

OUT = ROOT / "out"
BASELINE = slice(0, 100)  # subgroups used to estimate limits, then frozen


def constants_check() -> list[dict]:
    published_d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
                    7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    published_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
                    7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    rows = []
    for n in range(2, 11):
        rows.append({
            "n": n,
            "d2_derived": constants.d2(n), "d2_published": published_d2[n],
            "d2_abs_err": abs(constants.d2(n) - published_d2[n]),
            "c4": constants.c4(n),
            "A2_derived": constants.A2(n), "A2_published": published_A2[n],
            "A2_abs_err": abs(constants.A2(n) - published_A2[n]),
        })
    return rows


def chart_run(ch: generate.Characteristic) -> dict:
    meas, truth = generate.simulate(ch)
    xchart, rchart = charts.xbar_r(meas, baseline=BASELINE)
    z = rules.zones(xchart.stat, xchart.center, xchart.zone_width)
    fired = rules.apply_rules(z)
    any_fire = rules.any_violation(z)

    # The wrong chart, for comparison.
    mu_n, ucl_n, lcl_n = charts.naive_limits(meas["value"].to_numpy())
    xbar = xchart.stat
    naive_sigma_of_mean = (ucl_n - mu_n) / 3 / np.sqrt(ch.subgroup_size)
    z_naive = (xbar - mu_n) / naive_sigma_of_mean
    naive_fire = rules.any_violation(z_naive)

    # Score each planted disturbance: was it detected, and how many subgroups did
    # detection take from the moment the disturbance started?
    detections = []
    for _, d in truth.iterrows():
        s, e = int(d["start_subgroup"]), int(d["end_subgroup"])
        win = np.arange(len(xbar))
        inside = (win >= s) & (win < e)
        hit = np.flatnonzero(any_fire & inside)
        hit_naive = np.flatnonzero(naive_fire & inside)
        first_rule = None
        if len(hit):
            first_rule = sorted(r for r, v in fired.items() if v[hit[0]])
        detections.append({
            "kind": d["kind"], "magnitude_sigma": float(d["magnitude_sigma"]),
            "note": d["note"], "window": f"{s}-{e}",
            "detected": bool(len(hit)),
            "subgroups_to_detect": int(hit[0] - s) if len(hit) else None,
            "first_rules": first_rule,
            "detected_by_naive_chart": bool(len(hit_naive)),
            "subgroups_to_detect_naive": int(hit_naive[0] - s) if len(hit_naive) else None,
        })

    # False alarms outside every planted window.
    clean = np.ones(len(xbar), dtype=bool)
    for _, d in truth.iterrows():
        clean[int(d["start_subgroup"]):int(d["end_subgroup"])] = False
    clean[: BASELINE.stop] = clean[: BASELINE.stop]  # baseline is in-control by construction
    n_clean = int(clean.sum())

    r_violations = int(rchart.beyond_limits().sum())
    return {
        "characteristic": ch.name,
        "n_subgroups": ch.n_subgroups, "subgroup_size": ch.subgroup_size,
        "sigma_within_hat": xchart.sigma_hat, "sigma_process_true": ch.sigma_process,
        "sigma_gauge_true": ch.sigma_gauge,
        "limits_correct": {"center": xchart.center, "ucl": xchart.ucl, "lcl": xchart.lcl},
        "limits_naive_overall_sigma": {"center": mu_n, "ucl": ucl_n, "lcl": lcl_n,
                                       "width_ratio": (ucl_n - lcl_n) / (xchart.ucl - xchart.lcl)},
        "detections": detections,
        "false_alarms_on_clean_subgroups": int((any_fire & clean).sum()),
        "clean_subgroups": n_clean,
        "false_alarm_rate_clean": float((any_fire & clean).sum() / max(1, n_clean)),
        "r_chart_violations": r_violations,
        "_meas": meas, "_xchart": xchart, "_z": z, "_any_fire": any_fire,
    }


def capability_run(ch: generate.Characteristic, run: dict) -> dict:
    meas = run["_meas"]
    values = meas["value"].to_numpy()
    xchart = run["_xchart"]
    baseline_vals = meas[meas["subgroup"] < BASELINE.stop]["value"].to_numpy()
    ooc = int(rules.any_violation(run["_z"][: BASELINE.stop]).sum())
    try:
        res = capability.assess(baseline_vals, xchart.sigma_hat, ch.lsl, ch.usl,
                                in_control=(ooc == 0), out_of_control_points=ooc)
        res["refused"] = False
    except capability.NotInControl as exc:
        res = {"refused": True, "reason": str(exc), "out_of_control_points": ooc}
    # Whole-series capability, computed anyway to show what the refusal protects
    # against: this is the number a naive tool would have printed.
    full = capability.normal_capability(values, xchart.sigma_hat, ch.lsl, ch.usl)
    res["whole_series_if_computed_anyway"] = {
        "Cpk": full["Cpk"], "Ppk": full["Ppk"], "Cpk_over_Ppk": full["Cpk_over_Ppk"],
        "observed_ppm": full["observed_ppm"],
        "expected_ppm_long_term": full["expected_ppm_long_term"],
    }
    res["characteristic"] = ch.name
    return res


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    res: dict = {}

    print("1/5 control-chart constants ...", flush=True)
    res["constants"] = constants_check()

    print("2/5 charts + rules on planted disturbances ...", flush=True)
    runs, caps = [], []
    for ch in generate.catalogue():
        r = chart_run(ch)
        caps.append(capability_run(ch, r))
        runs.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"    {ch.name:<16} "
              f"{sum(d['detected'] for d in r['detections'])}/{len(r['detections'])} planted "
              f"disturbances detected, {r['false_alarms_on_clean_subgroups']} false alarms "
              f"on {r['clean_subgroups']} clean subgroups", flush=True)
    res["chart_runs"] = runs
    res["capability"] = caps

    print("3/5 per-rule false-alarm and detection rates ...", flush=True)
    res["rule_false_alarm"] = arl.per_rule_performance(reps=6000, shift=0.0)
    res["rule_detection_1sigma"] = arl.per_rule_performance(reps=6000, shift=1.0)

    print("4/5 calibrating EWMA / CUSUM to matched ARL0 ...", flush=True)
    L = arl.calibrate("ewma", reps=4000)
    h = arl.calibrate("cusum", lo=3.0, hi=9.0, reps=4000)
    res["calibration"] = {
        "ewma_L_calibrated": L, "ewma_L_textbook": 2.962,
        "ewma_arl0_at_calibrated": arl.arl_ewma(0.0, L=L, reps=6000),
        "ewma_arl0_at_textbook": arl.arl_ewma(0.0, L=2.962, reps=6000),
        "cusum_h_calibrated": h, "cusum_h_textbook": 4.77,
        "cusum_arl0_at_calibrated": arl.arl_cusum(0.0, h=h, reps=6000),
    }
    print(f"    EWMA L*={L:.3f} (textbook 2.962), CUSUM h*={h:.3f} (textbook 4.77)", flush=True)

    print("5/5 ARL table ...", flush=True)
    res["arl_table"] = arl.arl_table(reps=6000, ewma_L=L, cusum_h=h)

    print("    gauge R&R ...", flush=True)
    tol = 0.40 - (-0.40)
    res["gauge_rr"] = {
        "good_gauge": gauge_rr.anova_grr(
            gauge_rr.simulate_study(sigma_part=1.0, sigma_repeat=0.05,
                                    sigma_operator=0.03, sigma_interaction=0.01), tolerance=6.0),
        "marginal_gauge": gauge_rr.anova_grr(
            gauge_rr.simulate_study(sigma_part=1.0, sigma_repeat=0.18,
                                    sigma_operator=0.10, sigma_interaction=0.05), tolerance=6.0),
        "bad_gauge": gauge_rr.anova_grr(
            gauge_rr.simulate_study(sigma_part=1.0, sigma_repeat=0.45,
                                    sigma_operator=0.30, sigma_interaction=0.15), tolerance=6.0),
    }
    # The misdiagnosis case: bore_rough_um has a tight process and a noisy gauge.
    ch = [c for c in generate.catalogue() if c.name == "bore_rough_um"][0]
    res["gauge_rr"]["bore_rough_um_study"] = gauge_rr.anova_grr(
        gauge_rr.simulate_study(sigma_part=ch.sigma_process, sigma_repeat=ch.sigma_gauge,
                                sigma_operator=ch.sigma_gauge * 0.4,
                                sigma_interaction=ch.sigma_gauge * 0.2),
        tolerance=ch.usl - ch.lsl)
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/RESULTS.md and out/results.json ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# DATA-2 results — generated by `run_spc.py`, not hand-edited\n")

    A("## 1. Control-chart constants, derived vs published\n")
    A("`d2` is computed by numerically integrating the expected range of *n* standard "
      "normals; `c4` from log-gamma. The published column is Montgomery, *Introduction "
      "to Statistical Quality Control*, Appendix VI. If these disagreed, every limit in "
      "the platform would be wrong by a constant and no test downstream would notice.\n")
    A("| n | d2 derived | d2 published | abs err | c4 | A2 derived | A2 published | abs err |")
    A("|---|---|---|---|---|---|---|---|")
    for r in res["constants"]:
        A(f"| {r['n']} | {r['d2_derived']:.5f} | {r['d2_published']:.3f} | "
          f"{r['d2_abs_err']:.2e} | {r['c4']:.4f} | {r['A2_derived']:.4f} | "
          f"{r['A2_published']:.3f} | {r['A2_abs_err']:.2e} |")
    worst = max(r["d2_abs_err"] for r in res["constants"])
    A(f"\nWorst absolute disagreement on d2: {worst:.2e} — i.e. the published table "
      "rounded to three decimals.")

    A("\n## 2. Within-subgroup limits vs the overall-sigma mistake\n")
    A("| characteristic | correct limit width | naive limit width | ratio | disturbances detected (correct) | detected (naive) |")
    A("|---|---|---|---|---|---|")
    for r in res["chart_runs"]:
        c = r["limits_correct"]
        n = r["limits_naive_overall_sigma"]
        det = sum(d["detected"] for d in r["detections"])
        detn = sum(d["detected_by_naive_chart"] for d in r["detections"])
        A(f"| {r['characteristic']} | {c['ucl']-c['lcl']:.4f} | {n['ucl']-n['lcl']:.4f} | "
          f"{n['width_ratio']:.2f}x | {det}/{len(r['detections'])} | {detn}/{len(r['detections'])} |")
    A("\nThe ratio column is the whole argument. Overall sigma contains the very "
      "between-subgroup variation the chart exists to detect, so a disturbed process "
      "widens its own limits and then looks calm inside them. Note the direction of the "
      "effect: it is worst exactly when the process is worst.")

    A("\n### Every planted disturbance, scored\n")
    A("| characteristic | disturbance | detected | subgroups to detect | first rule(s) | naive chart |")
    A("|---|---|---|---|---|---|")
    for r in res["chart_runs"]:
        for d in r["detections"]:
            rl = ",".join(str(x) for x in (d["first_rules"] or [])) or "—"
            nv = ("yes, +%s subgroups" % d["subgroups_to_detect_naive"]
                  if d["detected_by_naive_chart"] else "**missed**")
            A(f"| {r['characteristic']} | {d['kind']} {d['magnitude_sigma']}σ "
              f"({d['note']}) | {'yes' if d['detected'] else '**no**'} | "
              f"{d['subgroups_to_detect'] if d['detected'] else '—'} | {rl} | {nv} |")

    A("\n## 3. Per-rule economics: what each rule costs and what it buys\n")
    A("Probability the rule fires at least once over a 200-point chart. The left column "
      "is the false-alarm rate on a perfectly in-control process; the right is the "
      "detection rate when the process has shifted 1σ. Both matter — a rule with high "
      "power and high false-alarm rate is how charts get ignored.\n")
    A("| rule | description | P(fires) in control | P(fires) at 1σ shift |")
    A("|---|---|---|---|")
    det = {r["rule"]: r["p_fires"] for r in res["rule_detection_1sigma"]}
    for r in res["rule_false_alarm"]:
        A(f"| {r['rule']} | {r['description']} | {r['p_fires']*100:.1f}% | "
          f"{det.get(r['rule'], float('nan'))*100:.1f}% |")
    fa1 = next(r["p_fires"] for r in res["rule_false_alarm"] if r["rule"] == "1")
    fa4 = next(r["p_fires"] for r in res["rule_false_alarm"] if r["rule"] == "1-4 stacked")
    fa8 = next(r["p_fires"] for r in res["rule_false_alarm"] if r["rule"] == "1-8 stacked")
    A(f"\n**Stacking inflation, measured.** Rule 1 alone fires on {fa1*100:.1f}% of clean "
      f"200-point charts. The four classic Western Electric rules together: {fa4*100:.1f}%. "
      f"All eight: {fa8*100:.1f}% — a {fa8/fa1:.1f}× increase in nuisance alarms over rule 1 "
      "alone. Every rule is another test on the same data. Shops that enable all eight "
      "because the software offers them are buying sensitivity with a currency they have "
      "not been shown the price of.")

    A("\n## 4. ARL: the head-to-head, at matched false-alarm budgets\n")
    c = res["calibration"]
    A(f"EWMA (λ=0.2) is used at **L = {c['ewma_L_calibrated']:.3f}**, not the textbook "
      f"2.962. At the textbook value this implementation measures ARL₀ = "
      f"{c['ewma_arl0_at_textbook']:.0f}, because it uses exact time-varying limits rather "
      "than the asymptotic ones the published constants assume. Calibrated, ARL₀ = "
      f"{c['ewma_arl0_at_calibrated']:.0f}. CUSUM (k=0.5) at h = {c['cusum_h_calibrated']:.3f} "
      f"gives ARL₀ = {c['cusum_arl0_at_calibrated']:.0f}. Comparing detectors at different "
      "ARL₀ is how any chart can be made to win.\n")
    A("| shift | Shewhart (rule 1) | Shewhart (WE 1-4) | Shewhart (all 8) | EWMA | CUSUM |")
    A("|---|---|---|---|---|---|")
    for r in res["arl_table"]:
        A(f"| {r['shift_sigma']}σ | {r['shewhart_rule1']:.1f} | {r['shewhart_rules1_4']:.1f} | "
          f"{r['shewhart_all8']:.1f} | {r['ewma']:.1f} | {r['cusum']:.1f} |")
    row0 = res["arl_table"][0]
    row1 = next(r for r in res["arl_table"] if r["shift_sigma"] == 1.0)
    row3 = next(r for r in res["arl_table"] if r["shift_sigma"] == 3.0)
    A(f"\nRow 1 is the false-alarm budget: rule 1 alone measures ARL₀ = "
      f"{row0['shewhart_rule1']:.0f} against the theoretical 370.4, while the same chart "
      f"with all eight rules enabled measures {row0['shewhart_all8']:.0f} — the stacking "
      "cost again, now in run-length units.")
    A(f"\nAt a 1σ shift, EWMA signals in {row1['ewma']:.1f} points and CUSUM in "
      f"{row1['cusum']:.1f}, against {row1['shewhart_rule1']:.1f} for a 3σ Shewhart chart. "
      "So why not EWMA everywhere? Look at the 3σ row: "
      f"Shewhart needs {row3['shewhart_rule1']:.1f} points, EWMA {row3['ewma']:.1f}. The "
      "memory that makes EWMA sensitive to small shifts also makes it *slower* on large "
      "ones, because the shifted observation is averaged with in-control history. Add that "
      "an operator can read a Shewhart chart without explaining what a smoothing constant "
      "is, and the answer is tool-per-purpose: Shewhart for the floor, EWMA/CUSUM alongside "
      "it for the slow drifts the floor chart will not catch.")

    A("\n## 5. Capability, and the refusals\n")
    A("| characteristic | verdict | Cp | Cpk | Pp | Ppk | method |")
    A("|---|---|---|---|---|---|---|")
    for cp in res["capability"]:
        if cp.get("refused"):
            A(f"| {cp['characteristic']} | **REFUSED** ({cp['out_of_control_points']} OOC points) "
              f"| — | — | — | — | — |")
        else:
            g = lambda k: f"{cp[k]:.2f}" if k in cp else "—"
            A(f"| {cp['characteristic']} | computed | {g('Cp')} | {g('Cpk')} | "
              f"{g('Pp_percentile') if 'Pp_percentile' in cp else g('Pp')} | "
              f"{g('Ppk_percentile') if 'Ppk_percentile' in cp else g('Ppk')} | "
              f"{cp.get('method','—')} |")
    A("\nA refusal is a feature. `capability.assess()` raises `NotInControl` rather than "
      "returning a number with a footnote, because Cpk converts a mean and a sigma into a "
      "tail probability, and an unstable process has neither. The footnote gets deleted on "
      "the way to the customer; the number does not.\n")
    A("### What the refusal protects against\n")
    A("Numbers a tool without the gate would have printed:\n")
    A("| characteristic | Cpk | Ppk | Cpk/Ppk | predicted PPM | observed PPM |")
    A("|---|---|---|---|---|---|")
    for cp in res["capability"]:
        w = cp["whole_series_if_computed_anyway"]
        A(f"| {cp['characteristic']} | {w['Cpk']:.2f} | {w['Ppk']:.2f} | "
          f"{w['Cpk_over_Ppk']:.2f} | {w['expected_ppm_long_term']:,.0f} | "
          f"{w['observed_ppm']:,.0f} |")
    A("\nThe Cpk/Ppk column is the diagnostic. A ratio near 1 means the process is not "
      "moving: short-term and long-term spread agree. A large ratio means the process is "
      "capable within a subgroup and is not staying put between them — and the action that "
      "follows is completely different. Chasing spread on a drifting process is how "
      "improvement budgets get spent on the wrong thing.")

    A("\n## 6. Gauge R&R — is it the process or the instrument?\n")
    A("ANOVA method, 10 parts × 3 operators × 3 repeats, crossed. AIAG MSA-4 bands.\n")
    A("| study | %GRR of total variation | ndc | repeatability (EV) | reproducibility (AV) | interaction kept | verdict |")
    A("|---|---|---|---|---|---|---|")
    for name, g in res["gauge_rr"].items():
        A(f"| {name} | {g['pct_GRR_of_TV']:.1f}% | {g['ndc']:.1f} | {g['EV_repeatability']:.4f} | "
          f"{g['AV_reproducibility']:.4f} | {'yes' if g['interaction_kept'] else 'pooled away'} | "
          f"**{g['verdict_AIAG']}** |")
    b = res["gauge_rr"]["bore_rough_um_study"]
    A(f"\n**The misdiagnosis case.** `bore_rough_um` has σ_process = 0.040 µm and "
      f"σ_gauge = 0.075 µm — the instrument is noisier than the thing it measures. Its "
      f"gauge study returns %GRR = {b['pct_GRR_of_TV']:.1f}% and ndc = {b['ndc']:.1f}: the "
      "measurement system can distinguish roughly "
      f"{max(1, int(b['ndc']))} categor{'y' if int(b['ndc'])<=1 else 'ies'} of part. Any "
      "capability study on this characteristic is measuring the gauge. Any improvement "
      "project chartered off its control chart is chartered against noise. ndc is the "
      "number to put in front of a manager: not a percentage, but *how many different "
      "sizes of part this instrument can actually tell apart*.")

    A("\n---\n*All data here is simulated by `src/generate.py` with the disturbances "
      "recorded as ground truth, which is what makes 'detected' a scoreable claim rather "
      "than an assertion.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
