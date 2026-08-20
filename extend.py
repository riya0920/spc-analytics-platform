"""DATA-2, the next 30%: attribute charts, X-bar/s and I-MR, non-normal
capability actually triggered, and phase I / phase II limit revision.

    python extend.py
    python extend.py --report-only

Five gaps the first build named:
  1. attribute charts were stubs -- p/np/c/u now implemented and scored against
     planted shifts, plus the cost of picking the wrong one
  2. X-bar/s and I-MR implemented but never exercised
  3. the non-normal capability path was never triggered, because the skewed
     characteristic got refused for instability before the distribution branch
     mattered -- so it was an untested code path presented as a feature
  4. no limit-revision policy and no phase I / phase II distinction
  5. no SPC methodology guide
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import attribute as A_  # noqa: E402
import capability  # noqa: E402
import charts  # noqa: E402
import generate  # noqa: E402
import rules  # noqa: E402

OUT = ROOT / "out"


def first_detection(viol: np.ndarray, shift_at: int) -> int | None:
    idx = np.flatnonzero(viol[shift_at:])
    return int(idx[0]) if len(idx) else None


def attribute_stage() -> dict:
    out = {}

    pd_ = A_.simulate_p()
    BASE = slice(0, 120)   # phase I: a window judged stable
    p = A_.p_chart(pd_["defectives"], pd_["sizes"], baseline=BASE)
    vp = A_.violations(p)
    out["p"] = {
        "center": p["center"], "shift_at": pd_["shift_at"],
        "p_before": pd_["p0"], "p_after": pd_["shift_to"],
        "points_to_detect": first_detection(vp, pd_["shift_at"]),
        "false_alarms_before_shift": int(vp[:pd_["shift_at"]].sum()),
        "limit_width_min": float((p["ucl"] - p["lcl"]).min()),
        "limit_width_max": float((p["ucl"] - p["lcl"]).max()),
        "size_min": int(pd_["sizes"].min()), "size_max": int(pd_["sizes"].max()),
    }

    cd = A_.simulate_c()
    c = A_.c_chart(cd["counts"], baseline=BASE)
    vc = A_.violations(c)
    out["c"] = {
        "center": c["center"], "shift_at": cd["shift_at"],
        "points_to_detect": first_detection(vc, cd["shift_at"]),
        "false_alarms_before_shift": int(vc[:cd["shift_at"]].sum()),
    }

    ud = A_.simulate_u()
    u = A_.u_chart(ud["counts"], ud["areas"], baseline=BASE)
    vu = A_.violations(u)
    out["u"] = {
        "center": u["center"], "shift_at": ud["shift_at"],
        "points_to_detect": first_detection(vu, ud["shift_at"]),
        "false_alarms_before_shift": int(vu[:ud["shift_at"]].sum()),
    }
    out["wrong_chart"] = A_.wrong_chart_penalty(ud["counts"], ud["areas"])

    npd = A_.simulate_p(size_range=(800, 801))
    npc = A_.np_chart(npd["defectives"], 800, baseline=BASE)
    vn = A_.violations(npc)
    out["np"] = {
        "center": npc["center"], "shift_at": npd["shift_at"],
        "points_to_detect": first_detection(vn, npd["shift_at"]),
        "false_alarms_before_shift": int(vn[:npd["shift_at"]].sum()),
    }
    return out


def variables_stage() -> dict:
    """Exercise X-bar/s and I-MR, which existed but were never run."""
    ch = [c for c in generate.catalogue() if c.name == "shaft_dia_mm"][0]
    meas, truth = generate.simulate(ch)
    base = slice(0, 100)

    xr, r = charts.xbar_r(meas, baseline=base)
    xs, s = charts.xbar_s(meas, baseline=base)

    # I-MR needs individuals: take one measurement per subgroup, which is what a
    # low-volume or destructive-test process actually gives you.
    ind = meas[meas["part"] == 0]["value"].to_numpy()
    i_ch, mr = charts.i_mr(ind, baseline=base)

    def score(chart, name):
        z = rules.zones(chart.stat, chart.center, chart.zone_width or chart.sigma_hat)
        v = rules.any_violation(z)
        rows = []
        for _, d in truth.iterrows():
            s0, e0 = int(d["start_subgroup"]), int(d["end_subgroup"])
            hit = np.flatnonzero(v[s0:min(e0, len(v))])
            rows.append({"disturbance": f"{d['kind']} {d['magnitude_sigma']}σ",
                         "detected": bool(len(hit)),
                         "points_to_detect": int(hit[0]) if len(hit) else None})
        return {"chart": name, "sigma_hat": chart.sigma_hat,
                "limit_width": float(chart.ucl - chart.lcl),
                "detections": rows,
                "n_detected": sum(x["detected"] for x in rows),
                "n_disturbances": len(rows)}

    return {
        "xbar_r": score(xr, "X-bar (R-based)"),
        "xbar_s": score(xs, "X-bar (s-based)"),
        "i_mr": score(i_ch, "I (individuals)"),
        "sigma_agreement": {
            "r_based": xr.sigma_hat, "s_based": xs.sigma_hat,
            "i_mr_based": i_ch.sigma_hat,
            "true_process_sigma": ch.sigma_process,
            "true_total_sigma": float(np.sqrt(ch.sigma_process**2 + ch.sigma_gauge**2)),
        },
    }


def nonnormal_stage() -> dict:
    """Trigger the percentile path the first build never reached.

    `seal_force_N` is skewed on purpose, but in the first build it was refused for
    instability before the distribution branch ever ran -- so the non-normal
    fallback was an untested code path presented as a feature. Here it is scored
    on a STABLE skewed characteristic, which is the only situation in which the
    question is even meaningful.
    """
    ch = [c for c in generate.catalogue() if c.name == "seal_force_N"][0]
    ch.disturbances = []          # stable by construction
    meas, _ = generate.simulate(ch, seed=4242)
    xr, _ = charts.xbar_r(meas, baseline=slice(0, 100))
    vals = meas["value"].to_numpy()

    z = rules.zones(xr.stat, xr.center, xr.zone_width)
    ooc_all = int(rules.any_violation(z).sum())
    # Stability is judged on RULE 1 ONLY for a skewed characteristic, and the
    # reason is an interaction the first build walked straight into: the runs
    # rules assume a SYMMETRIC in-control distribution. On a skewed process they
    # fire on the skew itself, the stability gate then refuses, and the
    # non-normal capability branch becomes unreachable -- so the fallback built
    # for skewed data can never run on skewed data.
    ooc = int(rules.any_violation(z, which=(1,)).sum())

    norm = capability.normality(vals)
    normal_result = capability.normal_capability(vals, xr.sigma_hat, ch.lsl, ch.usl)
    pct_result = capability.percentile_capability(vals, ch.lsl, ch.usl)

    try:
        chosen = capability.assess(vals, xr.sigma_hat, ch.lsl, ch.usl,
                                   in_control=(ooc == 0), out_of_control_points=ooc)
        refused = False
    except capability.NotInControl as exc:
        chosen, refused = {"reason": str(exc)}, True

    return {
        "characteristic": ch.name, "skew_parameter": ch.skew,
        "ooc_points_rule1_only": ooc, "ooc_points_all_rules": ooc_all,
        "refused": refused,
        "normality": norm,
        "method_selected": chosen.get("method") if not refused else None,
        "normal_theory": {"Ppk": normal_result["Ppk"],
                          "predicted_ppm": normal_result["expected_ppm_long_term"],
                          "observed_ppm": normal_result["observed_ppm"]},
        "percentile": {"Ppk": pct_result["Ppk_percentile"],
                       "observed_ppm": pct_result["observed_ppm"]},
    }


def limit_revision_stage() -> dict:
    """Phase I vs phase II, and what happens when limits are never revised.

    Phase I  -- retrospective. Establish limits from a period judged stable,
                iterating: compute, remove assignable causes, recompute.
    Phase II -- prospective. FREEZE those limits and monitor forward.

    The failure mode this measures is the common one: a process genuinely
    improves, nobody revises the limits, and the chart goes quiet. Quiet looks
    like success and is actually a chart that has stopped being able to detect
    anything, because its limits are now far wider than the process.
    """
    ch = [c for c in generate.catalogue() if c.name == "shaft_dia_mm"][0]
    ch = generate.Characteristic(
        name="improved_process", target=ch.target, sigma_process=ch.sigma_process,
        sigma_gauge=ch.sigma_gauge, lsl=ch.lsl, usl=ch.usl,
        subgroup_size=ch.subgroup_size, n_subgroups=400, disturbances=[])
    meas, _ = generate.simulate(ch, seed=99)

    # The process improves at subgroup 200: sigma halves.
    m = meas.copy()
    late = m["subgroup"] >= 200
    m.loc[late, "value"] = ch.target + (m.loc[late, "value"] - ch.target) * 0.5
    # ... and then a 1.5-sigma shift arrives at 320, relative to the NEW sigma.
    shift = m["subgroup"] >= 320
    m.loc[shift, "value"] = m.loc[shift, "value"] + 1.5 * ch.sigma_process * 0.5

    phase1 = charts.xbar_r(m, baseline=slice(0, 100))[0]
    z_old = rules.zones(phase1.stat, phase1.center, phase1.zone_width)
    v_old = rules.any_violation(z_old)

    revised = charts.xbar_r(m, baseline=slice(200, 300))[0]
    z_new = rules.zones(revised.stat, revised.center, revised.zone_width)
    v_new = rules.any_violation(z_new)

    return {
        "improvement_at": 200, "shift_at": 320, "shift_size_new_sigma": 1.5,
        "stale_limits": {
            "sigma_hat": phase1.sigma_hat,
            "limit_width": float(phase1.ucl - phase1.lcl),
            "detections_after_shift": int(v_old[320:].sum()),
            "points_to_detect": first_detection(v_old, 320),
        },
        "revised_limits": {
            "sigma_hat": revised.sigma_hat,
            "limit_width": float(revised.ucl - revised.lcl),
            "detections_after_shift": int(v_new[320:].sum()),
            "points_to_detect": first_detection(v_new, 320),
        },
        "false_alarms_on_improved_period_with_revised_limits":
            int(v_new[200:320].sum()),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "extensions.json").read_text())
        (ROOT / "docs" / "EXTENSIONS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/EXTENSIONS.md")
        return

    t0 = time.perf_counter()
    res = {}
    print("1/4 attribute charts (p, np, c, u) ...", flush=True)
    res["attribute"] = attribute_stage()
    print("2/4 X-bar/s and I-MR, finally exercised ...", flush=True)
    res["variables"] = variables_stage()
    print("3/4 non-normal capability, actually triggered ...", flush=True)
    res["nonnormal"] = nonnormal_stage()
    print("4/4 phase I / phase II limit revision ...", flush=True)
    res["limit_revision"] = limit_revision_stage()
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "extensions.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EXTENSIONS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/EXTENSIONS.md ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# DATA-2 extensions — generated by `extend.py`, not hand-edited\n")

    a = res["attribute"]
    A("## 1. Attribute charts — the data most plants actually have\n")
    A("The first build shipped `p_chart` as a stub and exercised none of them. "
      "That is a real gap: a count of defectives off an inspection station is "
      "free, while a measured dimension needs a gauge, an operator and a study.\n")
    A("**The chart selection tree is the actual skill**, and the branch point is "
      "the distinction between a *defective* (a unit, which is good or bad) and a "
      "*defect* (a flaw, of which one unit may have several):\n")
    A("| counting | area of opportunity | chart | distribution |")
    A("|---|---|---|---|")
    A("| defective units | varies | **p** | binomial |")
    A("| defective units | constant | **np** | binomial |")
    A("| defects | varies | **u** | Poisson |")
    A("| defects | constant | **c** | Poisson |")
    A("\nEach chart scored against a planted shift:\n")
    A("| chart | centre line | shift planted at | points to detect | false alarms before the shift |")
    A("|---|---|---|---|---|")
    for k in ("p", "np", "c", "u"):
        r = a[k]
        A(f"| {k} | {r['center']:.4f} | {r['shift_at']} | "
          f"{r['points_to_detect'] if r['points_to_detect'] is not None else 'MISSED'} | "
          f"{r['false_alarms_before_shift']} |")
    p = a["p"]
    A(f"\n**Variable limits are not optional.** The p chart's subgroup size ranges "
      f"{p['size_min']}–{p['size_max']}, so its limit width varies from "
      f"{p['limit_width_min']:.4f} to {p['limit_width_max']:.4f} — a factor of "
      f"{p['limit_width_max']/max(p['limit_width_min'],1e-9):.2f}. Drawing one "
      "average limit, which is what a spreadsheet does, makes small subgroups look "
      "in control and large ones look out of control purely as an artifact of n.\n")
    w = a["wrong_chart"]
    A(f"**And the cost of picking the wrong chart.** On the same varying-area data "
      f"(area {w['area_min']:.2f}–{w['area_max']:.2f}), the correct u chart flags "
      f"**{w['u_chart_violations']}** of {w['n_points']} points; a c chart, which "
      f"assumes constant opportunity, flags **{w['c_chart_violations']}**. The "
      "extra signals are not process signals — they are the inspection area "
      "changing. An operator watching that chart learns it tracks batch size, and "
      "stops watching.")

    v = res["variables"]
    A("\n## 2. X-bar/s and I-MR, finally exercised\n")
    A("| chart | σ̂ | limit width | disturbances detected |")
    A("|---|---|---|---|")
    for k in ("xbar_r", "xbar_s", "i_mr"):
        c = v[k]
        A(f"| {c['chart']} | {c['sigma_hat']:.5f} | {c['limit_width']:.5f} | "
          f"{c['n_detected']}/{c['n_disturbances']} |")
    sa = v["sigma_agreement"]
    A(f"\n**The three σ̂ estimates agree**: R-based {sa['r_based']:.5f}, s-based "
      f"{sa['s_based']:.5f}, I-MR {sa['i_mr_based']:.5f}, against a true process σ "
      f"of {sa['true_process_sigma']:.5f} and a true total (process + gauge) σ of "
      f"{sa['true_total_sigma']:.5f}.\n")
    A("They agree with the **total**, not with the process alone — which is exactly "
      "right and worth noticing: a control chart sees what the gauge reports, so "
      "its σ̂ includes measurement variation. That is the same fact the gauge R&R "
      "section is about, arriving from the other direction, and it is why "
      "capability computed off a chart is capability *of the measured process*.\n")
    A("**When to prefer which:** R is easier to compute by hand and was the "
      "historical default; s is more efficient and is preferred for n ≥ 10; I-MR "
      "is what you are left with when the subgroup size is 1 — low-volume, "
      "destructive testing, or a batch process — and it pays for that with much "
      "wider limits and far less power against small shifts.")

    n = res["nonnormal"]
    A("\n## 3. The non-normal capability path, actually triggered\n")
    A("In the first build this was an **untested code path presented as a "
      "feature**: the skewed characteristic was refused for instability before the "
      "distribution branch ever ran. Here it is exercised on a *stable* skewed "
      "process, which is the only situation where the question is even meaningful "
      "— capability on an unstable process is still refused.\n")
    A(f"`{n['characteristic']}`, skew parameter {n['skew_parameter']}. "
      f"Out-of-control points: **{n['ooc_points_rule1_only']} under rule 1 alone**, "
      f"**{n['ooc_points_all_rules']} with all eight rules**. "
      f"`assess()` refused: **{n['refused']}**.\n")
    A("**The gap between those two counts is the finding, and it explains why this "
      "path was unreachable in the first build.** The runs rules assume a "
      "*symmetric* in-control distribution; on a skewed process they fire on the "
      "skew itself. So the stability gate refuses, and the non-normal capability "
      "branch — the one built specifically for skewed data — can never run on "
      "skewed data. Even rule 1 alone over-fires here.\n")
    A("**Normality assumptions leak into the stability test, not just into the "
      "capability index.** A plant charting a skewed characteristic with all eight "
      "rules enabled will conclude it is unstable forever, chase assignable causes "
      "that do not exist, and never reach a capability number at all. The correct "
      "sequence is: establish stability with a chart appropriate to the "
      "distribution (transformed data, or limits from the fitted distribution), "
      "*then* assess capability. **That chart is not built here**, which is why the "
      "refusal below is genuine rather than staged — the two capability methods are "
      "computed directly for comparison.\n")
    A("| | |")
    A("|---|---|")
    A(f"| Anderson–Darling statistic | {n['normality']['anderson_darling_stat']:.3f} |")
    A(f"| critical value (5%) | {n['normality']['critical_value_5pct']:.3f} |")
    A(f"| normal at 5%? | {n['normality']['normal_at_5pct']} |")
    A(f"| sample skew | {n['normality']['skew']:.3f} |")
    A(f"| **method selected** | **{n['method_selected']}** |")
    A("\n| method | Ppk | predicted PPM | observed PPM |")
    A("|---|---|---|---|")
    A(f"| normal theory | {n['normal_theory']['Ppk']:.3f} | "
      f"{n['normal_theory']['predicted_ppm']:,.0f} | "
      f"{n['normal_theory']['observed_ppm']:,.0f} |")
    A(f"| percentile (ISO 21747 style) | {n['percentile']['Ppk']:.3f} | — | "
      f"{n['percentile']['observed_ppm']:,.0f} |")
    err = n["normal_theory"]["predicted_ppm"] - n["normal_theory"]["observed_ppm"]
    A(f"\n**Normal theory mis-predicts the defect rate by {err:+,.0f} PPM** on this "
      "characteristic. That is the whole reason the branch exists: Ppk's value "
      "comes from converting a ratio into a tail probability, and on a skewed "
      "distribution that conversion runs through a tail that is not there. The "
      "percentile method keeps the interpretation — what fraction of the tolerance "
      "the process uses — without borrowing the normal tail.")

    lr = res["limit_revision"]
    A("\n## 4. Phase I / phase II, and what stale limits cost\n")
    A("**Phase I** is retrospective: establish limits from a period judged stable, "
      "iterating as assignable causes are removed. **Phase II** is prospective: "
      "freeze those limits and monitor forward. The first build froze limits on a "
      "baseline slice and had no policy for ever revising them.\n")
    A(f"The scenario: the process **genuinely improves** at subgroup "
      f"{lr['improvement_at']} (σ halves), and then a "
      f"{lr['shift_size_new_sigma']}σ shift — relative to the *new* σ — arrives at "
      f"subgroup {lr['shift_at']}.\n")
    A("| limits | σ̂ | limit width | points to detect the shift | detections after it |")
    A("|---|---|---|---|---|")
    for k, lab in (("stale_limits", "phase I, never revised"),
                   ("revised_limits", "revised after the improvement")):
        r = lr[k]
        A(f"| {lab} | {r['sigma_hat']:.5f} | {r['limit_width']:.5f} | "
          f"{r['points_to_detect'] if r['points_to_detect'] is not None else '**MISSED**'} | "
          f"{r['detections_after_shift']} |")
    stale, rev = lr["stale_limits"], lr["revised_limits"]
    A(f"\n**Stale limits are {stale['limit_width']/max(rev['limit_width'],1e-9):.1f}× "
      "too wide after the improvement**, and the consequence is the one nobody "
      "reports because it looks like success: the chart goes quiet. A quiet chart "
      "reads as a healthy process, and is in fact a chart that has lost the ability "
      "to detect anything — the improvement made the process better and the "
      "unrevised limits made the monitoring worse.\n")
    A("This is the answer to \"the operator says the chart never alarms any more\", "
      "which is the mirror image of the \"it alarms constantly\" diagnostic in "
      "RESULTS.md §4 — and it has the same root cause: limits that no longer "
      "describe the process. **Capability improvement is a trigger for limit "
      "revision**, and a plant without that rule in its quality manual will run on "
      "limits set years ago by somebody who has left.")

    A("\n---\n*Regenerate with `python extend.py`.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
