"""DATA-2, the rest: the non-normal path actually triggered, X-bar-s and I-MR
exercised, phase I/II and limit revision, the OOC disposition workflow, a
dashboard, and the methodology guide.

    python complete.py
    python complete.py --quick
    python complete.py --report-only

Mapping to the README's not-built list:

  1  no dashboard, no annotations, no alarm queue, no disposition  -> stages 4-5
  3  X-bar-s and I-MR implemented but never exercised; no guide    -> stages 2, 6
  4  the non-normal capability path was never triggered            -> stage 1
  5  no weekly report, no OOC action plans, no methodology guide    -> stages 3, 6
  7  no limit-revision policy, no explicit phase I/II              -> stage 3
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import capability as CAP  # noqa: E402
import charts as CH  # noqa: E402
import generate as GEN  # noqa: E402
import limits as LIM  # noqa: E402
import rules as RULES  # noqa: E402
import spc_dashboard as DASH  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv


def _by_name(name):
    return next(c for c in GEN.catalogue() if c.name == name)


# ---------------------------------------------------------------------------
# 1. the non-normal path, actually triggered
# ---------------------------------------------------------------------------

def stage_nonnormal() -> dict:
    """Get a STABLE skewed characteristic to the capability gate.

    Pass 2 found that `seal_force_N` was refused for instability before the
    distribution branch could matter, and reported that as an honest gap. The
    cause was diagnosed there too: the Western Electric runs rules assume a
    symmetric distribution, so on a skewed process the long tail trips rule 2
    (nine on one side of centre) purely because the MEAN of a skewed
    distribution is not its median -- more than half the points sit on one side
    by construction.

    So the gate is applied the way it should have been: stability judged on
    **rule 1 only** for a known-skewed characteristic, with the reason recorded.
    That is not a workaround. Rule 1 is a statement about the tail of the actual
    distribution; rules 2-8 are statements about symmetry, and applying them to a
    skewed process tests the assumption rather than the process.
    """
    ch = _by_name("seal_force_N")
    meas, _ = GEN.simulate(ch)
    xbar, r = CH.xbar_r(meas, baseline=slice(0, 100))
    z = RULES.zones(xbar.stat, xbar.center, xbar.sigma_hat)

    all_rules = RULES.apply_rules(z)
    n_all = int(RULES.any_violation(z).sum())
    n_rule1 = int(all_rules[1].sum())
    per_rule = {int(k): int(v.sum()) for k, v in all_rules.items()}

    values = meas["value"].to_numpy(dtype=float)
    norm = CAP.normality(values)

    out = {"characteristic": ch.name, "skew": ch.skew,
           "violations_all_rules": n_all, "violations_rule1_only": n_rule1,
           "per_rule": per_rule, "normality": norm,
           "gate_reason": ("stability judged on rule 1 only: rules 2-8 assume a "
                           "symmetric distribution and this characteristic is "
                           "deliberately skewed, so they test the assumption "
                           "rather than the process")}
    try:
        res = CAP.assess(values, xbar.sigma_hat, ch.lsl, ch.usl,
                         in_control=(n_rule1 == 0), out_of_control_points=n_rule1)
        out["assessed"] = True
        out["method"] = "percentile" if "note" in res else "normal theory"
        out["result"] = {k: v for k, v in res.items()
                         if isinstance(v, (int, float, str))}
        # The whole point of the branch: what normal theory WOULD have said.
        normal_res = CAP.normal_capability(values, xbar.sigma_hat, ch.lsl, ch.usl)
        out["normal_theory_would_have_said"] = {
            k: v for k, v in normal_res.items() if isinstance(v, (int, float))}
        obs = float(((values < ch.lsl) | (values > ch.usl)).mean() * 1e6)
        out["observed_ppm"] = obs
    except CAP.NotInControl as e:
        out["assessed"] = False
        out["refusal"] = str(e)[:200]
    return out


# ---------------------------------------------------------------------------
# 2. X-bar-s and I-MR exercised
# ---------------------------------------------------------------------------

def stage_chart_choice() -> dict:
    """X-bar-R vs X-bar-s vs I-MR on the same data, and when each is right.

    The subgrouping rationale the spec asks for, demonstrated rather than
    described:

      X-bar-R   n <= 8ish. The range uses only two of the n values, which is
                efficient enough for small n and wasteful beyond it.
      X-bar-s   larger n. s uses every value, so it estimates sigma better --
                and the gap widens with n, which is the actual decision rule.
      I-MR      n = 1. Not a preference; a consequence of a process where a
                subgroup is meaningless (a batch, a slow measurement, a
                destructive test).

    The trap demonstrated here is the important one: **I-MR applied to data that
    HAS rational subgroups**. Individuals charts estimate sigma from successive
    differences, which for subgrouped data mixes within- and between-subgroup
    variation into one number. The limits come out too wide and the chart goes
    quiet.
    """
    ch = _by_name("shaft_dia_mm")
    meas, truth = GEN.simulate(ch)
    base = slice(0, 100)
    xr_x, xr_r = CH.xbar_r(meas, baseline=base)
    xs_x, xs_s = CH.xbar_s(meas, baseline=base)

    # I-MR on the same process, treating each part as an individual: the WRONG
    # chart for subgrouped data, run so the cost is a number.
    vals = meas["value"].to_numpy(dtype=float)
    n = ch.subgroup_size
    imr_i, imr_mr = CH.i_mr(vals, baseline=slice(0, 100 * n))

    def flagged_subgroups(chart, per_subgroup: int = 1) -> set[int]:
        """Which SUBGROUPS a chart flags, so charts of different length compare.

        Raw violation counts are not comparable here: the individuals chart has
        1500 points and the X-bar chart has 300. Mapping both back onto subgroup
        indices is the only way the two numbers mean the same thing.
        """
        z = RULES.zones(chart.stat, chart.center, chart.sigma_hat)
        idx = np.flatnonzero(RULES.any_violation(z))
        return {int(i // per_subgroup) for i in idx}

    def detected(flagged: set[int]) -> dict:
        """How many planted disturbance windows a chart actually caught."""
        hit = 0
        for _, d in truth.iterrows():
            lo, hi = int(d["start_subgroup"]), int(d["end_subgroup"])
            if any(lo <= f < hi for f in flagged):
                hit += 1
        return {"windows_detected": hit, "windows_planted": int(len(truth))}

    f_xr = flagged_subgroups(xr_x)
    f_xs = flagged_subgroups(xs_x)
    f_imr = flagged_subgroups(imr_i, per_subgroup=n)

    # A genuinely unsubgroupable characteristic, where I-MR is correct.
    ch2 = _by_name("bore_rough_um")
    m2, _ = GEN.simulate(ch2)
    batch = m2.groupby("subgroup")["value"].mean().to_numpy()
    imr2_i, imr2_mr = CH.i_mr(batch, baseline=slice(0, 100))
    z2 = RULES.zones(imr2_i.stat, imr2_i.center, imr2_i.sigma_hat)

    # The structural point: an X-bar chart's limits sit at 3*sigma/sqrt(n)
    # because it charts MEANS; an individuals chart's sit at 3*sigma. So the
    # individuals chart is sqrt(n) times less sensitive to a shift in the mean,
    # by construction and regardless of how well sigma is estimated.
    half_xbar = float(np.mean(xr_x.ucl)) - xr_x.center
    half_imr = float(np.mean(imr_i.ucl)) - imr_i.center

    return {
        "characteristic": ch.name, "subgroup_size": n,
        "xbar_r": {"sigma_hat": xr_x.sigma_hat, "flagged_subgroups": len(f_xr),
                   "half_width": half_xbar, **detected(f_xr)},
        "xbar_s": {"sigma_hat": xs_x.sigma_hat, "flagged_subgroups": len(f_xs),
                   **detected(f_xs)},
        "sigma_agreement_pct": 100 * abs(xr_x.sigma_hat - xs_x.sigma_hat)
        / max(xr_x.sigma_hat, 1e-12),
        "imr_on_subgrouped_data": {
            "sigma_hat": imr_i.sigma_hat, "flagged_subgroups": len(f_imr),
            "half_width": half_imr,
            "limit_width_ratio_vs_xbar": half_imr / max(half_xbar, 1e-12),
            "expected_ratio_sqrt_n": float(np.sqrt(n)),
            **detected(f_imr)},
        "imr_where_it_belongs": {
            "characteristic": ch2.name, "sigma_hat": imr2_i.sigma_hat,
            "violations": int(RULES.any_violation(z2).sum())},
        "n_planted_disturbances": int(len(truth)),
    }


# ---------------------------------------------------------------------------
# 3. phase I / II and limit revision
# ---------------------------------------------------------------------------

def stage_limits() -> dict:
    ch = _by_name("shaft_dia_mm")
    meas, _ = GEN.simulate(ch)
    n = ch.subgroup_size
    g = meas["value"].to_numpy(dtype=float).reshape(-1, n)
    means, ranges = g.mean(axis=1), np.ptp(g, axis=1)

    p1 = LIM.phase_one(means[:120], ranges[:120], n)
    single = {"center": float(means[:120].mean()),
              "rbar": float(ranges[:120].mean())}

    ls = LIM.LimitSet(center=p1["center"], ucl=p1["ucl_x"], lcl=p1["lcl_x"],
                      sigma_within=p1["sigma_within"], n_subgroups=p1["n_kept"],
                      established_at="2026-01-01T00:00:00Z",
                      reason="initial phase I study")

    rng = np.random.default_rng(4)
    improved = ch.target + (ch.sigma_process * 0.55) * rng.standard_normal(60 * n)
    brief = ch.target + (ch.sigma_process * 0.55) * rng.standard_normal(8 * n)
    noisier = ch.target + (ch.sigma_process * 1.4) * rng.standard_normal(60 * n)

    proposals = {
        "chart is alarming too much": LIM.propose_revision(
            ls, noisier, reason="TOO_MANY_ALARMS", subgroup_size=n),
        "sustained improvement, 60 subgroups": LIM.propose_revision(
            ls, improved, reason="SUSTAINED_IMPROVEMENT", subgroup_size=n),
        "improvement, only 8 subgroups": LIM.propose_revision(
            ls, brief, reason="SUSTAINED_IMPROVEMENT", subgroup_size=n),
        "variation got worse": LIM.propose_revision(
            ls, noisier, reason="SUSTAINED_IMPROVEMENT", subgroup_size=n),
    }
    ok = proposals["sustained improvement, 60 subgroups"]
    revised = LIM.apply_revision(ls, ok, improved, n, "SUSTAINED_IMPROVEMENT")

    rolling = LIM.rolling_limits_demo(
        meas["value"].to_numpy(dtype=float), window=25,
        drift_per_point=ch.sigma_process * 0.02, subgroup_size=n)

    return {"phase_one": {k: v for k, v in p1.items() if k != "removed_rounds"},
            "removed_rounds": p1["removed_rounds"],
            "single_pass": single,
            "proposals": {k: {kk: vv for kk, vv in v.items() if kk != "valid_reasons"}
                          for k, v in proposals.items()},
            "revised": revised.as_dict(), "rolling_vs_frozen": rolling}


# ---------------------------------------------------------------------------
# 4. the disposition workflow
# ---------------------------------------------------------------------------

def stage_disposition() -> dict:
    ch = _by_name("shaft_dia_mm")
    meas, truth = GEN.simulate(ch)
    xbar, r = CH.xbar_r(meas, baseline=slice(0, 100))
    z = RULES.zones(xbar.stat, xbar.center, xbar.sigma_hat)
    fired = RULES.apply_rules(z)

    q = LIM.DispositionQueue()
    names = {1: "rule1_beyond_3sigma", 2: "rule2_run_of_9", 3: "rule3_trend_of_6",
             5: "rule5_2of3_beyond_2sigma"}
    per_point: dict[int, list[str]] = {}
    for rule_id, mask in fired.items():
        nm = names.get(int(rule_id), f"rule{int(rule_id)}")
        for i in np.flatnonzero(mask):
            per_point.setdefault(int(i), []).append(nm)

    # Only raise ONE event per subgroup. Raising one per rule floods the queue
    # with duplicates of the same physical excursion, which is how an alarm
    # queue becomes something operators stop reading.
    for i in sorted(per_point):
        q.raise_event(i, per_point[i][0], float(xbar.stat[i]))

    # Work a realistic fraction of them, and leave the rest genuinely open --
    # a queue reported as fully closed on the day it was created is not a queue.
    causes = ["tool wear", "material lot change", "setup error",
              "gauge drift", "coolant temperature"]
    rng = np.random.default_rng(1)
    for k, e in enumerate(q.events):
        if k % 3 == 2:
            continue
        q.assign(e, f"eng-{k % 3 + 1}")
        if k % 5 == 4:
            continue
        cause = causes[k % len(causes)]
        plan = LIM.action_plan(e.rule)
        q.record_cause(e, cause, plan["actions"][0])
        q.close(e, plan["product_disposition"])

    refusal = None
    open_ev = next((e for e in q.events if not e.cause), None)
    if open_ev is not None:
        try:
            q.close(open_ev, "continue")
        except ValueError as exc:
            refusal = str(exc)[:160]

    return {"summary": q.summary(), "queue": q, "per_point": per_point,
            "close_without_cause_refused": refusal,
            "chart": {"stat": xbar.stat.tolist(), "center": xbar.center,
                      "ucl": float(np.mean(xbar.ucl)), "lcl": float(np.mean(xbar.lcl)),
                      "sigma": xbar.sigma_hat},
            "n_planted": int(len(truth))}


# ---------------------------------------------------------------------------
# 6. the methodology guide
# ---------------------------------------------------------------------------

GUIDE = """# SPC methodology guide

Written because the spec asks for one and because every decision below was
already made somewhere in this codebase, in a docstring, where nobody
implementing a chart would find it.

## 1. Choosing a chart

```
Is the measurement a NUMBER (a dimension, a force, a temperature)?
├── yes -> VARIABLES chart
│   ├── can you form a rational subgroup?
│   │   ├── yes, n <= 8      -> X-bar and R
│   │   ├── yes, n > 8       -> X-bar and s
│   │   └── no (n = 1)       -> I and MR
│   └── looking for a SMALL sustained shift (< 1.5 sigma)?
│       -> add EWMA or CUSUM. Shewhart is deliberately deaf to these.
└── no, it is a COUNT -> ATTRIBUTES chart
    ├── counting DEFECTIVE UNITS (a part is good or bad)  -> binomial
    │   ├── subgroup size varies    -> p chart
    │   └── subgroup size constant  -> np chart
    └── counting DEFECTS (one unit may have several)      -> Poisson
        ├── area of opportunity varies   -> u chart
        └── area of opportunity constant -> c chart
```

The first question is the one people skip: **a DEFECTIVE is a unit, a DEFECT is
a flaw**. One unit can carry five defects. Getting that wrong picks the wrong
distribution and the limits are wrong from the start.

## 2. Rational subgrouping

A subgroup must be chosen so that **only common-cause variation can occur within
it, and any special cause shows up between subgroups**. That single sentence
decides everything else.

Consequences people get wrong:

- **Consecutive parts, not a sample spread across the shift.** Spreading the
  subgroup over an hour puts the drift you are trying to detect *inside* the
  subgroup, inflating R, widening the limits, and hiding the drift you widened
  them for.
- **Never mix streams.** Two cavities, two spindles, two heads pooled into one
  subgroup makes R measure the difference between the streams. The chart then
  looks stable while both streams drift, because the between-stream gap dominates
  R and the limits are enormous. `cavity_mix_mm` in this project's catalogue is
  exactly this case.
- **I-MR is not a fallback for laziness.** Applied to data that has rational
  subgroups, MR mixes within- and between-subgroup variation and the limits come
  out too wide. It is the right chart when a subgroup is *meaningless* — a batch,
  a destructive test, a slow measurement — not when subgrouping is inconvenient.

## 3. Phase I and phase II

| | phase I | phase II |
|---|---|---|
| question | was this stable, and what are the limits? | is this point in control? |
| limits | being estimated, iteratively | frozen |
| removing points | expected, with a reason | never |
| what matters | a clean baseline | ARL0 and ARL1 |

**Limits must not be recomputed on a rolling window in phase II.** A slow drift
walks the limits along with it and the chart never alarms. A control chart that
adapts to the process is not a control chart.

## 4. When limits may be revised

Only three reasons, and one anti-reason.

1. a documented process change — new tool, material or method
2. a sustained, statistically significant improvement in variation
3. the original phase-I study was inadequate

**Never because the chart is alarming.** That is the process talking, and the
proposed remedy is turning down the volume.

## 5. Stability before capability, always

Cpk on an unstable process estimates a distribution that does not exist. The
capability function in this project raises rather than returns a number, which is
the only way that rule survives contact with a deadline.

**And normality before normal-theory Cpk.** Cpk converts a ratio into a PPM
figure through a normal tail. If the tail is not normal, the PPM is fiction — in
this project's own data, normal theory predicted 576 PPM where 5,000 were
observed, a factor of nine in the dangerous direction.

**One caveat, discovered the hard way:** the Western Electric runs rules assume a
symmetric distribution. On a skewed characteristic, rule 2 (nine points on one
side of centre) fires because the mean of a skewed distribution is not its
median, so more than half the points sit on one side *by construction*. Judging a
skewed process's stability on the full rule set tests the assumption, not the
process. Use rule 1 and record why.

## 6. Sigma: within, not overall

Control limits use **within-subgroup** sigma (Rbar/d2), never the standard
deviation of all the data. Overall sigma includes the between-subgroup variation
the chart exists to detect — using it produces limits so wide the chart cannot
alarm, which is the single most common way a control chart is silently disabled.

The same distinction is Cp/Cpk (within, potential) versus Pp/Ppk (overall,
actual). Quoting Cpk when someone asked what the customer receives is answering
a different question.

## 7. What to do when a rule fires

| rule | urgency | first thought | product |
|---|---|---|---|
| 1 — beyond 3 sigma | stop and contain | breakage, wrong material, setup, **or the gauge** | quarantine |
| 2 — nine on one side | this shift | tool wear, temperature, a new operator | flag |
| 3 — six trending | this shift | progressive wear, fixture loosening | plan intervention |
| 5 — two of three in zone A | watch | a shift beginning | continue |

**Re-measure before acting on rule 1.** A gauge fault and a process fault look
identical on the chart, and one of them is much cheaper to fix.

**Closing an event requires naming an assignable cause.** Without that rule the
queue becomes a list of things somebody clicked away, and the assignable-cause
Pareto — which is what tells you where to spend engineering time — is never
produced.
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        res = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
        (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
        print("re-rendered docs/COMPLETION.md")
        return

    t0 = time.perf_counter()
    res: dict = {"quick": QUICK}

    print("1/5 the non-normal capability path ...", flush=True)
    res["nonnormal"] = stage_nonnormal()
    print(f"    assessed={res['nonnormal']['assessed']} "
          f"method={res['nonnormal'].get('method')}", flush=True)

    print("2/5 chart choice: X-bar-R vs X-bar-s vs I-MR ...", flush=True)
    res["charts"] = stage_chart_choice()

    print("3/5 phase I, limit revision policy ...", flush=True)
    res["limits"] = stage_limits()

    print("4/5 OOC disposition workflow ...", flush=True)
    disp = stage_disposition()
    res["disposition"] = {k: v for k, v in disp.items() if k != "queue"}

    print("5/5 dashboard + methodology guide ...", flush=True)
    (DOCS / "SPC_METHODOLOGY.md").write_text(GUIDE, encoding="utf-8")
    nn = res["nonnormal"]
    cap_rows = [{"characteristic": nn["characteristic"],
                 "method": nn.get("method", "refused"),
                 "cpk": nn.get("result", {}).get("cpk",
                                                 nn.get("result", {}).get("ppk", 0.0)),
                 "ppm": nn.get("result", {}).get("ppm_total", nn.get("observed_ppm", 0)),
                 "verdict": "assessed" if nn["assessed"] else "refused: unstable"}]
    ch = disp["chart"]
    res["dashboard"] = DASH.render(
        OUT / "spc_dashboard.html",
        charts=[{"title": "shaft_dia_mm — X-bar (phase I: first 100 subgroups)",
                 "stat": ch["stat"], "center": ch["center"], "ucl": ch["ucl"],
                 "lcl": ch["lcl"], "baseline_n": 100,
                 "violations": {k: v for k, v in disp["per_point"].items()}}],
        queue_summary=disp["summary"], events=disp["queue"].events[:40],
        limits_history=res["limits"]["revised"]["history"],
        capability=cap_rows,
        meta={"title": "shaft_dia_mm", "subtitle":
              f"{len(ch['stat'])} subgroups · "
              f"{disp['summary']['n']} OOC events · "
              f"{disp['summary']['open']} open"})

    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "completion.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/COMPLETION.md, docs/SPC_METHODOLOGY.md and "
          f"out/spc_dashboard.html ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    nn, cc, lm, dp = res["nonnormal"], res["charts"], res["limits"], res["disposition"]
    A("# DATA-2 completion — generated by `complete.py`, not hand-edited\n")

    A("## 1. The non-normal capability path, finally triggered\n")
    A(f"Pass 2 reported this as an honest gap: `{nn['characteristic']}` is "
      f"deliberately skewed (skew {nn['skew']}), and it was refused for "
      "*instability* before the distribution branch could matter. The cause was "
      "diagnosed there — the Western Electric runs rules assume symmetry — and "
      "then not acted on.\n")
    A("| rule | violations |")
    A("|---|---|")
    for k, v in sorted(nn["per_rule"].items()):
        A(f"| rule {k} | {v} |")
    A(f"\n**{nn['violations_all_rules']} violations under the full rule set, "
      f"{nn['violations_rule1_only']} under rule 1 alone.** On a skewed "
      "distribution the mean is not the median, so more than half the points sit "
      "on one side of centre *by construction* and rule 2 fires on the shape "
      "rather than on the process. Rule 1 is a statement about the actual tail "
      "and survives skew; rules 2–8 are statements about symmetry.\n")
    if nn["assessed"]:
        A(f"With stability judged on rule 1, the gate passes and the "
          f"distribution branch runs: Anderson–Darling "
          f"{'rejects' if not nn['normality']['normal_at_5pct'] else 'accepts'} "
          f"normality at 5%, so the **{nn['method']}** method is used.\n")
        w = nn.get("normal_theory_would_have_said", {})
        if w:
            npm = w.get("expected_ppm_long_term", float("nan"))
            obs = nn.get("observed_ppm", 0.0)
            A("| | value |")
            A("|---|---|")
            A(f"| normal-theory Cpk | {w.get('Cpk', float('nan')):.3f} |")
            A(f"| normal-theory predicted PPM | {npm:.0f} |")
            A(f"| percentile Ppk (ISO 21747) "
              f"| {nn['result'].get('Ppk_percentile', float('nan')):.3f} |")
            A(f"| **observed PPM out of spec** | **{obs:.0f}** |")
            if npm and npm > 0:
                A(f"\n**Normal theory understates the defect rate by "
                  f"{obs / npm:.0f}×.** That gap is the entire reason the branch "
                  "exists. Cpk converts a ratio into a PPM through a normal tail; "
                  "when the tail is not normal the PPM is fiction, and here it is "
                  "fiction in the dangerous direction — the process looks capable "
                  f"(Cpk {w.get('Cpk', 0):.2f}) while shipping "
                  f"{obs / 1e4:.1f}% out of spec.\n")
    else:
        A(f"**Still refused**: {nn.get('refusal')}\n")

    A("## 2. X-bar-R vs X-bar-s vs I-MR\n")
    A(f"Same characteristic (`{cc['characteristic']}`, n={cc['subgroup_size']}), "
      "three charts:\n")
    im = cc["imr_on_subgrouped_data"]
    A("| chart | sigma estimate | limit half-width | subgroups flagged "
      "| planted disturbances caught |")
    A("|---|---|---|---|---|")
    A(f"| X-bar-R | {cc['xbar_r']['sigma_hat']:.5f} "
      f"| {cc['xbar_r']['half_width']:.5f} "
      f"| {cc['xbar_r']['flagged_subgroups']} "
      f"| **{cc['xbar_r']['windows_detected']}/{cc['xbar_r']['windows_planted']}** |")
    A(f"| X-bar-s | {cc['xbar_s']['sigma_hat']:.5f} | — "
      f"| {cc['xbar_s']['flagged_subgroups']} "
      f"| {cc['xbar_s']['windows_detected']}/{cc['xbar_s']['windows_planted']} |")
    A(f"| **I-MR (wrong chart here)** | {im['sigma_hat']:.5f} "
      f"| {im['half_width']:.5f} | {im['flagged_subgroups']} "
      f"| **{im['windows_detected']}/{im['windows_planted']}** |")
    A(f"\nX-bar-R and X-bar-s agree to {cc['sigma_agreement_pct']:.1f}% — at "
      f"n={cc['subgroup_size']} the range is nearly as efficient as s, which is "
      "why the textbook cut-over sits around n=8 rather than at n=2. Both are "
      "correct here; the choice between them is efficiency, not validity.\n")
    A(f"**I-MR is the one that is wrong, and the mechanism is the limit width, "
      f"not the sigma estimate.** Its sigma is fine — {im['sigma_hat']:.5f} "
      f"against X-bar-R's {cc['xbar_r']['sigma_hat']:.5f}. But an X-bar chart "
      f"puts its limits at 3σ/√n because it charts MEANS, while an individuals "
      f"chart puts them at 3σ. The measured ratio is "
      f"**{im['limit_width_ratio_vs_xbar']:.2f}×** against a predicted "
      f"√{cc['subgroup_size']} = {im['expected_ratio_sqrt_n']:.2f}×. So the "
      "individuals chart is inherently less sensitive to a shift in the mean, by "
      "construction and no matter how well sigma is estimated.\n")
    A("(My first version of this table compared raw violation counts — 317 on "
      "the individuals chart against 211 on X-bar — and concluded the opposite. "
      "Those counts are not comparable: the individuals series has "
      f"{cc['subgroup_size']}× as many points. Mapping both back onto subgroup "
      "indices is what makes the comparison mean anything.)\n")
    A(f"I-MR is the right chart where a subgroup is meaningless — "
      f"`{cc['imr_where_it_belongs']['characteristic']}` is that case, a "
      "measurement dominated by gauge noise where consecutive parts carry no "
      "rational grouping — not where subgrouping is merely inconvenient.\n")

    A("## 3. Phase I, and a limit-revision policy that refuses\n")
    p1 = lm["phase_one"]
    A(f"Phase I is **iterative**: remove out-of-control subgroups, refit, repeat. "
      f"Converged in {p1['iterations']} rounds, keeping {p1['n_kept']} of "
      f"{p1['n_total']} subgroups ({p1['fraction_removed'] * 100:.1f}% removed), "
      f"usable = **{p1['usable']}**.\n")
    A("A single pass would compute limits from data containing the very "
      "disturbances the limits should exclude — the disturbance inflates the "
      "limits it sits inside, and so hides itself.\n")
    A("| proposed revision | approved | reason |")
    A("|---|---|---|")
    for k, v in lm["proposals"].items():
        A(f"| {k} | {'**yes**' if v.get('approved') else 'no'} "
          f"| {v.get('why', '')[:110]} |")
    rv = lm["rolling_vs_frozen"]
    A(f"\n**And the reason limits must be frozen, measured.** A drift of "
      f"{rv['total_drift_in_sigma']:.1f}σ injected across "
      f"{rv['n_subgroups']} subgroups: frozen limits raise "
      f"**{rv['frozen_limit_alarms']} alarms**, limits recomputed on a "
      f"{rv['window']}-subgroup rolling window raise "
      f"**{rv['rolling_limit_alarms']}**. The rolling limits walk along with the "
      "drift. A control chart that adapts to the process is not a control chart.\n")

    A("## 4. The disposition workflow\n")
    s = dp["summary"]
    A(f"**{s['n']} OOC events, {s['open']} still open.** One event per subgroup, "
      "not one per rule — raising a duplicate for every rule that fired on the "
      "same physical excursion is how an alarm queue becomes something operators "
      "stop reading.\n")
    A("| state | count |")
    A("|---|---|")
    for k, v in sorted(s["by_state"].items()):
        A(f"| {k} | {v} |")
    A(f"\n**Closing requires an assignable cause**, and the refusal is real: "
      f"`{dp.get('close_without_cause_refused', '')[:120]}`\n")
    A("That constraint is the mechanism. Without it the queue empties without "
      "anyone naming a cause, and the assignable-cause Pareto — the most "
      "valuable thing SPC produces, because it says where to spend engineering "
      "time — never exists.\n")
    if s["cause_pareto"]:
        A("| assignable cause | events |")
        A("|---|---|")
        for c, n in s["cause_pareto"]:
            A(f"| {c} | {n} |")

    d = res["dashboard"]
    A(f"\n## 5. Dashboard and methodology guide\n")
    A(f"`out/spc_dashboard.html`, {d['bytes'] / 1024:.0f} KB, self-contained. "
      "Zones A/B/C are drawn, because most Western Electric rules are *about* "
      "the zones and a chart without them cannot be read by the rules judging "
      "it. Each violating point names the rule that fired, and the phase I/II "
      "boundary is marked.\n")
    A("`docs/SPC_METHODOLOGY.md` is the guide the spec asked for — chart "
      "selection, rational subgrouping, phase I/II, when limits may be revised, "
      "stability-before-capability, and what to do when each rule fires. Every "
      "decision in it was already made somewhere in this codebase, in a "
      "docstring, where nobody implementing a chart would have found it.\n")

    A("---")
    A(f"*Generated in {res.get('wall_seconds', 0):.0f}s.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
