"""DATA-2's four remaining items: the skew FIX, gauge-corrected capability, a
persistent disposition queue, and the weekly report.

    python run_weekly.py
    python run_weekly.py --report-only

The README named all four as not built. Three were named with the fix already
described — "the right answer is a transformation or a distribution-appropriate
chart", "nothing subtracts measurement variation", "no persistence" — which is
reporting rather than doing.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import capability as CAP  # noqa: E402
import charts as CH  # noqa: E402
import gauge_rr as GRR  # noqa: E402
import generate as GEN  # noqa: E402
import limits as LIM  # noqa: E402
import rules as RULES  # noqa: E402
import transforms as TR  # noqa: E402
import weekly as WK  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"


def _by_name(name):
    return next(c for c in GEN.catalogue() if c.name == name)


# ---------------------------------------------------------------------------
# 1. the skew fix
# ---------------------------------------------------------------------------

def stage_skew() -> dict:
    """Transform the skewed characteristic, and check the transformation worked."""
    ch = _by_name("seal_force_N")
    meas, _ = GEN.simulate(ch)
    v = meas["value"].to_numpy(dtype=float)

    cmp = TR.compare_transforms(v)
    lam = cmp.get("boxcox", {}).get("lambda")
    shift = cmp.get("boxcox", {}).get("shift", 0.0)
    tcap = (TR.transformed_capability(v, ch.lsl, ch.usl, lam, shift)
            if lam is not None else None)

    # What the three approaches say about the SAME data.
    xbar, _ = CH.xbar_r(meas, baseline=slice(0, 100))
    normal = CAP.normal_capability(v, xbar.sigma_hat, ch.lsl, ch.usl)
    pct = CAP.percentile_capability(v, ch.lsl, ch.usl)
    observed_ppm = float(((v < ch.lsl) | (v > ch.usl)).mean() * 1e6)

    # And the payoff: on TRANSFORMED data the runs rules become valid again,
    # because they assume symmetry. This is the real argument for transforming
    # rather than for the rule-1-only policy pass 3 used.
    rules_raw = int(RULES.any_violation(
        RULES.zones(xbar.stat, xbar.center, xbar.sigma_hat)).sum())
    n = ch.subgroup_size
    g = TR.boxcox(v, lam, shift).reshape(-1, n) if lam is not None else None
    if g is not None:
        m = g.mean(axis=1)
        r = np.ptp(g, axis=1)
        from constants import A2, d2
        base = slice(0, 100)
        sig = float(r[base].mean()) / d2(n)
        rules_tx = int(RULES.any_violation(
            RULES.zones(m, float(m[base].mean()), sig)).sum())
    else:
        rules_tx = None

    return {
        "characteristic": ch.name, "skew_parameter": ch.skew,
        "comparison": cmp, "transformed_capability": tcap,
        "three_answers": {
            "normal theory (wrong here)": {
                "cpk": normal["Cpk"], "ppm": normal["expected_ppm_long_term"]},
            "percentile / ISO 21747": {
                "ppk": pct["Ppk_percentile"], "ppm": pct["observed_ppm"]},
            "Box-Cox transformed": {
                "cpk": tcap["Cpk_transformed"] if tcap else None,
                "ppm": tcap["expected_ppm"] if tcap else None},
        },
        "observed_ppm": observed_ppm,
        "runs_rule_violations_raw": rules_raw,
        "runs_rule_violations_transformed": rules_tx,
    }


# ---------------------------------------------------------------------------
# 2. gauge-corrected capability
# ---------------------------------------------------------------------------

def stage_gauge() -> dict:
    """The characteristic built to be gauge-dominated, corrected."""
    ch = _by_name("bore_rough_um")
    meas, _ = GEN.simulate(ch)
    v = meas["value"].to_numpy(dtype=float)

    # The study must be OF THIS CHARACTERISTIC. Calling simulate_study() with
    # its defaults produces a generic study at sigma_part = 1.0, which has
    # nothing to do with a roughness measured in micrometres -- and applying its
    # GRR here is what made the correction refuse.
    #
    # GRR = sqrt(EV^2 + AV^2), so the three components are split to land on the
    # characteristic's declared sigma_gauge: 0.8^2 + 0.5^2 + 0.2^2 = 0.93.
    sg = ch.sigma_gauge
    generic = GRR.anova_grr(GRR.simulate_study(), tolerance=ch.usl - ch.lsl)
    study = GRR.simulate_study(
        n_parts=10, n_operators=3, n_repeats=3,
        sigma_part=ch.sigma_process, sigma_repeat=0.8 * sg,
        sigma_operator=0.5 * sg, sigma_interaction=0.2 * sg)
    grr = GRR.anova_grr(study, tolerance=ch.usl - ch.lsl)
    # `GRR` is already a STANDARD DEVIATION, not a variance -- gauge_rr.py takes
    # the square root before returning it. My first version read a `var_grr` key
    # that does not exist, so `.get()` handed back its 0.0 default and the
    # "correction" corrected nothing while reporting 0.78 -> 0.78 without
    # complaint. A silent default makes a broken correction look exactly like a
    # working one, which is why this now raises instead.
    if "GRR" not in grr:
        raise KeyError(f"anova_grr returned {sorted(grr)}, no GRR")
    sigma_gauge = float(grr["GRR"])

    xbar, _ = CH.xbar_r(meas, baseline=slice(0, 100))
    corrected = TR.gauge_corrected(
        sigma_observed=float(v.std(ddof=1)), sigma_gauge=sigma_gauge,
        lsl=ch.lsl, usl=ch.usl, mean=float(v.mean()))

    # The truth is known here, because the generator was told both numbers.
    truth = TR.gauge_corrected(
        sigma_observed=float(np.hypot(ch.sigma_process, ch.sigma_gauge)),
        sigma_gauge=ch.sigma_gauge, lsl=ch.lsl, usl=ch.usl,
        mean=float(ch.target))
    # What happened when the study was the generic one: kept, because a guard
    # firing on my own mistake is better evidence it works than a test for it.
    mismatched = TR.gauge_corrected(
        sigma_observed=float(v.std(ddof=1)), sigma_gauge=float(generic["GRR"]),
        lsl=ch.lsl, usl=ch.usl, mean=float(v.mean()))
    return {"characteristic": ch.name, "grr_pct": grr.get("pct_GRR_of_TV"),
            "sigma_gauge_from_study": sigma_gauge,
            "sigma_gauge_true": ch.sigma_gauge,
            "sigma_observed": float(v.std(ddof=1)),
            "corrected": corrected, "with_true_sigmas": truth,
            "mismatched_study": {
                "sigma_gauge": float(generic["GRR"]),
                "valid": mismatched.get("valid"),
                "why": mismatched.get("why")}}


# ---------------------------------------------------------------------------
# 3 & 4. persistent queue + weekly report
# ---------------------------------------------------------------------------

def stage_weekly(skew: dict, gauge: dict) -> dict:
    db = OUT / "quality.db"
    if db.exists():
        db.unlink()
    q = WK.PersistentQueue(db)

    rng = np.random.default_rng(3)
    causes = ["tool wear", "material lot change", "setup error", "gauge drift",
              "coolant temperature"]
    now = time.time()

    # Two weeks of history, so the report has a CHANGE column to lead with.
    for week, offset in (("2026-W33", 7), ("2026-W34", 0)):
        for ch in GEN.catalogue():
            meas, _ = GEN.simulate(ch, seed=zlib.crc32(week.encode()) % 1000)
            v = meas["value"].to_numpy(dtype=float)
            xbar, _ = CH.xbar_r(meas, baseline=slice(0, 100))
            try:
                res = CAP.assess(v, xbar.sigma_hat, ch.lsl, ch.usl,
                                 in_control=True)
                cpk = res.get("Cpk", res.get("Ppk_percentile"))
                ppm = res.get("expected_ppm_long_term", res.get("observed_ppm"))
                method = "percentile" if "note" in res else "normal theory"
            except CAP.NotInControl:
                cpk, ppm, method = None, None, "refused: unstable"
            q.record_capability(week, ch.name, cpk, ppm, method)

    # Events with realistic ages, which is the whole point of persisting them.
    n_new = 0
    for k, ch in enumerate(GEN.catalogue()):
        for j in range(3):
            age_days = float(rng.uniform(0.5, 22.0))
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(now - age_days * 86400))
            if q.raise_event(ch.name, 100 + j, f"rule{1 + j % 3}",
                             25.0 + j, ts=ts):
                n_new += 1

    # Idempotency: re-running the analysis must not duplicate open work.
    dup = q.raise_event(GEN.catalogue()[0].name, 100, "rule1", 25.0)

    for e in q.open_items(now)[:6]:
        q.assign(e["event_id"], f"eng-{e['event_id'] % 3 + 1}")
        if e["event_id"] % 2 == 0:
            q.close(e["event_id"], causes[e["event_id"] % len(causes)],
                    "check tool life", "flag")

    refused = None
    try:
        q.close(q.open_items(now)[0]["event_id"], "", "x", "y")
    except ValueError as exc:
        refused = str(exc)[:120]

    changes = q.capability_change("2026-W34", "2026-W33")
    open_items = q.open_items(now)
    pareto = q.cause_pareto()
    rep = WK.render(OUT / "weekly_report.html", week="2026-W34",
                    changes=changes, open_items=open_items, pareto=pareto,
                    transforms=skew["comparison"],
                    gauge=gauge["corrected"],
                    meta={"n_characteristics": len(changes)})
    out = {"report": rep, "n_new_events": n_new,
           "duplicate_suppressed": not dup,
           "close_without_cause_refused": refused,
           "n_open": len(open_items),
           "oldest_open_days": open_items[0]["age_days"] if open_items else None,
           "changes": changes, "pareto": pareto}
    q.close_db()
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        res = json.loads((OUT / "weekly.json").read_text(encoding="utf-8"))
        (DOCS / "WEEKLY_AND_TRANSFORMS.md").write_text(report(res),
                                                       encoding="utf-8")
        print("re-rendered docs/WEEKLY_AND_TRANSFORMS.md")
        return

    t0 = time.perf_counter()
    print("1/3 the skew fix ...", flush=True)
    skew = stage_skew()
    bc = skew["comparison"].get("boxcox", {})
    print(f"    Box-Cox lambda {bc.get('lambda', float('nan')):.3f}, "
          f"normal after: {bc.get('normal_after')}", flush=True)

    print("2/3 gauge-corrected capability ...", flush=True)
    gauge = stage_gauge()
    c = gauge["corrected"]
    if c.get("valid"):
        print(f"    Cpk {c['cpk_observed']:.2f} observed -> "
              f"{c['cpk_gauge_corrected']:.2f} corrected", flush=True)
    else:
        print(f"    REFUSED: {c['why'][:70]}", flush=True)

    print("3/3 persistent queue + weekly report ...", flush=True)
    wk = stage_weekly(skew, gauge)

    res = {"skew": skew, "gauge": gauge, "weekly": wk,
           "wall_seconds": time.perf_counter() - t0}
    (OUT / "weekly.json").write_text(json.dumps(res, indent=1, default=str),
                                     encoding="utf-8")
    (DOCS / "WEEKLY_AND_TRANSFORMS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/WEEKLY_AND_TRANSFORMS.md and out/weekly_report.html "
          f"({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    sk, gg, wk = res["skew"], res["gauge"], res["weekly"]
    cmp = sk["comparison"]
    bc, js = cmp.get("boxcox", {}), cmp.get("johnson_su", {})
    A("# DATA-2: the skew fix, gauge correction, and the weekly report\n")
    A("Generated by `run_weekly.py`, not hand-edited. Four items the README named "
      "as not built — and three of them it named *with the fix already "
      "described*, which is reporting rather than doing.\n")

    A("## 1. The skew fix, replacing a policy\n")
    A(f"The README said: *\"the skew workaround is a policy, not a fix… the right "
      f"answer for a skewed characteristic is a transformation or a "
      f"distribution-appropriate chart, and neither is implemented.\"* Both are "
      f"now. `{sk['characteristic']}` has skew parameter {sk['skew_parameter']}, "
      f"sample skew **{cmp['skew_before']:.2f}**.\n")
    A("| fit | skew after | Anderson–Darling | normal at 5%? |")
    A("|---|---|---|---|")
    A(f"| none (raw) | {cmp['skew_before']:.2f} "
      f"| {cmp['anderson_before']['stat']:.3f} "
      f"| {'yes' if cmp['anderson_before']['normal'] else '**no**'} |")
    if "lambda" in bc:
        A(f"| Box-Cox (λ = {bc['lambda']:.3f}) | {bc['skew_after']:.2f} "
          f"| {bc['anderson_stat']:.3f} "
          f"| {'**yes**' if bc['normal_after'] else 'no'} |")
    A(f"| Johnson S<sub>U</sub> | {js['skew_after']:.2f} "
      f"| {js['anderson_stat']:.3f} "
      f"| {'**yes**' if js['normal_after'] else 'no'} |")
    A("\n**The test after the transformation is the point.** A transformation is "
      "a hypothesis — *this map makes the data normal* — and shipping one without "
      "re-testing is exactly the error it was supposed to fix.\n")
    if bc.get("normal_after") is False and js.get("normal_after"):
        A(f"**And Box-Cox is not enough here.** It removes the skew almost "
          f"perfectly ({cmp['skew_before']:.2f} → {bc['skew_after']:.3f}) and "
          f"still fails the normality test (AD {bc['anderson_stat']:.2f} against "
          f"a 5% critical value of {bc['crit_5pct']:.2f}). Johnson S<sub>U</sub> "
          f"passes ({js['anderson_stat']:.2f}).\n")
        A(f"The reason is that **Box-Cox only addresses skew**, and this "
          f"characteristic also has excess kurtosis "
          f"({cmp['kurtosis_before']:.2f} against 3.0 for a normal). Johnson "
          "fits skew *and* kurtosis, which is why it has two more parameters — "
          "and those two extra parameters are its cost, estimated from the same "
          "data. Reaching for Box-Cox because it is the familiar one would have "
          "produced symmetric non-normal data and a chart still built on the "
          "wrong tail.\n")
    if bc.get("shift"):
        A(f"*A note on the fit itself:* Box-Cox is location-sensitive, and this "
          f"data spans 170–216 — a max/min ratio of 1.27, over which x^λ is "
          f"nearly linear whatever λ is. Fitting it unshifted returned "
          f"**λ = −13.4** and destroyed the data through catastrophic "
          f"cancellation. Shifting the minimum to near zero (the two-parameter "
          f"Box-Cox, shift = {bc['shift']:.1f}) makes λ identifiable: "
          f"{bc['lambda']:.3f}, essentially a log.\n")

    A("### What the three approaches say about the same data\n")
    A("| approach | index | predicted PPM |")
    A("|---|---|---|")
    for name, v in sk["three_answers"].items():
        idx = v.get("cpk") if v.get("cpk") is not None else v.get("ppk")
        A(f"| {name} | {idx:.3f} | {v['ppm']:,.0f} |")
    A(f"| **observed** | — | **{sk['observed_ppm']:,.0f}** |")
    A("\nThe three disagree, and the disagreement is the size of the problem. "
      "Normal theory is the one that is wrong here and it is also the one with "
      "the familiar name.\n")

    tc = sk.get("transformed_capability")
    if tc:
        A(f"**And the transformed control limits are asymmetric about the centre "
          f"line** — {tc['lcl_original_units']:.1f} / "
          f"{tc['center_original_units']:.1f} / {tc['ucl_original_units']:.1f} in "
          f"original units, a ratio of {tc['asymmetry_ratio']:.2f}. That is "
          "correct and it looks broken to anybody taught that control limits are "
          "symmetric. It is a training problem rather than a statistical one, and "
          "it is the usual reason transformations get abandoned in practice.\n")

    raw, tx = sk["runs_rule_violations_raw"], sk["runs_rule_violations_transformed"]
    if tx is not None:
        A(f"### The real argument for transforming\n")
        A(f"Runs-rule violations on the **raw** data: **{raw}**. On the "
          f"**transformed** data: **{tx}**.\n")
        A("Pass 3 handled the skew by judging stability on rule 1 alone, because "
          "rules 2–8 assume symmetry and fire on the shape rather than the "
          "process. That was defensible and it threw away seven rules. On "
          "transformed data the symmetry is restored, so **the full rule set "
          "becomes valid again** — which is what a policy could never deliver.\n")

    A("## 2. Gauge-corrected capability\n")
    c = gg["corrected"]
    if c.get("valid"):
        A(f"`{gg['characteristic']}` is deliberately gauge-dominated. The R&R "
          f"study estimates σ_gauge = {gg['sigma_gauge_from_study']:.4f} against "
          f"a true {gg['sigma_gauge_true']:.4f}.\n")
        A("| | value |")
        A("|---|---|")
        A(f"| Cpk as measured | {c['cpk_observed']:.3f} |")
        A(f"| Cpk of the process (gauge removed) | **{c['cpk_gauge_corrected']:.3f}** |")
        A(f"| variance attributable to the gauge | "
          f"**{c['gauge_share_of_variance'] * 100:.0f}%** |")
        A(f"\n**{c['verdict']}.** Cpk on measured values blames the process for "
          "the instrument, and those are different budgets and different teams. "
          f"Correcting moves the index by "
          f"{c['cpk_gauge_corrected'] - c['cpk_observed']:+.2f}.\n")
    else:
        A(f"**The correction refused**: {c['why']}\n")
        A("That refusal is the designed behaviour. σ_process² = σ_observed² − "
          "σ_gauge² can come out negative, which is not a rounding artefact to "
          "clamp — it means the gauge study and the process study are "
          "inconsistent, and reporting a clamped zero would hide exactly the "
          "problem that should stop the analysis.\n")
    mm = gg.get("mismatched_study")
    if mm and mm.get("valid") is False:
        A(f"### The guard fired on my own mistake\n")
        A(f"The first version of this ran `simulate_study()` with its **defaults** "
          f"— a generic study at σ_part = 1.0 — and applied its GRR "
          f"(σ = {mm['sigma_gauge']:.3f}) to a roughness measured in "
          f"micrometres with σ_observed = {gg['sigma_observed']:.3f}. The two had "
          "nothing to do with each other, and the correction refused:\n")
        A(f"> {mm['why']}\n")
        A("That is a better demonstration than any test I could have written for "
          "it. The obvious alternative implementation — clamp the negative "
          "variance to zero and carry on — would have reported a plausible "
          "corrected Cpk built on a gauge study of a different characteristic, "
          "and nothing would ever have flagged it.\n")

    A("## 3. A disposition queue that survives a restart\n")
    A(f"**{wk['n_open']} open items, oldest "
      f"{wk['oldest_open_days']:.1f} days.** Duplicate suppressed on re-run: "
      f"**{wk['duplicate_suppressed']}**. Closing without a cause refused: "
      f"`{(wk['close_without_cause_refused'] or '')[:80]}`\n")
    A("**Ageing is the single most useful thing a disposition queue does, and it "
      "is impossible without persistence.** An in-process queue is born every "
      "morning: it cannot tell a three-week-old item from one raised yesterday, "
      "and an item nobody has touched in three weeks is a different problem from "
      "the same item raised yesterday.\n")
    A("The UNIQUE key on (characteristic, subgroup, rule) is what makes the "
      "weekly job idempotent. Without it, re-running the analysis on Monday "
      "raises a second event for every excursion already being worked, and the "
      "queue is unreadable within a month.\n")
    A("Still not here: users, permissions, and an audit trail beyond the event "
      "rows. Stated rather than implied.\n")

    A("## 4. The weekly report\n")
    r = wk["report"]
    A(f"`out/weekly_report.html`, {r['bytes'] / 1024:.0f} KB, self-contained, "
      f"week {r['week']}.\n")
    A("A weekly report is not the dashboard with a date range on it. A dashboard "
      "answers *what is happening now* for someone who already has the context; "
      "a weekly report is read in a meeting by people who do not, and it has to "
      "answer three questions in order: **what changed**, **what is anyone doing "
      "about it**, and **what is the pattern**.\n")
    A("| characteristic | Cpk | vs last week | method |")
    A("|---|---|---|---|")
    for c_ in wk["changes"][:6]:
        d = "new" if c_["delta"] is None else f"{c_['delta']:+.2f}"
        cpk = "—" if c_["cpk"] is None else f"{c_['cpk']:.2f}"
        A(f"| {c_['characteristic']} | {cpk} | {d} | {c_['method']} |")
    A("\n**Ordered by movement, not by magnitude.** A characteristic sitting at "
      "Cpk 1.9 for a year is not news; one that fell from 1.9 to 1.4 this week is "
      "the whole meeting. A report that leads with current values makes the "
      "reader do the differencing, and they will not.\n")

    A("---")
    A(f"*Generated in {res.get('wall_seconds', 0):.0f}s.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
