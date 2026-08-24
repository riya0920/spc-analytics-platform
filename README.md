# DATA-2 — Statistical Process Control Analytics Platform

**Status: complete.** The chart engine, the runs-rules engine with measured
false-alarm economics, the ARL bake-off, capability with its refusals, and ANOVA
gauge R&R are built. The quality-engineer dashboard, the alarm disposition
workflow, and attribute-chart coverage are not.

```bash
python run_spc.py
```

~90 seconds. Writes [docs/RESULTS.md](docs/RESULTS.md) and `out/results.json`.

## The first check, passed

Control limits come from **within-subgroup** variation — `R̄/d₂` and `s̄/c₄` — never
from the standard deviation of the whole series.

This is not a stylistic preference. A control chart asks whether variation
*between* subgroups exceeds variation *within* them. Overall σ contains both, so a
process that has shifted inflates its own limits, which widens the band, which
hides the shift. **The chart is at its blindest exactly when the process is at its
worst.**

Measured on planted disturbances:

| characteristic | correct limit width | naive (overall-σ) width | ratio | disturbances caught, correct | caught, naive |
|---|---|---|---|---|---|
| shaft_dia_mm | 0.0260 | 0.0775 | **2.99×** | 3/3 | 2/3 |
| wall_thk_mm | 0.0531 | 0.1693 | **3.19×** | 2/2 | 2/2 |
| cavity_mix_mm | 0.0746 | 0.1565 | **2.10×** | 1/1 | 1/1 |

The naive chart **missed the 0.5σ sustained shift entirely** and was 21 subgroups
late on the tool-wear trend.

### The constants are derived, not pasted

`d₂` by numerically integrating the expected range of *n* standard normals; `c₄`
from log-gamma. Agreement with Montgomery's Appendix VI is to 5 decimal places
(worst absolute error 4.9e-05 — i.e. the published table's rounding). If these
disagreed, every limit in the platform would be wrong by a constant and no
downstream test would notice.

### The stratification case

`cavity_mix_mm` pools two cavities into every subgroup for the whole run. The
inflated *within*-subgroup variation makes the limits wide and the chart calm — and
the rule that catches it is **rule 7, 15 consecutive points within 1σ**, at
subgroup 21. That is the textbook stratification signature, and it is the chart
correctly reporting a badly chosen rational subgroup. The fix is to re-subgroup by
cavity, not to change the arithmetic.

## Alarm economics, measured

Probability each rule fires at least once over a 200-point chart — the form a
quality engineer actually experiences:

| rule | in control | at 1σ shift |
|---|---|---|
| 1 — one point beyond 3σ | 42.0% | 99.2% |
| 2 — 2 of 3 beyond 2σ, same side | 33.1% | 100.0% |
| 3 — 4 of 5 beyond 1σ, same side | 49.5% | 100.0% |
| 4 — 8 consecutive on one side | 54.0% | 100.0% |
| 5 — 6 consecutive trending | 6.4% | 6.4% |
| 6 — 14 consecutive alternating | 17.8% | 17.8% |
| 7 — 15 consecutive within 1σ | 17.4% | 0.2% |
| 8 — 8 consecutive beyond 1σ | 1.4% | 41.3% |
| **rules 1–4 stacked** | **89.3%** | 100.0% |
| **all 8 stacked** | **93.6%** | 100.0% |

**Stacking inflation, measured: rule 1 alone fires on 42% of clean charts; all
eight fire on 94%.** Every rule is another test on the same data. Shops that enable
all eight because the software offers them are buying sensitivity with a currency
nobody showed them the price of.

Rules 5 and 6 are worth reading twice: their detection rate at a 1σ *shift* equals
their false-alarm rate, because a sustained mean shift produces no trends and no
alternation. They are not weak rules — they are rules for a different failure mode.
Rule 7's detection rate *drops* under a shift, which is exactly right: it is
looking for points hugging the centre line.

**Naming, precisely:** rules 1–4 are the classic Western Electric zone tests
(*Statistical Quality Control Handbook*, 1956). Rules 5–8 are Nelson's additions
(*JQT*, 1984), which most software ships under the Western Electric label. The
distinction is in the code.

## The ARL table

| shift | Shewhart (rule 1) | Shewhart (WE 1–4) | Shewhart (all 8) | EWMA | CUSUM |
|---|---|---|---|---|---|
| 0.0σ | **374.2** | 92.3 | 77.2 | **366.3** | **376.4** |
| 0.5σ | 156.8 | 27.7 | 26.9 | 35.0 | 35.6 |
| 1.0σ | 43.2 | 9.4 | 9.4 | 8.7 | 9.8 |
| 1.5σ | 14.8 | 5.1 | 5.1 | 4.3 | 5.5 |
| 2.0σ | 6.2 | 3.4 | 3.4 | 2.7 | 3.8 |
| 3.0σ | 2.0 | 1.8 | 1.8 | **1.5** | 2.5 |

Sanity: rule 1's measured ARL₀ of 374.2 against the theoretical 370.4, ARL₁ at 1σ
of 43.2 against the theoretical 43.9, and WE rules 1–4 at 92.3 against the
published ~91.75.

**EWMA and CUSUM are calibrated by simulation, not quoted from a table.** The
published pairing (λ=0.2, L=2.962 → ARL₀=370) is derived for the *asymptotic* EWMA
limits; this implementation uses the exact time-varying limits, which are tighter
over the first few dozen points. At L=2.962 the measured ARL₀ is **487.5**, not
370. Quoting the constant would have put EWMA and Shewhart on different
false-alarm budgets and made the head-to-head meaningless. Bisecting on measured
ARL₀ gives L\* = 2.861 (ARL₀ = 366) and h\* = 4.781 (ARL₀ = 376).

**So why not EWMA everywhere?** Look at the 3σ row: Shewhart 2.0 points, EWMA 1.5,
CUSUM 2.5 — the memory that makes EWMA sensitive to small shifts also makes it
slower to fully respond to large ones, because the shifted observation is averaged
with in-control history. Add that an operator can read a Shewhart chart without
being told what a smoothing constant is, and the answer is tool-per-purpose:
Shewhart on the floor, EWMA/CUSUM alongside it for the slow drifts the floor chart
will not catch.

## Capability, and the refusals

| characteristic | verdict | Cp | Cpk | Pp | Ppk |
|---|---|---|---|---|---|
| shaft_dia_mm | **REFUSED** (2 OOC points) | — | — | — | — |
| wall_thk_mm | computed | 1.01 | 1.01 | 1.03 | 1.03 |
| cavity_mix_mm | **REFUSED** (45 OOC points) | — | — | — | — |
| seal_force_N | **REFUSED** (4 OOC points) | — | — | — | — |
| bore_rough_um | computed | 0.78 | 0.78 | 0.77 | 0.76 |

`capability.assess()` raises `NotInControl` rather than returning a number with a
caveat, because Cpk converts a mean and a sigma into a tail probability and an
unstable process has neither. The caveat gets deleted on the way to the customer;
the number does not.

RESULTS.md §5 then prints what a tool *without* the gate would have shown, so the
refusal has something to point at — including `shaft_dia_mm` at Cpk 1.54 / Ppk 1.15
(ratio 1.34), which reads as a capable process and is actually a process with a
planted 2σ shift and a tool-wear ramp in it.

## Gauge R&R — process problem or measurement problem?

| study | %GRR | ndc | verdict |
|---|---|---|---|
| good gauge | 4.6% | 30.6 | acceptable |
| marginal gauge | 16.0% | 8.7 | conditional |
| bad gauge | 37.9% | 3.4 | UNACCEPTABLE |
| **bore_rough_um** | **82.7%** | **1.0** | **UNACCEPTABLE** |

`bore_rough_um` has σ_process = 0.040 µm and σ_gauge = 0.075 µm — the instrument is
noisier than the thing it measures. Its Cpk of 0.78 above is **measuring the
gauge**, and any improvement project chartered off its control chart is chartered
against noise.

ndc is the number to put in front of a manager: not a percentage but *how many
different sizes of part this instrument can actually tell apart*. Here, one.

ANOVA method rather than average-and-range, because ANOVA estimates the
operator × part interaction that the range method structurally cannot see. Variance
components were validated against the generator across 30 seeds (EV 0.0490 vs a
true 0.050; PV 1.031 vs a true 1.000) and the sum-of-squares identity is checked.

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

`python extend.py` — five gaps this README previously named:

- **Attribute charts (p, np, c, u)**, with the selection tree that decides between
  them and the cost of getting it wrong: a c chart on varying-area data flags
  roughly twice as many points as the correct u chart, and none of the extras are
  process signals.
- **X̄-s and I-MR exercised.** The three σ̂ estimates agree — and they agree with
  the **total** (process + gauge) σ, not the process σ alone, which is the same
  fact the gauge R&R section makes from the other direction.
- **The non-normal capability path, actually triggered.** It was an untested code
  path presented as a feature. Running it exposed an interaction: **the runs rules
  assume symmetry, so a skewed process trips the stability gate and the fallback
  built for skewed data can never run on skewed data.** Normal theory
  under-predicts the defect rate **576 vs 5,000 PPM** here.
- **Phase I / phase II limit revision.** A process improves, nobody revises the
  limits, and the chart goes quiet — which reads as success and is a chart that has
  lost the ability to detect anything.

## Completed in the third pass — see [docs/COMPLETION.md](docs/COMPLETION.md)

```bash
python complete.py    # ~5 s; writes COMPLETION.md, SPC_METHODOLOGY.md, the dashboard
```

- **The non-normal capability path, finally triggered.** Pass 2 reported this as
  an open gap: `seal_force_N` is deliberately skewed and was refused for
  *instability* before the distribution branch could matter. The cause was
  diagnosed there and not acted on — the Western Electric runs rules assume
  symmetry, so on a skewed process rule 2 fires because the mean is not the
  median and more than half the points sit on one side **by construction**.
  Judging stability on rule 1 alone (a statement about the actual tail, which
  survives skew) gets it to the gate, and the branch pays for itself:
  **normal theory predicts 552 PPM where 5000 are observed, a
  9× understatement.** The process looks capable at
  Cpk 1.22 while shipping 0.5% out of spec.
- **X-bar-R, X-bar-s and I-MR on the same data.** X-bar-R and X-bar-s agree to
  1.0% at n=5, which is why the
  textbook cut-over is near n=8. **I-MR on subgrouped data is wrong for a reason
  that is not its sigma estimate** — that comes out at 0.01089
  against X-bar-R's 0.01092. An X-bar chart puts its
  limits at 3σ/√n because it charts *means*; an individuals chart puts them at
  3σ. Measured ratio **2.23×** against a
  predicted √5 = 2.24×, and it
  flags 86 subgroups where X-bar-R flags
  229.
- **Phase I made iterative, and a limit-revision policy that refuses.** Phase I
  removes out-of-control subgroups and refits until it converges, because a
  single pass computes limits from data containing the disturbances the limits
  should exclude — the disturbance inflates the limits it sits inside and so
  hides itself. Revision requires one of three documented reasons, and
  *"the chart is alarming"* is explicitly not one of them.
- **Why limits must be frozen, measured.** A 5.3σ
  drift across 300 subgroups: frozen limits raise
  **208 alarms**, limits recomputed on a
  25-subgroup rolling window raise **21**.
  The rolling limits walk along with the drift.
- **The OOC disposition workflow.** 229 events,
  106 still open, one event per subgroup rather than one per
  rule. **Closing requires an assignable cause and the refusal is enforced** —
  without that rule the queue empties without anyone naming a cause, and the
  assignable-cause Pareto, which is what says where to spend engineering time,
  never exists.
- **A dashboard** at `out/spc_dashboard.html`, self-contained, with zones A/B/C
  drawn — most Western Electric rules are *about* the zones, so a chart without
  them cannot be read by the rules judging it — each violating point naming the
  rule that fired, and the phase I/II boundary marked.
- **[docs/SPC_METHODOLOGY.md](docs/SPC_METHODOLOGY.md)**, the guide the spec
  asked for: chart selection, rational subgrouping, phase I/II, when limits may
  be revised, stability-before-capability, and what to do when each rule fires.

### A comparison I had to redo

The first version of the chart-choice table compared raw violation counts — 317
on the individuals chart against 211 on X-bar — and concluded I-MR was *more*
sensitive, which is the opposite of the truth. Those counts are not comparable:
the individuals series has five times as many points. Mapping both back onto
subgroup indices is what makes the comparison mean anything, and it is an easy
mistake to make with two charts of the same process at different granularity.

## The last four items — see [docs/WEEKLY_AND_TRANSFORMS.md](docs/WEEKLY_AND_TRANSFORMS.md)

```bash
python run_weekly.py    # ~6 s; writes the doc and out/weekly_report.html
```

- **The skew FIX, replacing a policy.** The README said the right answer was "a
  transformation or a distribution-appropriate chart, and neither is
  implemented". Both are now — and the comparison is the finding: **Box-Cox
  removes the skew almost perfectly (1.42 →
  0.007) and still fails the normality test.** Johnson S<sub>U</sub>
  passes. Box-Cox only addresses skew, and this characteristic also has excess
  kurtosis (6.25 against 3.0); Johnson fits both, which
  is what its two extra parameters buy.
- **And the payoff is the runs rules.** Pass 3 judged stability on rule 1 alone
  because rules 2–8 assume symmetry — defensible, and it threw away seven rules.
  On transformed data the symmetry is restored and **the full rule set becomes
  valid again**, which no policy could deliver.
- **Gauge-corrected capability.** Cpk **0.76 as measured →
  0.94** with the instrument removed;
  **36% of the observed variance is the
  gauge**. Cpk on measured values blames the process for the instrument, and
  those are different budgets and different teams.
- **A disposition queue that survives a restart.** 12 open items,
  oldest **17.6 days**. Ageing is the most useful thing a
  disposition queue does and it is impossible in-process — an in-process queue is
  born every morning. The UNIQUE key makes the weekly job idempotent, so Monday's
  re-run does not raise a second event for every excursion already being worked.
- **The weekly report**, ordered by **change and by age, not by magnitude**. A
  characteristic at Cpk 1.9 for a year is not news; one that fell from 1.9 to 1.4
  this week is the meeting.

### Two bugs found doing it

**Box-Cox needed a two-parameter fit to work at all.** The data spans 170–216 — a
max/min ratio of 1.27, over which x^λ is nearly linear whatever λ is — so the
likelihood is flat and the optimiser wandered to **λ = −13.4**, destroying the
data through catastrophic cancellation. Shifting the minimum to near zero makes λ
identifiable: **0.126**, essentially a log.

**And the gauge guard fired on my own mistake.** The first version ran the R&R
study with its *defaults* — a generic study at σ_part = 1.0 — and applied its GRR
to a roughness measured in micrometres. The correction refused with *"gauge
variance exceeds observed variance"*, which is exactly the inconsistency it
exists to detect. Clamping the negative variance to zero, which is the obvious
alternative, would have reported a plausible corrected Cpk built on a study of a
different characteristic and nothing would have flagged it.

### A reproducibility bug that had been there since pass 1

`src/generate.py` seeded its RNG with `hash(ch.name)`. **Python salts string
hashing per process** (PEP 456), so the generator produced different data on
every run — three consecutive runs gave means of 180.017141, 179.970739 and
179.917763. Every number published before this fix came from a dataset a re-run
could not reproduce, which silently broke the reproducibility this whole project
claims. Replaced with `zlib.crc32`; all reports regenerated. The same bug was in
ML-3 and SE-1 and is fixed there too.

## What is NOT built

1. **No real measurement data.** Everything is `src/generate.py`. The generators
   plant known disturbances, which is what makes detection scoreable, and it is
   also what makes every number here a statement about the generator.
2. **The queue has no users and no permissions.** It persists, ages and enforces
   a cause before closing; it does not authenticate anybody, and its audit trail
   is the event rows rather than a controlled record.
3. **Nothing schedules the weekly report.** It is a script that writes a
   self-contained page; nothing runs it on Mondays or delivers it.
4. **The transformation is fitted once, not monitored.** A transformation is
   valid near the operating point it was fitted at, and nothing here re-tests it
   when the process moves — which is the same discipline the limit-revision
   policy applies to control limits and it is not wired to it.
5. **Johnson S<sub>U</sub> is fitted but not charted.** It wins the normality
   test; the control chart still uses the Box-Cox scale, because back-transformed
   Johnson limits need the inverse and it is not implemented.

## Layout

```
src/constants.py    d2 by integration, c4 by log-gamma, A2/D3/D4/B3/B4
src/generate.py     measurement streams with planted disturbances + ground truth
src/charts.py       X-bar/R, X-bar/s, I-MR, p; and the overall-sigma error, on purpose
src/rules.py        WE 1-4 + Nelson 5-8, vectorised over batches; EWMA; CUSUM
src/arl.py          batched ARL simulation, ARL0 calibration by bisection
src/capability.py   Cp/Cpk vs Pp/Ppk, normality gate, percentile fallback, the refusal
src/gauge_rr.py     ANOVA gauge R&R with interaction pooling
run_spc.py          orchestration; writes docs/RESULTS.md
```
