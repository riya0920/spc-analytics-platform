# DATA-2 — Statistical Process Control Analytics Platform

**Status: ~50% slice.** The chart engine, the runs-rules engine with measured
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

## What is NOT built (the other 50%)

1. **No dashboard.** No charts rendered, no violation annotations, no alarm queue,
   no disposition workflow. Everything is tables in a markdown file. The spec's
   "violations get investigated and dispositioned, not just displayed" is not
   implemented at all.
2. **Attribute charts are stubs.** `p_chart` exists; np/c/u do not, and none are
   exercised or validated against planted attribute disturbances.
3. **X̄-s and I-MR are implemented but not exercised.** Only X̄-R is used in the
   results. The subgrouping-rationale *guide* that the spec asks for is not
   written; the rationale is in docstrings.
4. **No non-normal capability case actually triggered.** The percentile-method
   fallback is implemented and the normality test runs, but the skewed
   characteristic (`seal_force_N`) was refused for instability before the
   distribution branch mattered — so that path is untested by the pipeline.
   That is an honest gap, not a passing test.
5. **No weekly quality report, no OOC action-plan hooks, no SPC methodology guide**
   as a separate deliverable.
6. **No real measurement data.** Everything is `src/generate.py`.
7. **Limits are frozen on a baseline slice** (subgroups 0–99) but there is no
   limit-revision policy, no phase I / phase II distinction made explicit, and no
   handling of "the process improved, re-establish the limits".

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
