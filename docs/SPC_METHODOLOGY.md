# SPC methodology guide

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
