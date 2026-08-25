# Real measurement data, and NIST's answers to check it against

The first item on this project's not-built list was *no real measurement data — everything is `src/generate.py`*. Fetching real numbers closes that item and proves nothing on its own: a control chart computed on real data is still a control chart validated against itself.

What makes this worth doing is that **NIST publishes the answers**. The check-standard case study states its variance decomposition in prose, so the arithmetic here has a reference rather than a mirror.

> NIST/SEMATECH e-Handbook of Statistical Methods, section 2.6.2.2 (check standard for resistivity measurements)


## 1. Does this project's arithmetic reproduce NIST's?

25 days, 6 repetitions per day, resistivity of silicon check standard #137 on probe #2362.

| quantity | computed here | NIST publishes | agrees |
|---|---:|---:|:--:|
| pooled repeatability s1 | 0.06139 | 0.06139 | ✅ |
| level-2 s2 | 0.02680 | 0.0268 | ✅ |
| F(0.05, 5, 125) | 2.28677 | 2.29 | ✅ |
| s-chart UCL | 0.09283 | 0.09238 | ❌ |

**3 of 4 agree** to NIST's published precision. That is the first time anything in this project has been checked against a number it did not produce.

NIST also states that two days exceed the s-chart limit; this finds **2** — matching.


### The fourth is a discrepancy in the reference

NIST's published UCL does not follow from NIST's published inputs. They state s₁ = 0.06139, F = 2.29 and UCL = 0.09238, with the formula UCL = s₁·√F. But s₁·√F is **0.09290** with their rounded F and **0.09283** with the exact one. The value they print implies F = **2.2644**, which is not F(0.05, 5, 125).

It is 0.5%, and it changes nothing: **both limits flag exactly the 2 days NIST says are flagged** (`same_days_either_way = True`). It is written down rather than absorbed by a wider tolerance — a reference disagreeing with itself is worth recording, and quietly loosening a threshold until a check passes is the habit this project is against.


## 2. The project's own machinery, on data nobody shaped

An individuals chart on the 25 daily values: centre 97.0698, limits [96.9884, 97.1513], **0 points beyond**.

No runs rule fires on any of the eight — the process is stable, which is what NIST describes and is the first time this project's rule set has agreed with an outside source about real data.


### The scale trap, which cost a plausible and meaningless Cpk

The charted values are **averages of 6 measurements**, and NIST's `s1` is the standard deviation of **one**. Feeding `s1` in as σ_within against data that are means of six mixes two scales, and it does not look wrong — it produces a perfectly plausible index that answers no question anybody asked. Two questions live here:

| question | σ | value |
|---|---|---:|
| capability of a **single** measurement on a new wafer | √(s₁²+s₂²) | 0.06698 |
| capability of the **daily average**, which is what is plotted | √(s₁²/6+s₂²) | 0.03669 |
| σ_within if you just use `s1` — **the wrong one** | s₁ | 0.06139 |

The first is NIST's own level-1 + level-2 uncertainty construction. Everything below uses it.


### Capability against a ±3% window on the nominal 100 ohm·cm

The window is chosen on the **nominal**, not around the observed mean: centring a spec on your own data guarantees a flattering Cpk. This crystal runs at 97.07, which sits almost exactly on the lower limit of 97.0.

| index | correct scale | using s₁ (wrong scale) |
|---|---:|---:|
| Cp | **14.929** | 16.290 |
| Cpk | **0.348** | 0.379 |
| Pp | **37.316** | 37.316 |
| Ppk | **0.869** | 0.869 |

**Cp 14.9 against Cpk 0.35** is the textbook centring failure, and here it is entirely an artefact of the invented spec: the process is tight enough to fit inside the window many times over and sits on the edge of it. A real tolerance for this material would be centred near 97, and then Cp and Cpk would be close. It is left in because a capability index computed against a spec somebody made up is exactly what this project warns about elsewhere, and the demonstration is more useful than the number.

**Cpk/Ppk = 0.40, and it is below 1** — the reverse of the usual case. Normally within-subgroup σ is smaller than overall σ and Cpk exceeds Ppk. Here σ for a single measurement is larger than the spread of the daily *averages*, because averaging six repetitions divides the repeatability by √6. It is the same scale point arriving from the other direction.


Normality: Anderson–Darling **0.203** against a 5% critical value of 0.703 — **normal** (skew -0.29, excess kurtosis -0.40). The synthetic study needed a transformation to pass this; real check-standard data does not need one, which is a point in favour of the transformation machinery being exercised on a case that genuinely required it rather than one built to.


## 3. The gauge study, and a %GRR that hides a real bias

MPC61 is 5 check-standard wafers measured on 5 probes. Mapping **probe onto operator** is a modelling choice, written down rather than assumed: NIST's question of this data is whether the probes are equivalent or biased relative to each other, which is structurally the reproducibility question.

| | |
|---|---:|
| %GRR of total variation | 1.997 |
| number of distinct categories | 70.579 |
| EV (repeatability) | 0.054 |
| AV (reproducibility) | 0.022 |
| probe-to-probe spread in mean resistivity | 0.0550 ohm·cm |
| AIAG verdict | acceptable |

**The %GRR says the measurement system is excellent. The probes are still measurably biased against each other.** A one-way ANOVA on wafer-centred values — so wafer-to-wafer variation cannot masquerade as probe bias — gives F = 10.71, p = 4.07e-08: **significant**. The spread between probe means is 0.0550 ohm·cm, about 2.1× the day-to-day standard deviation.

Both things are true, and the reason is that **%GRR is a ratio to part variation**. These wafers span a wide range of resistivity, so a real systematic offset between probes disappears into a denominator. That is the failure mode of the metric, not of the gauge: a measurement system can be excellent *for telling these parts apart* and still be unfit for comparing results between probes — which, for a laboratory issuing certificates, is the question that matters. NIST's own conclusion on this study is that the probes are not equivalent, and the ANOVA agrees with NIST rather than with the %GRR headline.


## What this settles, and what it does not

- **It settles the first not-built item, and more than it asked for.** The data is real *and* the arithmetic now has a reference: 3 of 4 published quantities reproduced to NIST's precision, and the fourth is a discrepancy inside the reference itself.
- **It is one process and 25 days.** A check standard measured on one probe is not a production line, and nothing here exercises the subgroup-based charts, the attribute charts or the ARL work on real data — those still run on `src/generate.py`.
- **The spec limits are invented.** NIST's study has no tolerance; the ±3% window is a stand-in, stated as one, and every capability index below is a statement about that choice as much as about the process.

