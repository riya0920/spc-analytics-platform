"""Fetch NIST's real measurement data, and the answers NIST published for it.

WHY THIS AND NOT JUST ANY REAL DATA. The README's first not-built item was "no
real measurement data -- everything is `src/generate.py`". Fetching some real
numbers would close that item and prove nothing about the code: a control chart
computed on real data is still a control chart validated against itself.

The NIST/SEMATECH e-Handbook case studies come with **certified answers**. For
the check-standard study it publishes, in prose, on a page anybody can read:

    pooled repeatability   s1 = 0.06139 ohm.cm   with K(J-1) = 125 df
    level-2 (day-to-day)   s2 = 0.02680 ohm.cm   with K-1    =  24 df
    s-chart upper limit    UCL = s1 * sqrt(F(0.05, 5, 125)) = 0.09238

So this project's arithmetic can be checked against a reference rather than
against its own generator. That is the difference between "we have real data
now" and "the implementation is right".

    MPC62 -- check standard #137, probe #2362, resistivity of silicon wafers.
             J = 6 repetitions per day, K = 25 days. ASTM F84.
    MPC61 -- a gauge study: 5 check-standard wafers, 5 probes, 2 runs, 2
             operators, with an average and a short-term SD per measurement.

Data files are public domain (US Government work) and are cached under
data/NIST/, which is gitignored -- nothing is redistributed here.

    python fetch_nist.py
    python fetch_nist.py --check
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DEST = ROOT / "data" / "NIST"
BASE = "https://www.itl.nist.gov/div898/handbook/datasets"

# What NIST states on mpc622.htm. Quoted so the comparison is against their
# published numbers rather than against a value this project computed earlier
# and then enshrined.
NIST_PUBLISHED = {
    "s1_repeatability": 0.06139,
    "s1_df": 125,
    "s2_level2": 0.02680,
    "s2_df": 24,
    "f_crit_0_05_5_125": 2.29,
    "s_chart_ucl": 0.09238,
    "J_repetitions": 6,
    "K_days": 25,
    "source": ("NIST/SEMATECH e-Handbook of Statistical Methods, "
               "section 2.6.2.2 (check standard for resistivity measurements)"),
}

SPECS = {
    "MPC62": {
        "cols": ["crystal", "check_id", "month", "day", "hour", "minute",
                 "operator", "humidity", "probe", "temp", "checkstd",
                 "stddev", "df"],
        "what": "check standard 137, probe 2362, resistivity (ohm.cm)",
    },
    "MPC61": {
        "cols": ["run", "wafer", "probe", "month", "day", "operator", "temp",
                 "average", "stddev"],
        "what": "gauge study: 5 wafers x 5 probes x 2 runs x 2 operators",
    },
}


def _get(url: str, tries: int = 4, timeout: int = 30) -> bytes:
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(0.5 * (a + 1))
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def _parse(text: str, n_cols: int, name: str) -> np.ndarray:
    """Rows of exactly `n_cols` floats.

    The files carry a ~50-line prose header and Dataplot instructions. NIST's
    own reader says `SKIP 50`, and this does not: a hard-coded skip breaks
    silently the day the header gains a line, and it cannot tell a data row from
    a sentence that happens to start with a number. Parsing on shape instead --
    every line that is exactly `n_cols` floats is data.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != n_cols:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{name}: no rows of {n_cols} numeric columns found")
    return np.asarray(rows, dtype=float)


def fetch(name: str) -> np.ndarray:
    DEST.mkdir(parents=True, exist_ok=True)
    cache = DEST / f"{name}.DAT"
    if cache.exists():
        blob = cache.read_bytes()
    else:
        print(f"  {name}.DAT ...", flush=True)
        blob = _get(f"{BASE}/{name}.DAT")
        cache.write_bytes(blob)
    spec = SPECS[name]
    return _parse(blob.decode("utf-8", "replace"), len(spec["cols"]), name)


def main() -> None:
    if "--check" in sys.argv:
        p = DEST / "nist.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            for k in SPECS:
                if k in z.files:
                    print(f"present: {k} {z[k].shape}")
        else:
            print("not fetched")
        return

    out = {}
    for name in SPECS:
        a = fetch(name)
        out[name] = a
        print(f"  {name}: {a.shape} -- {SPECS[name]['what']}")
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DEST / "nist.npz",
        **out,
        **{f"{k}_cols": np.array(v["cols"]) for k, v in SPECS.items()},
        published=np.array(json.dumps(NIST_PUBLISHED)))
    print(f"wrote {DEST / 'nist.npz'}")
    print("source: NIST/SEMATECH e-Handbook of Statistical Methods "
          "(public domain). Not redistributed.")


if __name__ == "__main__":
    main()
