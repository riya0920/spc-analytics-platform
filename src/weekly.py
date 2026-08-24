"""The weekly quality report, and a disposition queue that survives a restart.

===========================================================================
THE REPORT
===========================================================================

The README's gap: *"No weekly quality report as a scheduled artefact. The
dashboard and the disposition queue are the ingredients; nothing emails a PDF on
Mondays."*

WHAT A WEEKLY QUALITY REVIEW IS ACTUALLY FOR, and why it is not the dashboard
with a date range on it. A dashboard answers "what is happening now" and is read
by someone who already knows the context. A weekly report is read in a meeting by
people who do not, and it has to answer three questions in order:

  1. **What changed?**  Not "what is the state" -- the state was fine last week
     too. A report that leads with current values makes the reader do the
     differencing, and they will not.
  2. **What is anyone doing about it?**  Open items, with owners and ages. An
     item open for three weeks is a different problem from the same item raised
     yesterday, and only the report can see that.
  3. **What is the pattern?**  The assignable-cause Pareto. Individual OOC
     events are noise; the same cause five times is a project.

So the report is ordered by CHANGE and by AGE, not by magnitude. A characteristic
sitting at Cpk 1.9 for a year is not news; one that fell from 1.9 to 1.4 this
week is the whole meeting.

===========================================================================
PERSISTENCE
===========================================================================

The README's other gap: *"The disposition queue is in-process. No persistence, no
users, no permissions, no audit trail beyond the event list."*

Persistence is the one that changes what the queue IS. An in-process queue cannot
age an item, cannot show that nobody has touched it in three weeks, and cannot
survive the restart that happens every time somebody deploys. Ageing is the
single most useful thing a disposition queue does, and it is impossible without
a durable store.

Users and permissions are still not here, and that is stated rather than implied.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS ooc_event (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    characteristic TEXT NOT NULL,
    subgroup    INTEGER NOT NULL,
    rule        TEXT NOT NULL,
    value       REAL NOT NULL,
    raised_ts   TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'OPEN',
    assignee    TEXT,
    cause       TEXT,
    action      TEXT,
    disposition TEXT,
    closed_ts   TEXT,
    UNIQUE(characteristic, subgroup, rule)
);
CREATE INDEX IF NOT EXISTS ix_ooc_state ON ooc_event(state);

CREATE TABLE IF NOT EXISTS capability_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week        TEXT NOT NULL,
    characteristic TEXT NOT NULL,
    cpk         REAL,
    ppm         REAL,
    method      TEXT,
    UNIQUE(week, characteristic)
);
"""


class PersistentQueue:
    """The disposition queue, on disk.

    The UNIQUE key on (characteristic, subgroup, rule) is what makes the queue
    idempotent: re-running the analysis on Monday must not raise a second event
    for an excursion already being worked. Without it a weekly job duplicates
    every open item every week, and the queue becomes unreadable within a month.
    """

    def __init__(self, path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def raise_event(self, characteristic: str, subgroup: int, rule: str,
                    value: float, ts: str | None = None) -> bool:
        """Returns True if this is NEW, False if already tracked."""
        try:
            self.conn.execute(
                "INSERT INTO ooc_event (characteristic, subgroup, rule, value, "
                "raised_ts) VALUES (?,?,?,?,?)",
                (characteristic, subgroup, rule, float(value),
                 ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def assign(self, event_id: int, who: str) -> None:
        self.conn.execute(
            "UPDATE ooc_event SET state='ASSIGNED', assignee=? "
            "WHERE event_id=? AND state='OPEN'", (who, event_id))
        self.conn.commit()

    def close(self, event_id: int, cause: str, action: str,
              disposition: str) -> None:
        if not cause:
            raise ValueError(
                "cannot close without an assignable cause -- a queue that can be "
                "emptied without naming causes produces no Pareto, and the Pareto "
                "is the point")
        self.conn.execute(
            "UPDATE ooc_event SET state='CLOSED', cause=?, action=?, "
            "disposition=?, closed_ts=? WHERE event_id=?",
            (cause, action, disposition,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event_id))
        self.conn.commit()

    def open_items(self, now_ts: float | None = None) -> list[dict]:
        """Open events WITH THEIR AGE, which is the reason persistence matters.

        An item open for three weeks is a different problem from the same item
        raised yesterday. An in-process queue cannot tell them apart, because it
        was born this morning.
        """
        now = now_ts or time.time()
        rows = []
        for r in self.conn.execute(
                "SELECT * FROM ooc_event WHERE state != 'CLOSED' "
                "ORDER BY raised_ts"):
            d = dict(r)
            try:
                raised = time.mktime(time.strptime(d["raised_ts"],
                                                   "%Y-%m-%dT%H:%M:%SZ"))
                d["age_days"] = max((now - raised) / 86400.0, 0.0)
            except (ValueError, OverflowError):
                d["age_days"] = float("nan")
            rows.append(d)
        return sorted(rows, key=lambda r: -(r["age_days"] or 0))

    def cause_pareto(self, weeks: int | None = None) -> list[tuple]:
        rows = self.conn.execute(
            "SELECT cause, COUNT(*) n FROM ooc_event WHERE cause IS NOT NULL "
            "GROUP BY cause ORDER BY n DESC").fetchall()
        return [(r["cause"], r["n"]) for r in rows]

    def record_capability(self, week: str, characteristic: str, cpk: float,
                          ppm: float, method: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO capability_history (week, characteristic, "
            "cpk, ppm, method) VALUES (?,?,?,?,?)",
            (week, characteristic, cpk, ppm, method))
        self.conn.commit()

    def capability_change(self, this_week: str, last_week: str) -> list[dict]:
        """Cpk movement week over week -- the column the report leads with."""
        cur = {r["characteristic"]: r for r in self.conn.execute(
            "SELECT * FROM capability_history WHERE week=?", (this_week,))}
        prev = {r["characteristic"]: r for r in self.conn.execute(
            "SELECT * FROM capability_history WHERE week=?", (last_week,))}
        out = []
        for name, r in cur.items():
            p = prev.get(name)
            out.append({
                "characteristic": name, "cpk": r["cpk"], "ppm": r["ppm"],
                "method": r["method"],
                "cpk_last_week": p["cpk"] if p else None,
                "delta": (r["cpk"] - p["cpk"]) if p and p["cpk"] is not None
                and r["cpk"] is not None else None,
            })
        # Ordered by the WORST movement, not by the worst value. A
        # characteristic sitting at 1.9 for a year is not news; one that fell
        # from 1.9 to 1.4 this week is the meeting.
        return sorted(out, key=lambda r: (r["delta"] if r["delta"] is not None
                                          else 0.0))

    def close_db(self) -> None:
        self.conn.close()


def render(path, *, week: str, changes: list[dict], open_items: list[dict],
           pareto: list[tuple], transforms: dict | None = None,
           gauge: dict | None = None, meta: dict | None = None) -> dict:
    """The weekly report, as a self-contained page."""
    import html

    def arrow(d):
        if d is None:
            return '<span class="mut">new</span>'
        if d < -0.05:
            return f'<span class="bad">▼ {d:+.2f}</span>'
        if d > 0.05:
            return f'<span class="ok">▲ {d:+.2f}</span>'
        return f'<span class="mut">— {d:+.2f}</span>'

    chg = "".join(
        f'<tr><td>{html.escape(c["characteristic"])}</td>'
        f'<td class="n">{c["cpk"]:.2f}</td>'
        f'<td class="n">{arrow(c["delta"])}</td>'
        f'<td class="n">{c["ppm"]:,.0f}</td>'
        f'<td>{html.escape(c["method"])}</td></tr>' for c in changes)

    items = "".join(
        f'<tr><td class="n">{r["event_id"]}</td>'
        f'<td>{html.escape(r["characteristic"])}</td>'
        f'<td>{html.escape(r["rule"])}</td>'
        f'<td class="n">{r["age_days"]:.1f}</td>'
        f'<td>{html.escape(r["state"])}</td>'
        f'<td>{html.escape(r["assignee"] or "—")}</td></tr>'
        for r in open_items[:25])

    par = "".join(f'<tr><td>{html.escape(c)}</td><td class="n">{n}</td></tr>'
                  for c, n in pareto)

    extra = ""
    if transforms:
        bc = transforms.get("boxcox", {})
        js = transforms.get("johnson_su", {})
        extra += f"""
  <div class="card">
    <h2>Skewed characteristic: which transformation works</h2>
    <table><thead><tr><th>fit</th><th class="n">skew after</th>
      <th>normal at 5%?</th></tr></thead><tbody>
      <tr><td>none (raw)</td><td class="n">{transforms['skew_before']:.2f}</td>
        <td>{'yes' if transforms['anderson_before']['normal'] else '<b>no</b>'}</td></tr>
      <tr><td>Box-Cox (λ={bc.get('lambda', float('nan')):.3f})</td>
        <td class="n">{bc.get('skew_after', float('nan')):.2f}</td>
        <td>{'<b>yes</b>' if bc.get('normal_after') else 'no'}</td></tr>
      <tr><td>Johnson S<sub>U</sub></td>
        <td class="n">{js.get('skew_after', float('nan')):.2f}</td>
        <td>{'<b>yes</b>' if js.get('normal_after') else 'no'}</td></tr>
    </tbody></table>
    <div class="note">A transformation is a hypothesis — "this map makes the data
     normal" — and shipping one without re-testing is the same error it was meant
     to fix. And on transformed data the runs rules become valid again, which is
     the real argument for transforming rather than judging stability on rule 1
     alone.</div>
  </div>"""
    if gauge and gauge.get("valid"):
        extra += f"""
  <div class="card">
    <h2>Gauge-corrected capability</h2>
    <div class="kpis">
      <div class="kpi"><b>{gauge['cpk_observed']:.2f}</b><span>Cpk as measured</span></div>
      <div class="kpi"><b>{gauge['cpk_gauge_corrected']:.2f}</b><span>Cpk of the process</span></div>
      <div class="kpi"><b>{gauge['gauge_share_of_variance'] * 100:.0f}%</b>
        <span>variance from the gauge</span></div>
    </div>
    <div class="note">{html.escape(gauge['verdict'])}. Cpk on measured values
     blames the process for the instrument, and those are different budgets and
     different teams.</div>
  </div>"""

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>Weekly quality report — {html.escape(week)}</title>
<style>
:root{{--bg:#f7fafc;--fg:#1a202c;--card:#fff;--line:#e2e8f0;--mut:#718096;
 --ok:#2f855a;--bad:#c53030}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171923;--fg:#e2e8f0;--card:#242c3d;
 --line:#3a4459;--mut:#a0aec0;--ok:#68d391;--bad:#fc8181}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);
 font:14px/1.55 system-ui,sans-serif;max-width:1100px}}
h1{{font-size:21px;margin:0 0 2px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);
 margin:0 0 10px}}
.sub{{color:var(--mut);margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:16px;margin-bottom:16px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-size:11px;text-transform:uppercase}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.ok{{color:var(--ok);font-weight:600}} .bad{{color:var(--bad);font-weight:600}}
.mut{{color:var(--mut)}}
.kpis{{display:flex;gap:22px;flex-wrap:wrap}}
.kpi b{{display:block;font-size:24px;line-height:1.1}}
.kpi span{{color:var(--mut);font-size:12px}}
.note{{font-size:12px;color:var(--mut);margin-top:10px}}
</style>
<h1>Weekly quality report</h1>
<div class="sub">week {html.escape(week)} &middot;
 {meta.get('n_characteristics', 0) if meta else 0} characteristics &middot;
 generated by <code>weekly.py</code>, not hand-edited</div>

<div class="card">
  <h2>1 — What changed</h2>
  <table><thead><tr><th>characteristic</th><th class="n">Cpk</th>
    <th class="n">vs last week</th><th class="n">PPM</th><th>method</th>
    </tr></thead><tbody>{chg}</tbody></table>
  <div class="note"><b>Ordered by movement, not by magnitude.</b> A
   characteristic sitting at Cpk 1.9 for a year is not news; one that fell from
   1.9 to 1.4 this week is the meeting. A report that leads with current values
   makes the reader do the differencing, and they will not.</div>
</div>

<div class="card">
  <h2>2 — What anyone is doing about it ({len(open_items)} open)</h2>
  <table><thead><tr><th class="n">#</th><th>characteristic</th><th>rule</th>
    <th class="n">age (days)</th><th>state</th><th>owner</th></tr></thead>
    <tbody>{items or '<tr><td colspan="6">nothing open</td></tr>'}</tbody></table>
  <div class="note"><b>Sorted oldest first.</b> Ageing is the single most useful
   thing a disposition queue does and it is impossible without persistence — an
   in-process queue is born every morning and cannot tell a three-week-old item
   from one raised yesterday.</div>
</div>

<div class="card">
  <h2>3 — The pattern</h2>
  <table><thead><tr><th>assignable cause</th><th class="n">events</th></tr>
    </thead><tbody>{par or '<tr><td colspan="2">no causes recorded</td></tr>'}
    </tbody></table>
  <div class="note">Individual OOC events are noise; the same cause five times is
   a project. This table only exists because closing an event requires naming a
   cause.</div>
</div>
{extra}
"""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return {"path": str(p), "bytes": p.stat().st_size, "week": week,
            "n_open": len(open_items), "self_contained": True}
