"""Control charts rendered, with violations annotated and a disposition queue.

The spec's red flag is "violations get investigated and dispositioned, not just
displayed", and the earlier passes displayed nothing at all -- everything was a
table in a markdown file.

WHAT A CONTROL CHART HAS TO SHOW to be worth rendering, beyond the points:

  THE ZONES. A, B and C at 1, 2 and 3 sigma. Most of the Western Electric rules
  are ABOUT the zones, so a chart drawn without them cannot be read by the rules
  it is being judged against -- the operator sees a flagged point and no reason.

  WHICH RULE FIRED, per point. "Out of control" is not actionable; "rule 3, six
  increasing" is, because it names tool wear and a different response than a
  single spike does.

  THE PHASE BOUNDARY. Where the limits were established, and where they started
  being applied prospectively. Without it a reader cannot tell which points
  helped set the limits they are being judged against.

Self-contained: one file, inline SVG, no CDN.
"""
from __future__ import annotations

import html
import pathlib


def _chart_svg(stat, center, ucl, lcl, violations, *, title, baseline_n=None,
               width=880, height=260, spec=None):
    pad_l, pad_r, pad_t, pad_b = 56, 16, 22, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    lo = min(min(stat), lcl, *( [spec[0]] if spec else []))
    hi = max(max(stat), ucl, *( [spec[1]] if spec else []))
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08

    def y(v):
        return pad_t + ph * (1 - (v - lo) / (hi - lo))

    def x(i):
        return pad_l + pw * (i / max(len(stat) - 1, 1))

    sigma = (ucl - center) / 3.0
    bands = []
    for k, cls in ((1, "zc"), (2, "zb"), (3, "za")):
        top, bot = center + k * sigma, center - k * sigma
        prev = (k - 1) * sigma
        bands.append(
            f'<rect x="{pad_l}" y="{y(top):.1f}" width="{pw}" '
            f'height="{max(y(center + prev) - y(top), 0):.1f}" class="{cls}"/>'
            f'<rect x="{pad_l}" y="{y(center - prev):.1f}" width="{pw}" '
            f'height="{max(y(bot) - y(center - prev), 0):.1f}" class="{cls}"/>')

    lines = [f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(center):.1f}" '
             f'y2="{y(center):.1f}" class="cl"/>',
             f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(ucl):.1f}" '
             f'y2="{y(ucl):.1f}" class="lim"/>',
             f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(lcl):.1f}" '
             f'y2="{y(lcl):.1f}" class="lim"/>']
    if spec:
        for sv, lbl in ((spec[0], "LSL"), (spec[1], "USL")):
            lines.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(sv):.1f}" '
                         f'y2="{y(sv):.1f}" class="spec"/>'
                         f'<text x="{width - pad_r - 2}" y="{y(sv) - 3:.1f}" '
                         f'text-anchor="end" class="ax">{lbl}</text>')
    if baseline_n:
        bx = x(baseline_n)
        lines.append(f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{pad_t}" '
                     f'y2="{pad_t + ph}" class="phase"/>'
                     f'<text x="{bx + 4:.1f}" y="{pad_t + 11}" class="ax">'
                     f'phase II &rarr;</text>')

    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}"
                    for i, v in enumerate(stat))
    dots = []
    for i, v in enumerate(stat):
        rules = violations.get(i)
        cls = "pt bad" if rules else "pt"
        t = f" ({', '.join(rules)})" if rules else ""
        dots.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="{3.6 if rules else 2.2}" '
                    f'class="{cls}"><title>#{i}: {v:.3f}{html.escape(t)}</title></circle>')

    ticks = "".join(
        f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end" class="ax">'
        f'{v:.2f}</text>' for v in (lcl, center, ucl))
    return (f'<div class="ct"><div class="ctitle">{html.escape(title)}</div>'
            f'<svg viewBox="0 0 {width} {height}" class="chart">'
            + "".join(bands) + "".join(lines) + ticks
            + f'<path d="{path}" class="ln"/>' + "".join(dots) + "</svg></div>")


def render(path: pathlib.Path, *, charts: list[dict], queue_summary: dict,
           events: list, limits_history: list, capability: list[dict],
           meta: dict) -> dict:
    chart_html = "".join(
        _chart_svg(c["stat"], c["center"], c["ucl"], c["lcl"],
                   c.get("violations", {}), title=c["title"],
                   baseline_n=c.get("baseline_n"), spec=c.get("spec"))
        for c in charts)

    ev_rows = "".join(
        f'<tr><td>{e.subgroup}</td><td>{html.escape(e.rule)}</td>'
        f'<td class="n">{e.value:.3f}</td>'
        f'<td><span class="st {e.state.lower()}">{e.state}</span></td>'
        f'<td>{html.escape(e.cause or "—")}</td>'
        f'<td>{html.escape(e.disposition or "—")}</td></tr>'
        for e in events)

    pareto = "".join(
        f'<tr><td>{html.escape(c)}</td><td class="n">{n}</td></tr>'
        for c, n in queue_summary.get("cause_pareto", []))

    cap_rows = "".join(
        f'<tr><td>{html.escape(c["characteristic"])}</td>'
        f'<td>{html.escape(c["method"])}</td>'
        f'<td class="n">{c["cpk"]:.2f}</td><td class="n">{c["ppm"]:.0f}</td>'
        f'<td>{html.escape(c["verdict"])}</td></tr>'
        for c in capability)

    hist = "".join(
        f'<tr><td>rev {h["revision"]}</td><td class="n">{h["center"]:.3f}</td>'
        f'<td class="n">{h["lcl"]:.3f} – {h["ucl"]:.3f}</td>'
        f'<td>{html.escape(h["reason"])}</td></tr>'
        for h in limits_history)

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>SPC</title>
<style>
:root {{ --bg:#f7fafc; --fg:#1a202c; --card:#fff; --line:#e2e8f0; --mut:#718096;
         --za:#fde8e8; --zb:#fdf6e3; --zc:#eef6ee; --bad:#c53030; --ok:#2f855a; }}
@media (prefers-color-scheme: dark) {{
 :root {{ --bg:#171923; --fg:#e2e8f0; --card:#242c3d; --line:#3a4459; --mut:#a0aec0;
          --za:#3b2226; --zb:#33301f; --zc:#1f2c22; --bad:#fc8181; --ok:#68d391; }} }}
*{{box-sizing:border-box}}
body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
        font:14px/1.55 system-ui,-apple-system,sans-serif; }}
h1 {{ font-size:20px; margin:0 0 2px; }}
h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.6px;
      color:var(--mut); margin:0 0 10px; }}
.sub {{ color:var(--mut); margin-bottom:20px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:16px; margin-bottom:16px; overflow-x:auto; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }}
.chart {{ width:100%; height:auto; }}
.ctitle {{ font-size:12px; color:var(--mut); margin:6px 0 2px; }}
rect.za{{fill:var(--za)}} rect.zb{{fill:var(--zb)}} rect.zc{{fill:var(--zc)}}
line.cl{{stroke:var(--mut);stroke-width:1}}
line.lim{{stroke:var(--bad);stroke-width:1.2;stroke-dasharray:5 3}}
line.spec{{stroke:#805ad5;stroke-width:1;stroke-dasharray:2 3}}
line.phase{{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3}}
path.ln{{fill:none;stroke:var(--fg);stroke-width:1.1;opacity:.75}}
circle.pt{{fill:var(--fg)}} circle.bad{{fill:var(--bad)}}
text.ax{{fill:var(--mut);font-size:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-size:11px;text-transform:uppercase}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.st{{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line)}}
.st.closed{{color:var(--ok);border-color:var(--ok)}}
.st.open,.st.assigned{{color:var(--bad);border-color:var(--bad)}}
.legend{{font-size:12px;color:var(--mut);margin-top:8px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;
     vertical-align:-1px;margin:0 4px 0 12px}}
</style>
<h1>SPC — {html.escape(meta.get('title', ''))}</h1>
<div class="sub">{html.escape(meta.get('subtitle', ''))} &middot; generated by
 <code>complete.py</code>, not hand-edited</div>

<div class="card">
  <h2>Control charts — zones drawn, violations annotated</h2>
  {chart_html}
  <div class="legend">
    <span class="sw" style="background:var(--zc)"></span>zone C (±1σ)
    <span class="sw" style="background:var(--zb)"></span>zone B (±2σ)
    <span class="sw" style="background:var(--za)"></span>zone A (±3σ)
    &nbsp;— most Western Electric rules are <em>about</em> the zones, so a chart
    drawn without them cannot be read by the rules judging it. Hover a point for
    the rules it violated.
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Disposition queue — {queue_summary['n']} events,
        {queue_summary['open']} open</h2>
    <table><thead><tr><th>#</th><th>rule</th><th class="n">value</th>
      <th>state</th><th>cause</th><th>disposition</th></tr></thead>
      <tbody>{ev_rows}</tbody></table>
    <div class="legend">Closing requires an assignable cause. Without that rule
      the queue becomes a list of things somebody clicked away, and the Pareto
      below — the most valuable output of SPC — is never produced.</div>
  </div>

  <div class="card">
    <h2>Assignable-cause Pareto</h2>
    <table><thead><tr><th>cause</th><th class="n">events</th></tr></thead>
      <tbody>{pareto or '<tr><td colspan="2">no causes recorded yet</td></tr>'}</tbody></table>
  </div>

  <div class="card">
    <h2>Capability</h2>
    <table><thead><tr><th>characteristic</th><th>method</th><th class="n">Cpk/Ppk</th>
      <th class="n">PPM</th><th>verdict</th></tr></thead>
      <tbody>{cap_rows}</tbody></table>
  </div>

  <div class="card">
    <h2>Limit revision history</h2>
    <table><thead><tr><th>revision</th><th class="n">centre</th>
      <th class="n">limits</th><th>reason</th></tr></thead>
      <tbody>{hist or '<tr><td colspan="4">no revisions</td></tr>'}</tbody></table>
    <div class="legend">Limits are never revised because the chart is alarming.
      That is the process talking and the response being to turn down the volume.</div>
  </div>
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size,
            "n_charts": len(charts), "self_contained": True}
