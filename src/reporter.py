from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import io
import csv

from .metrics import ClassificationMetrics, RegressionMetrics


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x*100:.1f}%"

def _fmt_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.3f}"


def render_html(
    run_title: str,
    cls_metrics: Optional[ClassificationMetrics],
    reg_metrics_spread: Optional[RegressionMetrics],
    reg_metrics_total: Optional[RegressionMetrics],
    top_edges_table: Optional[List[Dict[str, Any]]] = None,  # kept for backward compatibility
    top_plays: Optional[List[Dict[str, Any]]] = None,        # NEW: starred edges summary
    health: Optional[Dict[str, int]] = None,                 # NEW: tiny health table
) -> str:
    styles = """
    <style>
      body { font-family: Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#111; }
      h1 { font-size: 18px; margin-bottom: 8px; }
      h2 { font-size: 16px; margin: 18px 0 6px; }
      h3 { font-size: 15px; margin: 14px 0 6px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: left; vertical-align: top; }
      th { background: #f4f6f8; }
      .muted { color: #666; }
      .kpi { font-size: 14px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      .pill { padding:2px 6px; border-radius:10px; background:#eef5ff; border:1px solid #cfe1ff; }
      .stars { font-size: 14px; }
    </style>
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = [f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>"]
    html.append(f"<h1>{run_title}</h1>")
    html.append(f"<div class='muted'>Generated {ts}</div>")

    # ============ Top Plays (stars) ============
    if top_plays:
        html.append("<h2>Top Plays</h2>")
        html.append("<table><thead><tr><th>⭐</th><th>Game</th><th>Type</th><th>Edge</th><th>Model</th><th>Market</th><th>Rationale</th></tr></thead><tbody>")
        for r in top_plays:
            html.append(
                f"<tr>"
                f"<td class='stars'>{r.get('stars','')}</td>"
                f"<td>{r.get('matchup','')}</td>"
                f"<td>{r.get('edge_type','')}</td>"
                f"<td><span class='pill'>{r.get('edge_text','')}</span></td>"
                f"<td>{r.get('model_text','')}</td>"
                f"<td>{r.get('market_text','')}</td>"
                f"<td class='muted'>{r.get('rationale','')}</td>"
                f"</tr>"
            )
        html.append("</tbody></table>")

    # ============ Health (small) ============
    if health:
        html.append("<h3>Run Health</h3>")
        html.append("<table class='kpi'><tbody>")
        html.append(f"<tr><th>Total games</th><td>{int(health.get('total_games',0))}</td>"
                    f"<th>With market</th><td>{int(health.get('with_market',0))}</td>"
                    f"<th>Starred candidates</th><td>{int(health.get('starred_candidates',0))}</td></tr>")
        html.append("</tbody></table>")

    # ============ Classification KPIs ============
    if cls_metrics:
        html.append("<h2>Win/ML Classification</h2>")
        html.append("<table class='kpi'><tbody>")
        html.append(f"<tr><th>Games</th><td>{cls_metrics.n}</td><th>Accuracy</th><td>{_fmt_pct(cls_metrics.accuracy)}</td></tr>")
        html.append(f"<tr><th>Brier</th><td>{_fmt_float(cls_metrics.brier)}</td><th>LogLoss</th><td>{_fmt_float(cls_metrics.logloss)}</td></tr>")
        c = cls_metrics.confusion
        html.append(f"<tr><th>Confusion</th><td colspan='3' class='mono'>TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}</td></tr>")
        html.append("</tbody></table>")

        # Calibration details
        html.append("<details><summary>Calibration by probability bin</summary>")
        html.append("<table><thead><tr><th>Prob bin</th><th>Count</th><th>Avg Pred</th><th>Emp Rate</th></tr></thead><tbody>")
        for row in cls_metrics.calibration:
            rng = f"{row['lo']:.1f}–{row['hi']:.1f}"
            html.append(f"<tr><td>{rng}</td><td>{row['count']}</td><td>{_fmt_float(row['avg_pred'])}</td><td>{_fmt_float(row['emp_rate'])}</td></tr>")
        html.append("</tbody></table></details>")

    # ============ Regression KPIs ============
    if reg_metrics_spread or reg_metrics_total:
        html.append("<h2>Spread/Total Regression</h2>")
        html.append("<table class='kpi'><thead><tr><th></th><th>n</th><th>MAE</th><th>RMSE</th><th>Mean Err</th></tr></thead><tbody>")
        if reg_metrics_spread:
            r = reg_metrics_spread
            html.append(f"<tr><th>Spread</th><td>{r.n}</td><td>{_fmt_float(r.mae)}</td><td>{_fmt_float(r.rmse)}</td><td>{_fmt_float(r.mean_err)}</td></tr>")
        if reg_metrics_total:
            r = reg_metrics_total
            html.append(f"<tr><th>Total</th><td>{r.n}</td><td>{_fmt_float(r.mae)}</td><td>{_fmt_float(r.rmse)}</td><td>{_fmt_float(r.mean_err)}</td></tr>")
        html.append("</tbody></table>")

    # (Optional legacy) Top value edges table
    if top_edges_table:
        html.append("<h2>Top Value Edges</h2>")
        html.append("<table><thead><tr><th>Game</th><th>Market</th><th>Model</th><th>Edge</th><th>Type</th></tr></thead><tbody>")
        for row in top_edges_table:
            html.append(
                f"<tr><td>{row.get('matchup','')}</td>"
                f"<td>{row.get('market_text','')}</td>"
                f"<td>{row.get('model_text','')}</td>"
                f"<td><span class='pill'>{row.get('edge_text','')}</span></td>"
                f"<td>{row.get('edge_type','')}</td></tr>"
            )
        html.append("</tbody></table>")

    html.append("<p class='muted'>End of report.</p>")
    html.append("</body></html>")
    return "".join(html)


def to_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")
