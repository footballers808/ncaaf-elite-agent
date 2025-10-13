from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import io
import csv
import json

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
    top_edges_table: Optional[List[Dict[str, Any]]] = None,
) -> str:
    styles = """
    <style>
      body { font-family: Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#111; }
      h1 { font-size: 18px; margin-bottom: 8px; }
      h2 { font-size: 16px; margin: 18px 0 6px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: left; }
      th { background: #f4f6f8; }
      .muted { color: #666; }
      .kpi { font-size: 14px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      .pill { padding:2px 6px; border-radius:10px; background:#eef5ff; border:1px solid #cfe1ff; }
    </style>
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = [f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>"]
    html.append(f"<h1>{run_title}</h1>")
    html.append(f"<div class='muted'>Generated {ts}</div>")

    if cls_metrics:
        html.append("<h2>Win/ML Classification</h2>")
        html.append("<table class='kpi'><tbody>")
        html.append(f"<tr><th>Games</th><td>{cls_metrics.n}</td><th>Accuracy</th><td>{_fmt_pct(cls_metrics.accuracy)}</td></tr>")
        html.append(f"<tr><th>Brier</th><td>{_fmt_float(cls_metrics.brier)}</td><th>LogLoss</th><td>{_fmt_float(cls_metrics.logloss)}</td></tr>")
        c = cls_metrics.confusion
        html.append(f"<tr><th>Confusion</th><td colspan='3' class='mono'>TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}</td></tr>")
        html.append("</tbody></table>")

        html.append("<details><summary>Calibration by probability bin</summary>")
        html.append("<table><thead><tr><th>Prob bin</th><th>Count</th><th>Avg Pred</th><th>Emp Rate</th></tr></thead><tbody>")
        for row in cls_metrics.calibration:
            rng = f"{row['lo']:.1f}–{row['hi']:.1f}"
            html.append(f"<tr><td>{rng}</td><td>{row['count']}</td><td>{_fmt_float(row['avg_pred'])}</td><td>{_fmt_float(row['emp_rate'])}</td></tr>")
        html.append("</tbody></table></details>")

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
