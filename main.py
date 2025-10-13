from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .model import train_model, predict_games, save_model, load_model
from .labeler import label_latest_results
from .metrics import classification_metrics, regression_metrics
from .reporter import render_html, to_csv_bytes
from .mailer import send_email_html

try:
    import yaml
except ImportError:
    yaml = None

def load_config(path: str="config.yaml") -> Dict[str, Any]:
    if not yaml:
        raise RuntimeError("pyyaml not installed. Add pyyaml to your requirements.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run() -> int:
    cfg = load_config()
    out_dir = cfg.get("output_dir", "outputs")
    ensure_dir(out_dir)

    # 1) Predict
    predictions = predict_games(cfg)

    # 2) Label
    labeled_rows = label_latest_results(cfg, predictions)

    # 3) Learn
    model = load_model(cfg)
    model = train_model(cfg, model, labeled_rows)
    save_model(cfg, model)

    # 4) Save CSVs
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_csv_path = os.path.join(out_dir, f"predictions_{ts_tag}.csv")
    labels_csv_path = os.path.join(out_dir, f"labels_{ts_tag}.csv")

    if predictions:
        import csv
        with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in predictions for k in r.keys()}))
            w.writeheader()
            for r in predictions:
                w.writerow(r)
    if labeled_rows:
        import csv
        with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in labeled_rows for k in r.keys()}))
            w.writeheader()
            for r in labeled_rows:
                w.writerow(r)

    # ==== REPORTING START =======================================================
    report_cfg = (cfg.get("report") or {})
    if report_cfg.get("enabled", True):
        n_labeled = len([r for r in labeled_rows if r.get("y_true") is not None])
        if n_labeled >= int(report_cfg.get("min_games_for_report", 5)):
            y_true_cls = [int(r["y_true"]) for r in labeled_rows if r.get("y_true") is not None and r.get("y_prob") is not None]
            y_prob_cls = [float(r["y_prob"]) for r in labeled_rows if r.get("y_true") is not None and r.get("y_prob") is not None]
            cls_m = classification_metrics(y_true_cls, y_prob_cls, threshold=0.5, bins=10) if y_true_cls else None

            spread_true = [float(r["spread_true"]) for r in labeled_rows if r.get("spread_true") is not None and r.get("spread_pred") is not None]
            spread_pred = [float(r["spread_pred"]) for r in labeled_rows if r.get("spread_true") is not None and r.get("spread_pred") is not None]
            total_true  = [float(r["total_true"])  for r in labeled_rows if r.get("total_true")  is not None and r.get("total_pred")  is not None]
            total_pred  = [float(r["total_pred"])  for r in labeled_rows if r.get("total_true")  is not None and r.get("total_pred")  is not None]

            reg_spread = regression_metrics(spread_true, spread_pred) if spread_true else None
            reg_total  = regression_metrics(total_true, total_pred)   if total_true  else None

            top_k = int(report_cfg.get("top_edges_in_email", 10))
            edge_rows = []
            for r in predictions:
                ev = r.get("edge_value")
                et = r.get("edge_type")
                if ev is None or et is None:
                    continue
                edge_rows.append({
                    "matchup": r.get("matchup") or f"{r.get('home_team','?')} vs {r.get('away_team','?')}",
                    "market_text": r.get("market_text") or "",
                    "model_text": r.get("model_text") or "",
                    "edge_text": f"{float(ev):+.2f}",
                    "edge_type": et,
                    "edge_abs": abs(float(ev)),
                })
            edge_rows.sort(key=lambda x: x["edge_abs"], reverse=True)
            top_edges = edge_rows[:top_k] if edge_rows else None

            subject_prefix = report_cfg.get("subject_prefix", "[NCAAF]")
            subject = f"{subject_prefix} Run {ts_tag} — {n_labeled} labeled games"
            run_title = f"NCAAF Elite Agent — Run {ts_tag}"

            html = render_html(
                run_title=run_title,
                cls_metrics=cls_m,
                reg_metrics_spread=reg_spread,
                reg_metrics_total=reg_total,
                top_edges_table=top_edges
            )

            attachments = []
            if report_cfg.get("attach_predictions_csv", True) and predictions:
                fields = sorted({k for r in predictions for k in r.keys()})
                attachments.append((
                    f"predictions_{ts_tag}.csv",
                    to_csv_bytes(predictions, fields),
                    "text/csv"
                ))
            if report_cfg.get("attach_edges_csv", True) and edge_rows:
                fields = ["matchup","market_text","model_text","edge_type","edge_text","edge_abs"]
                attachments.append((
                    f"edges_{ts_tag}.csv",
                    to_csv_bytes(edge_rows, fields),
                    "text/csv"
                ))

            to = report_cfg.get("to") or []
            cc = report_cfg.get("cc") or []
            bcc = report_cfg.get("bcc") or []
            if not to:
                print("⚠️ report.to is empty; skipping email send.")
            else:
                send_email_html(subject, html, to=to, cc=cc, bcc=bcc, attachments=attachments)
                print(f"✅ Sent report email to {to}")
        else:
            print(f"ℹ️ Not enough labeled games ({n_labeled}) to send report.")
    # ==== REPORTING END =========================================================

    return 0

if __name__ == "__main__":
    raise SystemExit(run())
