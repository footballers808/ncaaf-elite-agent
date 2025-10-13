from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Core pipeline (unchanged)
from .model import train_model, predict_games, save_model, load_model
from .labeler import label_latest_results

# Optional reporting modules (guarded)
try:
    from .metrics import classification_metrics, regression_metrics
    HAVE_METRICS = True
except Exception:
    HAVE_METRICS = False

try:
    from .reporter import render_html, to_csv_bytes
    HAVE_REPORTER = True
except Exception:
    HAVE_REPORTER = False

try:
    from .mailer import send_email_html
    HAVE_MAILER = True
except Exception:
    HAVE_MAILER = False

# Config loader
try:
    import yaml
except ImportError:
    yaml = None


# ------------------------- Config / IO helpers -------------------------

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    import csv
    headers = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _append_daily_history(rows: List[Dict[str, Any]], ts_utc: datetime, history_dir: str = "history"):
    """
    Appends this run's predictions to history/preds_YYYYMMDD.csv (one file per day).
    Creates the directory if needed, writes header only if file doesn't exist.
    """
    if not rows:
        return
    ensure_dir(history_dir)
    day_tag = ts_utc.strftime("%Y%m%d")
    path = os.path.join(history_dir, f"preds_{day_tag}.csv")
    import csv
    headers = sorted({k for r in rows for k in r.keys()})
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------- Edge tiers / stars -------------------------

def _get_edge_thresholds(cfg: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Returns thresholds dict:
      {"spread":[1.5,2.5,4.0], "total":[2.0,3.0,4.5]}
    You can override via config:
      edge_tiers:
        spread: [1.5, 2.5, 4.0]
        total:  [2.0, 3.0, 4.5]
    """
    tiers = (cfg.get("edge_tiers") or {})
    spread = tiers.get("spread") or [1.5, 2.5, 4.0]
    total  = tiers.get("total")  or [2.0, 3.0, 4.5]
    return {"spread": spread, "total": total}

def _stars_for_edge(edge_value: Optional[float], thresholds: List[float]) -> str:
    """
    0/1/2/3 stars based on absolute edge and ascending thresholds.
    """
    if edge_value is None:
        return ""
    try:
        v = abs(float(edge_value))
    except Exception:
        return ""
    if len(thresholds) < 3:
        thresholds = (thresholds + [float("inf"), float("inf")])[:3]
    t1, t2, t3 = thresholds[0], thresholds[1], thresholds[2]
    if v >= t3:
        return "★★★"
    if v >= t2:
        return "★★"
    if v >= t1:
        return "★"
    return ""

def _short_rationale(row: Dict[str, Any]) -> str:
    """
    Builds a compact rationale string if hints are present.
    Looks for common keys you already compute; falls back to a generic note.
    """
    reasons: List[str] = []
    # Weather
    if row.get("weather_note"):
        reasons.append(f"weather: {row['weather_note']}")
    elif row.get("weather_flag"):
        reasons.append("weather impact")

    # Injuries
    if row.get("injury_note"):
        reasons.append(f"injuries: {row['injury_note']}")
    elif row.get("injury_index") not in (None, "", 0):
        try:
            if float(row["injury_index"]) != 0:
                reasons.append("injury edge")
        except Exception:
            pass

    # Matchup / pace / macro
    if row.get("matchup_note"):
        reasons.append(f"matchup: {row['matchup_note']}")
    elif row.get("pace_note"):
        reasons.append(f"pace: {row['pace_note']}")
    elif row.get("macro_note"):
        reasons.append(f"macro: {row['macro_note']}")

    if not reasons:
        return "model vs market delta"
    return "; ".join(reasons[:2])  # keep it short


# ------------------------- Reporting glue -------------------------

def _build_top_plays(predictions: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns rows that have at least one star (spread or total), sorted by abs edge.
    Each row includes: stars, type, edge_text, matchup, market_text, model_text, rationale.
    """
    if not predictions:
        return []

    tiers = _get_edge_thresholds(cfg)
    out: List[Dict[str, Any]] = []

    for r in predictions:
        matchup = r.get("matchup") or f"{r.get('away_team','?')} @ {r.get('home_team','?')}"
        market_text = r.get("market_text", "")
        model_text  = r.get("model_text", "")
        rationale = _short_rationale(r)

        # Case A: a single edge with 'edge_type' + 'edge_value'
        et = r.get("edge_type")
        ev = r.get("edge_value")
        if et in ("spread", "total") and ev is not None:
            stars = _stars_for_edge(ev, tiers[et])
            if stars:
                out.append({
                    "matchup": matchup,
                    "stars": stars,
                    "edge_type": et,
                    "edge_text": f"{float(ev):+.2f}",
                    "edge_abs": abs(float(ev)),
                    "market_text": market_text,
                    "model_text": model_text,
                    "rationale": rationale,
                })

        # Case B: explicit split edges (edge_spread / edge_total)
        es = r.get("edge_spread")
        etot = r.get("edge_total")
        if es is not None:
            stars = _stars_for_edge(es, tiers["spread"])
            if stars:
                out.append({
                    "matchup": matchup,
                    "stars": stars,
                    "edge_type": "spread",
                    "edge_text": f"{float(es):+.2f}",
                    "edge_abs": abs(float(es)),
                    "market_text": r.get("market_text_spread", market_text),
                    "model_text":  r.get("model_text_spread",  model_text),
                    "rationale": rationale,
                })
        if etot is not None:
            stars = _stars_for_edge(etot, tiers["total"])
            if stars:
                out.append({
                    "matchup": matchup,
                    "stars": stars,
                    "edge_type": "total",
                    "edge_text": f"{float(etot):+.2f}",
                    "edge_abs": abs(float(etot)),
                    "market_text": r.get("market_text_total", market_text),
                    "model_text":  r.get("model_text_total",  model_text),
                    "rationale": rationale,
                })

    out.sort(key=lambda x: (len(x["stars"]), x["edge_abs"]), reverse=True)
    return out

def _health_counts(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Small observability block for the email.
    """
    total = len(predictions or [])
    with_market = sum(1 for r in (predictions or []) if r.get("market_text") or r.get("market_text_spread") or r.get("market_text_total"))
    starred = 0
    for r in predictions or []:
        if r.get("edge_type") in ("spread","total") and r.get("edge_value") not in (None, ""):
            starred += 1
            continue
        for key in ("edge_spread","edge_total"):
            if r.get(key) not in (None, ""):
                starred += 1
                break
    return {"total_games": total, "with_market": with_market, "starred_candidates": starred}

def _maybe_send_report(cfg: Dict[str, Any], ts_tag: str, predictions: List[Dict[str, Any]], labeled_rows: List[Dict[str, Any]]):
    report_cfg = (cfg.get("report") or {})
    if not report_cfg.get("enabled", False):
        print("ℹ️ Reporting disabled in config.")
        return
    if not (HAVE_METRICS and HAVE_REPORTER and HAVE_MAILER):
        print("ℹ️ Reporting modules not fully available; skipping email.")
        return

    n_labeled = len([r for r in labeled_rows if r.get("y_true") is not None])
    if n_labeled < int(report_cfg.get("min_games_for_report", 5)):
        print(f"ℹ️ Not enough labeled games ({n_labeled}) to send report.")
        return

    # Classification metrics
    y_true_cls, y_prob_cls = [], []
    for r in labeled_rows:
        yt, yp = r.get("y_true"), r.get("y_prob")
        if yt is None or yp is None:
            continue
        try:
            y_true_cls.append(int(yt))
            y_prob_cls.append(float(yp))
        except Exception:
            pass
    cls_m = classification_metrics(y_true_cls, y_prob_cls, threshold=0.5, bins=10) if (HAVE_METRICS and y_true_cls) else None

    # Regression metrics (optional keys)
    def collect_pair(rows, true_key, pred_key):
        t, p = [], []
        for rr in rows:
            if rr.get(true_key) is None or rr.get(pred_key) is None:
                continue
            try:
                t.append(float(rr[true_key])); p.append(float(rr[pred_key]))
            except Exception:
                continue
        return t, p
    spread_t, spread_p = collect_pair(labeled_rows, "spread_true", "spread_pred")
    total_t,  total_p  = collect_pair(labeled_rows, "total_true",  "total_pred")
    reg_spread = regression_metrics(spread_t, spread_p) if (HAVE_METRICS and spread_t) else None
    reg_total  = regression_metrics(total_t,  total_p)  if (HAVE_METRICS and total_t)  else None

    # Top Plays + Health
    top_plays = _build_top_plays(predictions, cfg)
    health = _health_counts(predictions)

    subject_prefix = report_cfg.get("subject_prefix", "[NCAAF Elite Agent]")
    subject = f"{subject_prefix} Run {ts_tag} — {n_labeled} labeled games"
    run_title = f"NCAAF Elite Agent — Run {ts_tag}"

    html = render_html(
        run_title=run_title,
        cls_metrics=cls_m,
        reg_metrics_spread=reg_spread,
        reg_metrics_total=reg_total,
        top_edges_table=None,      # keep available if you still use it
        top_plays=top_plays,       # NEW
        health=health              # NEW
    )

    # Attachments
    attachments = []
    if report_cfg.get("attach_predictions_csv", True) and predictions:
        fields = sorted({k for r in predictions for k in r.keys()})
        attachments.append((f"predictions_{ts_tag}.csv", to_csv_bytes(predictions, fields), "text/csv"))

    # Optional: attach a detailed edges CSV if you still build it elsewhere
    # if report_cfg.get("attach_edges_csv", True) and top_plays:
    #     fields = ["matchup","edge_type","edge_text","stars","market_text","model_text","rationale","edge_abs"]
    #     attachments.append((f"top_plays_{ts_tag}.csv", to_csv_bytes(top_plays, fields), "text/csv"))

    to = report_cfg.get("to") or []
    cc = report_cfg.get("cc") or []
    bcc = report_cfg.get("bcc") or []
    if not to:
        print("⚠️ report.to is empty; skipping email send.")
        return

    try:
        send_email_html(subject, html, to=to, cc=cc, bcc=bcc, attachments=attachments)
        print(f"✅ Sent report email to {to}")
    except Exception as e:
        print(f"⚠️ Email send failed: {e}")


# ------------------------- Main pipeline -------------------------

def run() -> int:
    cfg = load_config()
    out_dir = cfg.get("output_dir", "outputs")
    ensure_dir(out_dir)

    # 1) Predict (unchanged)
    predictions = predict_games(cfg)

    # 2) Label (unchanged)
    labeled_rows = label_latest_results(cfg, predictions)

    # 3) Learn (unchanged)
    model = load_model(cfg)
    model = train_model(cfg, model, labeled_rows)
    save_model(cfg, model)

    # 4) Save run CSVs
    ts_utc = datetime.now(timezone.utc)
    ts_tag = ts_utc.strftime("%Y%m%dT%H%M%SZ")
    if predictions:
        _write_csv(os.path.join(out_dir, f"predictions_{ts_tag}.csv"), predictions)
    if labeled_rows:
        _write_csv(os.path.join(out_dir, f"labels_{ts_tag}.csv"), labeled_rows)

    # 5) Append to daily history snapshot (NEW)
    _append_daily_history(predictions, ts_utc, history_dir="history")

    # 6) Optional reporting email (feature-flag + guards)
    _maybe_send_report(cfg, ts_tag, predictions, labeled_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
