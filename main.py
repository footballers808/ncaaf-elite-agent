from __future__ import annotations
import os, importlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

# ------------------------- Config loader -------------------------
try:
    import yaml
except ImportError:
    yaml = None

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    headers = sorted({k for r in rows for k in r.keys()}) if rows else []
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")  # create empty file so artifacts still upload
            return
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _append_daily_history(rows: List[Dict[str, Any]], ts_utc: datetime, history_dir: str = "history"):
    import csv
    ensure_dir(history_dir)
    day_tag = ts_utc.strftime("%Y%m%d")
    path = os.path.join(history_dir, f"preds_{day_tag}.csv")
    headers = sorted({k for r in rows for k in r.keys()}) if rows else []
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        if not rows:
            if not file_exists:
                f.write("")  # touch
            return
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

# ------------------------- Optional reporting imports (guarded) -------------------------
try:
    from src.metrics import classification_metrics, regression_metrics
    HAVE_METRICS = True
except Exception:
    HAVE_METRICS = False

try:
    from src.reporter import render_html, to_csv_bytes
    HAVE_REPORTER = True
except Exception:
    HAVE_REPORTER = False

try:
    from src.mailer import send_email_html
    HAVE_MAILER = True
except Exception:
    HAVE_MAILER = False

# ------------------------- Resolution helpers -------------------------
def _resolve(module_name: str, candidates: List[str]) -> Optional[Callable]:
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"⚠️ Could not import {module_name}: {e}")
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"↪ Using {module_name}.{name}")
            return fn
    print(f"⚠️ None of the candidates exist in {module_name}: {candidates}")
    return None

def _resolve_model_api():
    predict_fn = _resolve("src.model", ["predict_games", "predict", "run_predict", "make_predictions"])
    train_fn   = _resolve("src.model", ["train_model", "train", "learn", "fit_model", "fit"])
    save_fn    = _resolve("src.model", ["save_model", "save", "persist_model"])
    load_fn    = _resolve("src.model", ["load_model", "load", "load_or_init_model", "init_model"])
    return predict_fn, train_fn, save_fn, load_fn

def _resolve_labeler_api():
    label_fn = _resolve("src.labeler", ["label_latest_results", "label_results", "build_labels", "label"])
    return label_fn

# ------------------------- Stars / edge helpers -------------------------
def _get_edge_thresholds(cfg: Dict[str, Any]) -> Dict[str, List[float]]:
    tiers = (cfg.get("edge_tiers") or {})
    spread = tiers.get("spread") or [1.5, 2.5, 4.0]
    total  = tiers.get("total")  or [2.0, 3.0, 4.5]
    return {"spread": spread, "total": total}

def _stars_for_edge(edge_value: Optional[float], thresholds: List[float]) -> str:
    if edge_value is None:
        return ""
    try:
        v = abs(float(edge_value))
    except Exception:
        return ""
    if len(thresholds) < 3:
        thresholds = (thresholds + [float("inf"), float("inf")])[:3]
    t1, t2, t3 = thresholds[0], thresholds[1], thresholds[2]
    if v >= t3: return "★★★"
    if v >= t2: return "★★"
    if v >= t1: return "★"
    return ""

def _short_rationale(row: Dict[str, Any]) -> str:
    reasons: List[str] = []
    if row.get("weather_note"): reasons.append(f"weather: {row['weather_note']}")
    elif row.get("weather_flag"): reasons.append("weather impact")
    if row.get("injury_note"): reasons.append(f"injuries: {row['injury_note']}")
    elif row.get("injury_index") not in (None, "", 0):
        try:
            if float(row["injury_index"]) != 0: reasons.append("injury edge")
        except Exception:
            pass
    if row.get("matchup_note"): reasons.append(f"matchup: {row['matchup_note']}")
    elif row.get("pace_note"): reasons.append(f"pace: {row['pace_note']}")
    elif row.get("macro_note"): reasons.append(f"macro: {row['macro_note']}")
    return "; ".join(reasons[:2]) if reasons else "model vs market delta"

def _build_top_plays(predictions: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not predictions: return []
    tiers = _get_edge_thresholds(cfg)
    out: List[Dict[str, Any]] = []
    for r in predictions:
        matchup = r.get("matchup") or f"{r.get('away_team','?')} @ {r.get('home_team','?')}"
        market_text = r.get("market_text", "")
        model_text  = r.get("model_text", "")
        rationale = _short_rationale(r)
        et = r.get("edge_type"); ev = r.get("edge_value")
        if et in ("spread","total") and ev is not None:
            stars = _stars_for_edge(ev, tiers[et])
            if stars:
                out.append({"matchup": matchup, "stars": stars, "edge_type": et,
                            "edge_text": f"{float(ev):+.2f}", "edge_abs": abs(float(ev)),
                            "market_text": market_text, "model_text": model_text, "rationale": rationale})
        es = r.get("edge_spread"); etot = r.get("edge_total")
        if es is not None:
            s = _stars_for_edge(es, tiers["spread"])
            if s:
                out.append({"matchup": matchup, "stars": s, "edge_type": "spread",
                            "edge_text": f"{float(es):+.2f}", "edge_abs": abs(float(es)),
                            "market_text": r.get("market_text_spread", market_text),
                            "model_text": r.get("model_text_spread", model_text),
                            "rationale": rationale})
        if etot is not None:
            s = _stars_for_edge(etot, tiers["total"])
            if s:
                out.append({"matchup": matchup, "stars": s, "edge_type": "total",
                            "edge_text": f"{float(etot):+.2f}", "edge_abs": abs(float(etot)),
                            "market_text": r.get("market_text_total", market_text),
                            "model_text": r.get("model_text_total", model_text),
                            "rationale": rationale})
    out.sort(key=lambda x: (len(x["stars"]), x["edge_abs"]), reverse=True)
    return out

def _health_counts(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(predictions or [])
    with_market = sum(1 for r in (predictions or []) if r.get("market_text") or r.get("market_text_spread") or r.get("market_text_total"))
    starred = 0
    for r in predictions or []:
        if r.get("edge_type") in ("spread","total") and r.get("edge_value") not in (None, ""):
            starred += 1; continue
        for key in ("edge_spread","edge_total"):
            if r.get(key) not in (None, ""):
                starred += 1; break
    return {"total_games": total, "with_market": with_market, "starred_candidates": starred}

# ------------------------- Reporting wrapper -------------------------
def _maybe_send_report(cfg: Dict[str, Any], ts_tag: str, predictions: List[Dict[str, Any]], labeled_rows: List[Dict[str, Any]]):
    report_cfg = (cfg.get("report") or {})
    if not report_cfg.get("enabled", False):
        print("ℹ️ Reporting disabled in config."); return
    if not (HAVE_METRICS and HAVE_REPORTER and HAVE_MAILER):
        print("ℹ️ Reporting modules not fully available; skipping email."); return

    # Allow forcing a test email by setting min_games_for_report: 0
    n_labeled = len([r for r in labeled_rows if r.get("y_true") is not None])
    min_games = int(report_cfg.get("min_games_for_report", 5))
    if n_labeled < min_games:
        print(f"ℹ️ Not enough labeled games ({n_labeled}) to send report (min={min_games})."); 
        if min_games > 0:
            return  # respect threshold

    # Metrics (safe even with empty lists)
    cls_m = None; reg_spread = None; reg_total = None
    if HAVE_METRICS:
        y_true_cls = [int(r["y_true"]) for r in labeled_rows if r.get("y_true") is not None and r.get("y_prob") is not None]
        y_prob_cls = [float(r["y_prob"]) for r in labeled_rows if r.get("y_true") is not None and r.get("y_prob") is not None]
        if y_true_cls:
            cls_m = classification_metrics(y_true_cls, y_prob_cls, threshold=0.5, bins=10)
        spread_t = [float(r["spread_true"]) for r in labeled_rows if r.get("spread_true") is not None and r.get("spread_pred") is not None]
        spread_p = [float(r["spread_pred"]) for r in labeled_rows if r.get("spread_true") is not None and r.get("spread_pred") is not None]
        total_t  = [float(r["total_true"])  for r in labeled_rows if r.get("total_true")  is not None and r.get("total_pred")  is not None]
        total_p  = [float(r["total_pred"])  for r in labeled_rows if r.get("total_true")  is not None and r.get("total_pred")  is not None]
        if spread_t: reg_spread = regression_metrics(spread_t, spread_p)
        if total_t:  reg_total  = regression_metrics(total_t,  total_p)

    top_plays = _build_top_plays(predictions, cfg)
    health = _health_counts(predictions)

    subject_prefix = report_cfg.get("subject_prefix", "[NCAAF Elite Agent]")
    subject = f"{subject_prefix} Run {ts_tag} — {n_labeled} labeled games"
    run_title = f"NCAAF Elite Agent — Run {ts_tag}"

    html = render_html(run_title=run_title,
                       cls_metrics=cls_m,
                       reg_metrics_spread=reg_spread,
                       reg_metrics_total=reg_total,
                       top_edges_table=None,
                       top_plays=top_plays,
                       health=health)

    attachments = []
    if report_cfg.get("attach_predictions_csv", True) and predictions:
        fields = sorted({k for r in predictions for k in r.keys()})
        attachments.append((f"predictions_{ts_tag}.csv", to_csv_bytes(predictions, fields), "text/csv"))

    to = report_cfg.get("to") or []; cc = report_cfg.get("cc") or []; bcc = report_cfg.get("bcc") or []
    if not to: 
        print("⚠️ report.to is empty; skipping email send."); 
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
    os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH','')}"

    # Try to resolve your functions, but don't fail the run if absent
    predict_fn, train_fn, save_fn, load_fn = _resolve_model_api()
    label_fn = _resolve_labeler_api()

    predictions: List[Dict[str, Any]] = []
    labeled_rows: List[Dict[str, Any]] = []

    # 1) Predict (optional)
    if predict_fn:
        try:
            predictions = predict_fn(cfg)
        except Exception as e:
            print(f"⚠️ predict failed: {e}")

    # 2) Label (optional)
    if label_fn:
        try:
            labeled_rows = label_fn(cfg, predictions)
        except Exception as e:
            print(f"⚠️ label failed: {e}")

    # 3) Learn (optional)
    if load_fn and train_fn and save_fn:
        try:
            model = load_fn(cfg)
        except Exception as e:
            print(f"⚠️ load_model failed: {e}")
            model = None
        try:
            model = train_fn(cfg, model, labeled_rows)
        except Exception as e:
            print(f"⚠️ train_model failed: {e}")
        try:
            save_fn(cfg, model)
        except Exception as e:
            print(f"⚠️ save_model failed: {e}")

    # 4) Persist CSVs
    ts_utc = datetime.now(timezone.utc)
    ts_tag = ts_utc.strftime("%Y%m%dT%H%M%SZ")
    _write_csv(os.path.join(out_dir, f"predictions_{ts_tag}.csv"), predictions)
    _write_csv(os.path.join(out_dir, f"labels_{ts_tag}.csv"), labeled_rows)

    # 5) Daily history
    _append_daily_history(predictions, ts_utc, history_dir="history")

    # 6) Reporting (optional; will still send if min_games_for_report == 0)
    _maybe_send_report(cfg, ts_tag, predictions, labeled_rows)

    print("✅ Pipeline completed (tolerant runner).")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
