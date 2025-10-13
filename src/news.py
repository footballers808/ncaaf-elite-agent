from __future__ import annotations
import json
import os
from typing import Dict, Any

# ------------------------------------------------------------
# News / Video signal adapter (CI-safe, no external calls)
# ------------------------------------------------------------
# Sources of signal (all optional; if missing, returns 0.0):
# 1) store/news_overrides.json      -> {"Team Name": 0.6, "Other Team": -0.4}
# 2) store/news_cache/<Team>.txt    -> free-form notes (we do simple keyword sentiment)
#
# Ranges:
# - get_team_sentiment() returns a float in [-1.0, +1.0].
# - You can blend this into spread/total using config weights/caps.
#
# Why this design?
# - It never breaks CI (no HTTP or API keys required).
# - Lets you hand-enter one-off items from beat writers/videos.
# - We can later swap in real feeds (Google News API, YouTube Data API, etc.)
#   by replacing fetchers here without touching the rest of the pipeline.

_POS = {
    "return", "returns", "cleared", "healthy", "upgrade", "upgraded",
    "probable", "available", "practiced", "starting", "start", "back",
    "activated", "improved", "improving", "boost", "boosted", "favorable",
}
_NEG = {
    "out", "doubtful", "questionable", "injured", "injury", "surgery",
    "tear", "broken", "fracture", "hamstring", "ankle", "knee", "concussion",
    "suspended", "ruled out", "won't play", "won’t play", "not playing",
    "miss", "missing", "limited", "setback", "downgrade", "downgraded",
}

def _load_overrides() -> Dict[str, float]:
    path = os.path.join("store", "news_overrides.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        # clamp to [-1,1]
        out = {}
        for k, v in data.items():
            try:
                x = float(v)
                if x > 1.0: x = 1.0
                if x < -1.0: x = -1.0
                out[str(k)] = x
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"⚠️ failed to read news_overrides.json: {e}")
        return {}

def _score_free_text(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POS if w in t)
    neg = sum(1 for w in _NEG if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    # normalize to [-1,1]
    s = (pos - neg) / max(1.0, pos + neg)
    if s > 1.0: s = 1.0
    if s < -1.0: s = -1.0
    return s

def _load_cache_text(team: str) -> str:
    # store/news_cache/<Team>.txt
    cache_dir = os.path.join("store", "news_cache")
    path = os.path.join(cache_dir, f"{team}.txt")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def get_team_sentiment(team: str, cfg: Dict[str, Any]) -> float:
    """
    Returns a sentiment score in [-1, +1] for a given team.
    Priority: overrides JSON (explicit) > cached text (keyword score) > 0.0
    Respects cfg["news"]["enabled"] (if explicitly false -> 0.0).
    """
    news_cfg = (cfg.get("news") or {})
    if news_cfg.get("enabled") is False:
        return 0.0

    if not team:
        return 0.0

    overrides = _load_overrides()
    if team in overrides:
        return float(overrides[team])

    txt = _load_cache_text(team)
    return _score_free_text(txt)
