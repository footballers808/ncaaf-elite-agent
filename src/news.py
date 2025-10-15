"""
Lightweight beat-news feature extractor.

- Pulls RSS feeds for each team (see FEEDS below; you can extend easily).
- Parses titles + summaries in the last N hours.
- Uses rule-based patterns to score signals:
  * qb_out, qb_questionable, star_out, tempo_up, tempo_down, suspension, coach_change
- Merges with optional manual overrides from news_overrides.json at repo root.

Outputs a DataFrame with columns:
  team, season, week, news_qb_out, news_qb_quest, news_star_out,
  news_tempo_up, news_tempo_down, news_suspension, news_coach_change, news_hits
"""

from __future__ import annotations
import re, json, pathlib, datetime as dt
from typing import Dict, List, Any
import feedparser
from bs4 import BeautifulSoup

from .common import cache_requests

# ---- Minimal starter feed map. Extend freely. ----
# Format: team name (CFBD-style) -> list of RSS feed URLs
FEEDS: Dict[str, List[str]] = {
    # Examples; add your teams’ beat feeds here:
    # "Arizona State": [
    #   "https://www.azcentral.com/arizonastateuniversity/asu-football/rss/",
    #   "https://www.houseofsparky.com/rss/index.xml",
    # ],
    # "Texas State": [
    #   "https://www.statesman.com/sports/texas-state/rss/",
    # ],
}

# Simple keywords per signal (lower-cased match)
KW = {
    "qb_out":        [r"\bqb\b.*\bout\b", r"\bstarting quarterback\b.*\bout\b"],
    "qb_quest":      [r"\bqb\b.*\bquestionable\b", r"\bprobable\b", r"\bgame[- ]time decision\b"],
    "star_out":      [r"\b(all-american|star|top|leading|starting)\b.*\bout\b", r"\bwr\b.*\bout\b", r"\brb\b.*\bout\b", r"\blt\b.*\bout\b"],
    "tempo_up":      [r"\bno-huddle\b", r"\bup[- ]tempo\b", r"\bfaster pace\b", r"\bplays per minute up\b"],
    "tempo_down":    [r"\bslow(ing)? down\b", r"\bmilk(ing)? clock\b", r"\b(ball control|ground game)\b"],
    "suspension":    [r"\bsuspend(ed|sion)\b", r"\bineligible\b"],
    "coach_change":  [r"\b(coach|coordinator)\b.*\b(out|fired|resign|hired|new)\b"],
}

def _now_utc():
    return dt.datetime.now(dt.timezone.utc)

def _clean_html(text: str) -> str:
    text = text or ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)

def _score_signals(text: str) -> Dict[str, int]:
    t = text.lower()
    scores = {k: 0 for k in KW.keys()}
    for key, pats in KW.items():
        for p in pats:
            if re.search(p, t):
                scores[key] += 1
    return scores

def _iso_week_from_date(d: dt.datetime):
    iso = d.isocalendar()
    return iso.year, iso.week

def _load_overrides(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _apply_overrides(signals, overrides, season: int, week: int):
    # Overrides structure idea:
    # {
    #   "Arizona State": {
    #     "2025-06": {"qb_out": 1, "tempo_up": 1},
    #     "all": {"tempo_down": 1}
    #   }
    # }
    for team, spec in overrides.items():
        team_rows = signals["team"] == team
        if not team_rows.any():
            continue
        # specific YYYY-WW
        key = f"{season}-{week:02d}"
        if key in spec:
            for k, v in spec[key].items():
                if k in signals.columns:
                    signals.loc[team_rows, k] = v
        # global to all weeks
        if "all" in spec:
            for k, v in spec["all"].items():
                if k in signals.columns:
                    signals.loc[team_rows, k] = v
    return signals

def collect_for_week(teams: List[str], hours_back: int = 168) -> List[Dict[str, Any]]:
    cache_requests()
    cutoff = _now_utc() - dt.timedelta(hours=hours_back)
    season, week = _iso_week_from_date(_now_utc())

    rows = []
    for team in teams:
        total_hits = 0
        agg = {k: 0 for k in KW.keys()}

        for url in FEEDS.get(team, []):
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            for e in feed.entries:
                # published_parsed may be missing; keep it if text matches anyway but prefer recent
                published = None
                if "published_parsed" in e and e.published_parsed:
                    published = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc)
                elif "updated_parsed" in e and e.updated_parsed:
                    published = dt.datetime(*e.updated_parsed[:6], tzinfo=dt.timezone.utc)

                if published and published < cutoff:
                    continue

                title = _clean_html(getattr(e, "title", ""))
                summary = _clean_html(getattr(e, "summary", ""))
                text = f"{title} {summary}".strip()
                if not text:
                    continue

                sc = _score_signals(text)
                if sum(sc.values()) > 0:
                    total_hits += 1
                for k, v in sc.items():
                    agg[k] += v

        row = {
            "team": team,
            "season": season,
            "week": week,
            "news_hits": total_hits,
            "news_qb_out": agg["qb_out"],
            "news_qb_quest": agg["qb_quest"],
            "news_star_out": agg["star_out"],
            "news_tempo_up": agg["tempo_up"],
            "news_tempo_down": agg["tempo_down"],
            "news_suspension": agg["suspension"],
            "news_coach_change": agg["coach_change"],
        }
        rows.append(row)
    return rows

def build_signals_df(all_teams: List[str], hours_back: int, min_conf: float, overrides_path: str = "news_overrides.json"):
    import pandas as pd
    rows = collect_for_week(all_teams, hours_back)
    df = pd.DataFrame(rows)

    # Normalize counts to [0,1] confidence by simple squashing (1 hit -> 0.6, 2 -> 0.8, 3+ -> 1.0)
    def squash(x): 
        return 1.0 if x >= 3 else (0.8 if x == 2 else (0.6 if x == 1 else 0.0))

    for col in ["news_qb_out","news_qb_quest","news_star_out","news_tempo_up","news_tempo_down","news_suspension","news_coach_change"]:
        df[col] = df[col].apply(squash)
        df.loc[df[col] < min_conf, col] = 0.0

    # Apply manual overrides if present
    overrides = _load_overrides(pathlib.Path(overrides_path))
    if not df.empty and overrides:
        # season/week already set to "now" week; overrides will target that.
        season = int(df["season"].iloc[0])
        week = int(df["week"].iloc[0])
        df = _apply_overrides(df, overrides, season, week)

    return df
