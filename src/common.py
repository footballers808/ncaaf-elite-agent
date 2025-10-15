from __future__ import annotations
import os, time, typing as T
import pandas as pd

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: str):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def iso_year_week(iso_week: str | None) -> tuple[int,int]:
    # iso_week: "YYYY-WW" or "auto"
    if not iso_week or iso_week == "auto":
        import datetime as dt
        d = dt.date.today()
        iso = d.isocalendar()
        return iso.year, iso.week
    y, w = iso_week.split("-")
    return int(y), int(w)
