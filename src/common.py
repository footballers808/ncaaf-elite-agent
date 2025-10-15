import os, time, json, math, pathlib, datetime as dt
from typing import Dict, Any, Optional
import requests, requests_cache

ART = pathlib.Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

def cache_requests():
    # Use the same cache location your Actions step restores
    cache_dir = pathlib.Path.home() / ".cache" / "requests_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    requests_cache.install_cache(
        cache_name=str(cache_dir / "http"),
        backend="sqlite",
        expire_after=6*3600,  # 6 hours
    )

def get_env(name: str, default: str=""):
    v = os.environ.get(name, default)
    return v

def dt_utc(d: dt.datetime) -> dt.datetime:
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)

def safe_write_parquet(df, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def safe_read_parquet(path: pathlib.Path):
    import pandas as pd
    return pd.read_parquet(path)
