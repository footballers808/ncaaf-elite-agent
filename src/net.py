# src/net.py
import hashlib, json, os, pathlib, time, random
from typing import Dict, Any, Optional
import requests

CFBD = "https://api.collegefootballdata.com"
CACHE_DIR = pathlib.Path(".cache/cfbd")
CACHE_TTL_SECS = 7 * 24 * 3600  # 7 days

def _headers() -> Dict[str,str]:
    key = os.environ.get("CFBD_API_KEY","")
    return {"Authorization": f"Bearer {key}"} if key else {}

def _key(url: str, params: Optional[Dict[str, Any]]) -> pathlib.Path:
    q = "&".join(f"{k}={params[k]}" for k in sorted(params or {}))
    base = f"{url}?{q}" if q else url
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"

def cfbd_get(path: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 5) -> Any:
    """
    Cached+retry GET for CFBD.
    path may be '/teams/fbs' or a full 'https://api.collegefootballdata.com/teams/fbs'
    """
    url = path if path.startswith("http") else f"{CFBD}{path}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _key(url, params)

    # serve cached if fresh
    if fp.exists() and (time.time() - fp.stat().st_mtime) < CACHE_TTL_SECS:
        return json.loads(fp.read_text(encoding="utf-8"))

    attempt = 0
    while True:
        r = requests.get(url, params=params or {}, headers=_headers(), timeout=30)
        if r.status_code == 200:
            data = r.json()
            fp.write_text(json.dumps(data), encoding="utf-8")
            return data
        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            attempt += 1
            sleep_s = min(30, 2 ** attempt) + random.uniform(0.0, 0.5)
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
