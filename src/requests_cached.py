# src/requests_cached.py
from typing import Any, Dict, Optional
import requests
from src.net import cfbd_get

_real_get = requests.get

def _proxy_get(url: str, params: Optional[Dict[str, Any]] = None, *a, **kw):
    # Only intercept CFBD; everything else uses real requests.get
    if "api.collegefootballdata.com" in url:
        return _ResponseShim(cfbd_get(url, params))
    return _real_get(url, params=params, *a, **kw)

class _ResponseShim:
    def __init__(self, data: Any):
        self._data = data
        self.status_code = 200
        self.ok = True
        self.text = ""  # unused
    def json(self) -> Any:
        return self._data
    def raise_for_status(self):
        return None

def install_requests_cache():
    requests.get = _proxy_get  # type: ignore[attr-defined]
