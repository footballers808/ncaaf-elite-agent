# src/requests_cached.py
from typing import Any, Dict, Optional
import threading
import requests
from src.net import cfbd_get

_real_get = requests.get
_guard = threading.local()

def _proxy_get(url: str, params: Optional[Dict[str, Any]] = None, *a, **kw):
    # Only intercept CFBD URLs, and never re-enter if already inside our shim
    if "api.collegefootballdata.com" in url and not getattr(_guard, "in_cfbd", False):
        try:
            _guard.in_cfbd = True
            return _ResponseShim(cfbd_get(url, params))
        finally:
            _guard.in_cfbd = False
    # For everything else (or re-entrant calls), use the real requests.get
    return _real_get(url, params=params, *a, **kw)

class _ResponseShim:
    def __init__(self, data: Any):
        self._data = data
        self.status_code = 200
        self.ok = True
        self.text = ""
    def json(self) -> Any:
        return self._data
    def raise_for_status(self):
        return None

def install_requests_cache():
    requests.get = _proxy_get  # type: ignore[attr-defined]
