# src/main.py
"""
CI entrypoint that delegates to the repository's root-level main.py.

Why: Your pipeline orchestration already lives in main.py at the repo root.
This wrapper lets GitHub Actions run `python -m src.main` reliably without
depending on specific function names in src/model.py or src/labeler.py.
"""

from __future__ import annotations

import importlib
import sys

def _delegate_to_root_main() -> int:
    try:
        # Import the repo's root-level main.py as module "main"
        root_main = importlib.import_module("main")
    except Exception as e:
        print(f"⚠️ Could not import root-level main.py as module 'main': {e}")
        print("Make sure a file named `main.py` exists at the repository root.")
        return 1

    # Try common entrypoints in order
    for fn_name in ("run", "main", "start", "pipeline"):
        fn = getattr(root_main, fn_name, None)
        if callable(fn):
            print(f"↪ Delegating to root main.{fn_name}()")
            ret = fn()
            try:
                return int(ret) if ret is not None else 0
            except Exception:
                return 0

    print("⚠️ root main.py has no callable run()/main()/start()/pipeline(). "
          "Please add a simple `def run(): ...` or `def main(): ...` in main.py.")
    return 1


def run() -> int:
    return _delegate_to_root_main()


if __name__ == "__main__":
    sys.exit(run())
