#!/usr/bin/env python3
import sys, yaml, pathlib, shutil, difflib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURR = ROOT / "config.yaml"
BEST = ROOT / "tuning-results" / "best_config.yaml"
MERGED = ROOT / "config.merged.yaml"

# keys you DO NOT overwrite from best_config (secrets, paths, org-specific switches)
PROTECT = {
    "email": ["smtp_host", "smtp_port", "from_addr", "to_addrs"],
    "paths": ["data_dir", "models_dir", "artifacts_dir"],
    "cfbd": ["api_key"],
    "report": ["send_email", "send_slack", "slack_webhook"],
}

def load(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def protect_overrides(dst, src):
    # copy src into dst, but keep protected keys from dst
    out = yaml.safe_load(yaml.dump(src))  # deep copy
    for top, subkeys in PROTECT.items():
        if top in dst:
            out.setdefault(top, {})
            for k in subkeys:
                if k in dst.get(top, {}):
                    out[top][k] = dst[top][k]
    return out

def show_diff(a_text, b_text, a_label="config.yaml", b_label="config.merged.yaml"):
    for line in difflib.unified_diff(
        a_text.splitlines(True), b_text.splitlines(True),
        fromfile=a_label, tofile=b_label
    ):
        sys.stdout.write(line)

def main():
    if not BEST.exists():
        print(f"❌ {BEST} not found. Run tune first.")
        sys.exit(1)
    curr = load(CURR)
    best = load(BEST)

    merged = protect_overrides(curr, best)
    merged_text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    curr_text = yaml.safe_dump(curr, sort_keys=False, allow_unicode=True)

    MERGED.write_text(merged_text, encoding="utf-8")
    print("🔎 Proposed diff (current → merged):\n")
    show_diff(curr_text, merged_text)

    print(f"\n✅ Wrote {MERGED}. If it looks good, run:\n")
    print(f"   cp {MERGED} {CURR}")

if __name__ == "__main__":
    main()
