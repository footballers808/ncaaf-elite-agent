import argparse, yaml
from .features import build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, required=True)
    ap.add_argument("--windows", type=str, default="3,5,10")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    # Build features for N most recent seasons
    import datetime as dt
    cur = dt.datetime.utcnow().year
    years = list(range(cur - args.years + 1, cur + 1))
    build_features(years, cfg.get("season_type","regular"), cfg)

if __name__ == "__main__":
    main()
