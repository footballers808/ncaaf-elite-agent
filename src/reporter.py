import argparse, pathlib, yaml, pandas as pd
from .common import ART, safe_read_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open("config.yaml","r") as f:
        _ = yaml.safe_load(f)

    P = safe_read_parquet(pathlib.Path(args.preds))
    labels_path = ART / "labels.parquet"
    err_block = ""
    if labels_path.exists():
        L = safe_read_parquet(labels_path)
        M = P.merge(L[["game_id","home_score","away_score"]], on="game_id", how="inner")
        if not M.empty:
            M["ae_home"] = (M["pred_home_score"] - M["home_score"]).abs()
            M["ae_away"] = (M["pred_away_score"] - M["away_score"]).abs()
            err_block = f"\n**Last run MAE** — home: {M['ae_home'].mean():.2f}, away: {M['ae_away'].mean():.2f}\n"

    lines = [
        "# NCAAF Elite Agent\n",
        "- Predictions generated.\n",
        err_block,
        "\nTop edges (vs market) will appear here when used with the emailer.\n"
    ]
    pathlib.Path(args.out).write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
