import argparse, pandas as pd, numpy as np, pathlib, re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def clean_text(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(t or "")).lower().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV with columns: team,text")
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(args.inp)
    except Exception:
        # produce empty parquet so downstream never fails
        pd.DataFrame(columns=["team","news_sentiment","injury_hint","motivation_hint"]).to_parquet(outp, index=False)
        print("news: input missing or unreadable; wrote empty features")
        return

    if df.empty or "team" not in df.columns or "text" not in df.columns:
        pd.DataFrame(columns=["team","news_sentiment","injury_hint","motivation_hint"]).to_parquet(outp, index=False)
        print("news: no usable rows; wrote empty features")
        return

    df = df[["team","text"]].copy()
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)

    sia = SentimentIntensityAnalyzer()
    df["news_sentiment"] = df["text"].map(lambda t: float(sia.polarity_scores(t)["compound"]))
    df["injury_hint"] = df["text"].map(lambda t: int(any(k in t for k in ["ankle","knee","hamstring","concussion","out","questionable","limited"])))
    df["motivation_hint"] = df["text"].map(lambda t: int(any(k in t for k in ["revenge","must win","rival","senior night","bowl eligibility"])))

    # Aggregate by team (in case multiple notes per team)
    agg = df.groupby("team", as_index=False).agg(
        news_sentiment=("news_sentiment","mean"),
        injury_hint=("injury_hint","max"),
        motivation_hint=("motivation_hint","max"),
    )

    agg.to_parquet(outp, index=False)
    print(f"news: wrote {len(agg)} team rows → {outp}")

if __name__ == "__main__":
    main()
