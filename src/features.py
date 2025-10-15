def build_features(years: List[int], season_type: str, cfg: Dict[str, Any]):
    import pandas as pd
    frames = []
    for y in years:
        base = _extract_game_rows(y, season_type)
        if base.empty:
            continue

        pace = _pace_from_stats(y, season_type)
        mkt  = _merge_market(y, season_type, week=None)
        inj  = _injury_counts(y, week=None, window_days=cfg["injury_window_days"])

        df = base.merge(pace, on=["game_id","team"], how="left") \
                 .merge(mkt, on="game_id", how="left") \
                 .merge(inj, on="team", how="left")

        # Rolling recent form per team by season
        df = df.sort_values(["team","season","week"])
        df["pf_mean"]   = df.groupby(["team","season"])["points_for"].apply(lambda s: _roll_mean(s.shift(1), cfg["form_games"]))
        df["pa_mean"]   = df.groupby(["team","season"])["points_against"].apply(lambda s: _roll_mean(s.shift(1), cfg["form_games"]))
        df["pace_mean"] = df.groupby(["team","season"])["pace"].apply(lambda s: _roll_mean(s.shift(1), cfg["pace_games"]))

        # ---- Beat-writer news merge (optional) ----
        if cfg.get("news_enabled", True):
            from . import news as newsmod
            all_teams = sorted(pd.unique(df["team"].dropna()))
            news_df = newsmod.build_signals_df(
                all_teams=all_teams,
                hours_back=int(cfg.get("news_hours_back", 168)),
                min_conf=float(cfg.get("news_min_confidence", 0.5)),
                overrides_path="news_overrides.json",
            )
            if not news_df.empty:
                df = df.merge(
                    news_df,
                    on=["team","season","week"],
                    how="left",
                )

                # Fill NaNs for news signals with zeros
                for col in ["news_hits","news_qb_out","news_qb_quest","news_star_out","news_tempo_up","news_tempo_down","news_suspension","news_coach_change"]:
                    if col in df.columns:
                        df[col] = df[col].fillna(0.0)

        frames.append(df)

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = ART / "features" / "penalties.parquet"
    safe_write_parquet(full, path)
    return path
