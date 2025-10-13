import os, yaml, csv
from src.emailer import send_email_html
from src.providers import today_local_iso, cfbd_games_date, cfbd_season_stats, cfbd_team_records
from src.features import build_team_power
from src.model import predict_game
from src.venues_dynamic import build_team_venues_map
from src.weather import enrich_weather_for_games, weather_adjustments
from src.injuries import build_injury_map, injury_adjustments
from src.matchups import matchup_adjustments
from src.odds import fetch_lines_for_games

CFG = yaml.safe_load(open("config.yaml","r"))

def render_email(date_iso, preds):
    head = f"<h3>NCAAF Predicted Scores — {date_iso}</h3>"
    if not preds:
        return head + "<p>No FBS games today.</p>"
    rows = "".join([
        f"<tr>"
        f"<td>{p.get('start_local','')}</td>"
        f"<td><b>{p['away']}</b> @ <b>{p['home']}</b></td>"
        f"<td align='right'>{p['pred_home_pts']}–{p['pred_away_pts']}</td>"
        f"<td align='right'>{p['spread']:+.1f}</td>"
        f"<td align='right'>{p['total']:.1f}</td>"
        f"<td align='right'>{p.get('mkt_spread','')}</td>"
        f"<td align='right'>{p.get('mkt_total','')}</td>"
        f"<td align='right'>{p.get('edge_spread','')}</td>"
        f"<td align='right'>{p.get('edge_total','')}</td>"
        f"<td align='right'>{p['p_home_cover']:.3f}</td>"
        f"<td>{p.get('wx_note','')}</td>"
        f"<td>{p.get('inj_note','')}</td>"
        f"<td>{p.get('mu_note','')}</td>"
        f"</tr>"
        for p in preds
    ])
    table = ("<table cellpadding='6' cellspacing='0' border='1' style='border-collapse:collapse;font-family:Arial,sans-serif;'>"
             "<thead><tr>"
             "<th>Local</th><th>Matchup</th><th>Pred (H–A)</th>"
             "<th>Spread</th><th>Total</th>"
             "<th>Mkt Spr</th><th>Mkt Tot</th>"
             "<th>Edge Spr</th><th>Edge Tot</th>"
             "<th>P(Home Covers)</th>"
             "<th>Weather</th><th>Injuries</th><th>Macro Matchup</th>"
             "</tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    return head + table

def main():
    tz_name = CFG.get("timezone", "America/Phoenix")
    date_iso = today_local_iso(tz_name)
    year = int(date_iso[:4])
    api_key = os.environ.get("CFBD_API_KEY","")

    # Build inputs
    venues_map = build_team_venues_map(year, api_key)
    slate = cfbd_games_date(date_iso, tzname=tz_name)
    stats = cfbd_season_stats(year)
    recs  = cfbd_team_records(year)
    team_tbl = build_team_power(stats, recs)

    # Weather / Injuries
    wx_map   = enrich_weather_for_games(slate, venues_map, local_tz_name=tz_name)
    inj_map  = build_injury_map(slate, year, decay_days=CFG.get("injury_decay_days",28))

    # Market lines (consensus-ish)
    game_ids = [g.get("id") for g in slate if g.get("id") is not None]
    mkt_map  = fetch_lines_for_games(year, game_ids)

    preds = []
    for g in slate:
        # WEATHER
        wx = wx_map.get(g.get("id")) or {}
        spr_wx, tot_wx = weather_adjustments(wx, CFG)
        wx_note = ""
        if wx.get("applied"):
            wx_note = f"{wx.get('stadium','')} — {int(wx['temp_f'])}F, {int(wx['wind_mph'])} mph, {wx['precip_in']:.2f}\""

        # INJURIES
        spr_inj, tot_inj, inj_note = injury_adjustments(g, inj_map, CFG)

        # MACRO MATCHUP
        spr_mu, tot_mu, mu_note = matchup_adjustments(g, team_tbl, CFG)

        # Combined adjustment applied to model
        spr_adj = spr_wx + spr_inj + spr_mu
        tot_adj = tot_wx  + tot_inj  + tot_mu

        p = predict_game(g, team_tbl, CFG, wx_adj=(spr_adj, tot_adj))
        if not p:
            continue

        # Attach notes
        if wx_note:   p["wx_note"]  = wx_note
        if inj_note:  p["inj_note"] = inj_note
        if mu_note:   p["mu_note"]  = mu_note

        # Market + edges
        mkt = mkt_map.get(g.get("id"), {})
        mkt_spread = mkt.get("spread"); mkt_total = mkt.get("total")
        if mkt_spread is not None:
            edge_s = round(p["spread"] - float(mkt_spread), 2)
            p["edge_spread"] = f"{edge_s:+.2f}"
            p["mkt_spread"]  = f"{float(mkt_spread):+.1f}"
            # star if strong edge
            if abs(edge_s) >= CFG.get("edge_threshold_spread", 1.5):
                p["edge_spread"] += " ★"
        if mkt_total is not None:
            edge_t = round(p["total"] - float(mkt_total), 2)
            p["edge_total"] = f"{edge_t:+.2f}"
            p["mkt_total"]  = f"{float(mkt_total):.1f}"
            if abs(edge_t) >= CFG.get("edge_threshold_total", 2.0):
                p["edge_total"] += " ★"

        preds.append(p)

    # Save CSV artifact for your records
    fieldnames = ["gameId","start_local","away","home","pred_home_pts","pred_away_pts",
                  "spread","total","mkt_spread","mkt_total","edge_spread","edge_total",
                  "p_home_cover","wx_note","inj_note","mu_note"]
    try:
        with open("predictions.csv","w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for p in preds:
                w.writerow({k: p.get(k,"") for k in fieldnames})
    except Exception:
        pass

    html = render_email(date_iso, preds)
    send_email_html(f"{CFG['email_subject_prefix']} {date_iso}", html)

if __name__ == "__main__":
    main()
