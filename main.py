import os, yaml
from src.emailer import send_email_html
from src.providers import today_local_iso, cfbd_games_date, cfbd_season_stats, cfbd_team_records
from src.features import build_team_power
from src.model import predict_game
from src.venues_dynamic import build_team_venues_map
from src.weather import enrich_weather_for_games, weather_adjustments

CFG = yaml.safe_load(open("config.yaml","r"))

def render_email(date_iso, preds):
    head = f"<h3>NCAAF Predicted Scores — {date_iso}</h3>"
    if not preds:
        return head + "<p>No FBS games today.</p>"
    rows = "".join([
        f"<tr><td>{p.get('start_local','')}</td>"
        f"<td><b>{p['away']}</b> @ <b>{p['home']}</b></td>"
        f"<td align='right'>{p['pred_home_pts']}–{p['pred_away_pts']}</td>"
        f"<td align='right'>{p['spread']:+.1f}</td>"
        f"<td align='right'>{p['total']:.1f}</td>"
        f"<td align='right'>{p['p_home_cover']:.3f}</td>"
        f"<td>{p.get('wx_note','')}</td></tr>"
        for p in preds
    ])
    table = ("<table cellpadding='6' cellspacing='0' border='1' style='border-collapse:collapse;font-family:Arial,sans-serif;'>"
             "<thead><tr><th>Local</th><th>Matchup</th><th>Pred (H–A)</th>"
             "<th>Spread</th><th>Total</th><th>P(Home Covers)</th><th>Weather</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    return head + table

def main():
    date_iso = today_local_iso(CFG["timezone"])
    year = int(date_iso[:4])

    # Build a complete venues map for ALL FBS teams this season
    api_key = os.environ.get("CFBD_API_KEY","")
    venues_map = build_team_venues_map(year, api_key)

    # Slate + team power
    slate = cfbd_games_date(date_iso, tzname=CFG["timezone"])
    stats = cfbd_season_stats(year)
    recs  = cfbd_team_records(year)
    team_tbl = build_team_power(stats, recs)

    # Weather for each game, using dynamic venues
    wx_map = enrich_weather_for_games(slate, venues_map, local_tz_name=CFG["timezone"])

    preds = []
    for g in slate:
        wx = wx_map.get(g.get("id")) or {}
        spr_adj, tot_adj = weather_adjustments(wx, CFG)
        p = predict_game(g, team_tbl, CFG, wx_adj=(spr_adj, tot_adj))
        if p:
            if wx.get("applied"):
                p["wx_note"] = f"{wx.get('stadium','')} — {int(wx['temp_f'])}F, {int(wx['wind_mph'])} mph, {wx['precip_in']:.2f}\""
            preds.append(p)

    html = render_email(date_iso, preds)
    send_email_html(f"{CFG['email_subject_prefix']} {date_iso}", html)

if __name__ == "__main__":
    main()

