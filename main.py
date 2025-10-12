import yaml
from src.emailer import send_email_html
from src.providers import today_local_iso, cfbd_games_date, cfbd_season_stats, cfbd_team_records
from src.features import build_team_power
from src.model import predict_game

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
        f"<td align='right'>{p['p_home_cover']:.3f}</td></tr>"
        for p in preds
    ])
    table = ("<table cellpadding='6' cellspacing='0' border='1' style='border-collapse:collapse;font-family:Arial,sans-serif;'>"
             "<thead><tr><th>Local</th><th>Matchup</th><th>Pred (H–A)</th>"
             "<th>Spread</th><th>Total</th><th>P(Home Covers)</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    return head + table

def main():
    date_iso = today_local_iso(CFG["timezone"])
    slate = cfbd_games_date(date_iso, tzname=CFG["timezone"])
    stats = cfbd_season_stats(int(date_iso[:4]))
    recs  = cfbd_team_records(int(date_iso[:4]))
    team_tbl = build_team_power(stats, recs)

    preds = []
    for g in slate:
        p = predict_game(g, team_tbl, CFG)
        if p: preds.append(p)

    html = render_email(date_iso, preds)
    send_email_html(f"{CFG['email_subject_prefix']} {date_iso}", html)

if __name__ == "__main__":
    main()
