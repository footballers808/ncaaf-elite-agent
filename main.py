import os, yaml, datetime as dt
from dateutil import tz
from src.emailer import send_email_html

CFG = yaml.safe_load(open("config.yaml","r"))
TZ = CFG.get("timezone","America/Phoenix")

def now_local_iso(tzname="America/Phoenix"):
    return dt.datetime.now(tz.gettz(tzname)).strftime("%Y-%m-%d %H:%M:%S")

def main():
    subject = f"{CFG.get('email_subject_prefix','[NCAAF Agent]')} {now_local_iso(TZ)}"
    html = (
        "<h3>NCAAF Agent — Health Check</h3>"
        "<p>If you received this, SMTP delivery from your own email is working.</p>"
    )
    send_email_html(subject, html)

if __name__ == "__main__":
    main()
