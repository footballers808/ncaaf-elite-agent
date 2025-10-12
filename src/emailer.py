import os, smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_html(subject, html, to_addrs=None, from_addr=None):
    user = os.environ["EMAIL_USER"]
    pwd  = os.environ["EMAIL_PASS"]
    to_addrs = to_addrs or os.environ.get("EMAIL_TO","").split(",")
    to_addrs = [a.strip() for a in to_addrs if a.strip()]
    from_addr = from_addr or user

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    server = os.environ.get("SMTP_SERVER", "smtp.office365.com")
    port = int(os.environ.get("SMTP_PORT", "587"))

    with smtplib.SMTP(server, port) as s:
        s.starttls(context=context)
        s.login(user, pwd)
        s.sendmail(from_addr, to_addrs, msg.as_string())
