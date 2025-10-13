import os
import smtplib
from email.message import EmailMessage
from typing import List, Tuple, Optional

def _smtp_settings():
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pwd  = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM", user)
    return host, port, user, pwd, from_addr

def send_email_html(
    subject: str,
    html_body: str,
    to: List[str],
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,  # (filename, content_bytes, mime)
):
    host, port, user, pwd, from_addr = _smtp_settings()
    if not host or not user or not pwd or not from_addr:
        raise RuntimeError("SMTP env vars missing. Set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/SMTP_FROM.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    all_rcpts = to + (cc or []) + (bcc or [])

    msg.set_content("This report requires an HTML-capable email client.")
    msg.add_alternative(html_body, subtype="html")

    if attachments:
        for fname, content, mime in attachments:
            main, sub = mime.split("/", 1) if "/" in mime else ("application", "octet-stream")
            msg.add_attachment(content, maintype=main, subtype=sub, filename=fname)

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(from_addr, all_rcpts, msg.as_string())
