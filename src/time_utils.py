from datetime import datetime, timezone

def parse_utc(value):
    if value is None: return None
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None: raise ValueError("Timestamp must include timezone information.")
    return dt.astimezone(timezone.utc)

def utc_iso(value):
    dt = parse_utc(value)
    return dt.isoformat().replace("+00:00", "Z") if dt else None

def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
