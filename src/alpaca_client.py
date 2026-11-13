import time, requests
from config import MAX_RETRIES, TIMEOUT_SECONDS

class AlpacaClient:
    def __init__(self, api_key, api_secret, data_url):
        self.base_url = data_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID":api_key,
          "APCA-API-SECRET-KEY":api_secret,"Accept":"application/json"})
    def get(self, path, params):
        for attempt in range(MAX_RETRIES):
            r = self.session.get(self.base_url+path,params=params,timeout=TIMEOUT_SECONDS)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == MAX_RETRIES-1: r.raise_for_status()
                wait = r.headers.get("Retry-After")
                time.sleep(float(wait) if wait else min(2**attempt,30))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Alpaca retries exhausted")
