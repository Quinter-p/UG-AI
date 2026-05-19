import requests
class OneBotClient:
    def __init__(self, config):
        cfg = config.get("onebot", {})
        self.api_base = str(cfg.get("api_base", "http://127.0.0.1:3000")).rstrip("/")
        self.access_token = str(cfg.get("access_token", "") or "")
    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.access_token: h["Authorization"] = f"Bearer {self.access_token}"
        return h
    def send_private_msg(self, user_id, message):
        return self.post("/send_private_msg", {"user_id": int(user_id), "message": str(message)})
    def send_group_msg(self, group_id, message):
        return self.post("/send_group_msg", {"group_id": int(group_id), "message": str(message)})
    def post(self, path, payload):
        r = requests.post(self.api_base + path, json=payload, headers=self.headers(), timeout=30)
        r.raise_for_status()
        return r.json()
