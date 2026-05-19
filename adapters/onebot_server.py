import gzip, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from adapters.onebot_client import OneBotClient
from core.config import load_config
from core.graph import AgentCore
from core.logger import log

class BotRuntime:
    def __init__(self):
        self.config = load_config(); self.onebot_cfg = self.config.get("onebot", {})
        self.client = OneBotClient(self.config); self.agent = AgentCore(); self.seen_message_ids = set(); self.lock = threading.Lock()
    def should_ignore_duplicate(self, message_id):
        if message_id is None: return False
        with self.lock:
            if message_id in self.seen_message_ids: return True
            self.seen_message_ids.add(message_id)
            if len(self.seen_message_ids) > 3000: self.seen_message_ids = set(list(self.seen_message_ids)[-1000:])
        return False
    def handle_event_async(self, event):
        threading.Thread(target=self.handle_event, args=(event,), daemon=True).start()
    def handle_event(self, event):
        try:
            if event.get("post_type") != "message": return
            message_type = event.get("message_type")
            if message_type not in ["private", "group"]: return
            message_id = event.get("message_id")
            if self.should_ignore_duplicate(message_id): return
            user_id = str(event.get("user_id") or ""); self_id = str(event.get("self_id") or event.get("_self_id") or "")
            if self.onebot_cfg.get("ignore_self_message", True) and self_id and user_id == self_id: return
            raw = event.get("raw_message") or event.get("message") or ""
            log("QQ_USER", f"{message_type}_{user_id}: {raw}")
            result = self.agent.handle_event(event)
            reply = str(result.get("final_reply") or "").strip(); route = result.get("route", "")
            log("AGENT_ROUTE", f"route={route}\nreply={reply}")
            if not reply: return
            if message_type == "group": self.client.send_group_msg(event.get("group_id"), reply)
            else: self.client.send_private_msg(user_id, reply)
            log("QQ_SEND", f"{message_type}: {reply}")
        except Exception as e:
            log("BOT_ERROR", f"{type(e).__name__}: {e}")

class OneBotHandler(BaseHTTPRequestHandler):
    runtime = None
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"UGAI Agent LangGraph v0 is running")
    def do_POST(self):
        try:
            body = self.read_body(); text = self.decode_body(body)
            if not text.strip(): self.send_response(204); self.end_headers(); return
            event = json.loads(text); self_id = self.headers.get("x-self-id") or self.headers.get("X-Self-ID") or ""
            if self_id: event["_self_id"] = self_id
            OneBotHandler.runtime.handle_event_async(event)
            self.send_response(204); self.end_headers()
        except Exception as e:
            log("ONEBOT_ERROR", f"{type(e).__name__}: {e}"); self.send_response(204); self.end_headers()
    def read_body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower(): return self.read_chunked_body()
        length = int(self.headers.get("Content-Length", "0") or "0")
        return b"" if length <= 0 else self.rfile.read(length)
    def read_chunked_body(self):
        chunks = []
        while True:
            size_line = self.rfile.readline()
            if not size_line: break
            size_line = size_line.strip()
            if not size_line: continue
            size = int(size_line.split(b";", 1)[0], 16)
            if size == 0:
                while True:
                    trailer = self.rfile.readline()
                    if trailer in (b"\r\n", b"\n", b""): break
                break
            chunks.append(self.rfile.read(size)); self.rfile.read(2)
        return b"".join(chunks)
    def decode_body(self, body):
        if (self.headers.get("Content-Encoding") or "").lower() == "gzip": body = gzip.decompress(body)
        return body.decode("utf-8", errors="replace")
    def log_message(self, format, *args): return

def run_server():
    config = load_config(); cfg = config.get("onebot", {}); host = cfg.get("listen_host", "127.0.0.1"); port = int(cfg.get("listen_port", 8765))
    runtime = BotRuntime(); OneBotHandler.runtime = runtime
    server = ThreadingHTTPServer((host, port), OneBotHandler)
    log("START", f"UGAI Agent LangGraph v0 started at http://{host}:{port}/onebot")
    try: server.serve_forever()
    except KeyboardInterrupt: log("STOP", "UGAI Agent stopped")
    finally: server.server_close()
