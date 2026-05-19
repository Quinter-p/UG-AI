import json
from datetime import datetime
from core.config import project_path


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class EventLogStore:
    """
    事件日志：JSONL。

    v7.0/v7.1 原则：
    - 记录 AI 实际发出过的回复
    - 不记录被 ignore 的群聊消息，避免刷爆日志
    - 每行一个 JSON，方便以后做 evidence / replay / reflection
    """
    def __init__(self, event_file="memory_runtime/event_log.jsonl"):
        self.event_file = event_file

    def path(self):
        full = project_path(self.event_file)
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def count(self):
        full = self.path()
        if not full.exists():
            return 0

        n = 0
        with full.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def append(self, event):
        full = self.path()
        event = dict(event or {})
        event.setdefault("time", now_text())
        event.setdefault("id", self.count() + 1)

        with full.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event

    def iter_events(self):
        full = self.path()
        if not full.exists():
            return

        with full.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def get_by_id(self, event_id):
        target = str(event_id).strip()
        for item in self.iter_events() or []:
            if str(item.get("id")) == target:
                return item
        return None

    def read_recent(self, limit=30):
        full = self.path()
        if not full.exists():
            return []

        limit = max(1, int(limit))
        lines = []

        with full.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    lines.append(line)

        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

        return rows

    def format_recent(self, limit=10):
        rows = self.read_recent(limit=limit)

        if not rows:
            return "暂无事件日志。"

        lines = ["【最近事件日志】"]
        for e in rows:
            speaker = e.get("speaker_name") or e.get("user_id") or "unknown"
            text = str(e.get("user_text", "")).replace("\n", " ")
            reply = str(e.get("assistant_reply", "")).replace("\n", " ")

            if len(text) > 60:
                text = text[:60] + "..."
            if len(reply) > 90:
                reply = reply[:90] + "..."

            lines.append(
                f"#{e.get('id')} {e.get('time')} [{e.get('message_type')}] "
                f"{speaker}: {text} -> {reply}"
            )

        return "\n".join(lines)

    def compact_for_reflection(self, limit=30):
        rows = self.read_recent(limit=limit)

        if not rows:
            return "暂无事件。"

        blocks = []
        for e in rows:
            speaker = e.get("speaker_name") or e.get("user_id") or "unknown"
            rel = e.get("relationship_role", "")
            mood = e.get("mood", "")
            speak_reason = e.get("speak_reason", "")
            blocks.append(
                f"- #{e.get('id')} {e.get('time')} [{e.get('message_type')}] "
                f"speaker={speaker}/{rel}, mood={mood}, speak={speak_reason}\n"
                f"  user: {e.get('user_text', '')}\n"
                f"  ai: {e.get('assistant_reply', '')}"
            )

        return "\n".join(blocks)
