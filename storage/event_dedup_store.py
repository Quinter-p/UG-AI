import json
from datetime import datetime
from core.config import project_path


DEFAULT_STORE = {
    "created_at": "",
    "updated_at": "",
    "seen_message_keys": [],
    "recent_traces": [],
    "max_seen": 1000,
    "max_recent_traces": 80
}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_store(path):
    full = project_path(path)

    if not full.exists():
        full.parent.mkdir(parents=True, exist_ok=True)
        data = dict(DEFAULT_STORE)
        data["created_at"] = now_text()
        data["updated_at"] = now_text()
        full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except Exception:
        data = dict(DEFAULT_STORE)

    for k, v in DEFAULT_STORE.items():
        data.setdefault(k, v)

    if not isinstance(data.get("seen_message_keys"), list):
        data["seen_message_keys"] = []
    if not isinstance(data.get("recent_traces"), list):
        data["recent_traces"] = []

    return data


def write_store(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_text()
    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class EventDedupStore:
    """
    OneBot / QQ 事件去重与 trace 运行时状态。

    只存很小的运行时索引，不存对话内容。
    event_log.jsonl 负责真正事件记录。
    """
    def __init__(self, runtime_file="memory_runtime/event_runtime.json"):
        self.runtime_file = runtime_file

    def is_seen(self, message_key):
        data = read_store(self.runtime_file)
        return str(message_key) in set(str(x) for x in data.get("seen_message_keys", []))

    def mark_seen(self, message_key):
        data = read_store(self.runtime_file)
        key = str(message_key)
        seen = [str(x) for x in data.get("seen_message_keys", [])]

        if key not in seen:
            seen.append(key)

        max_seen = int(data.get("max_seen", 1000))
        data["seen_message_keys"] = seen[-max_seen:]
        write_store(self.runtime_file, data)

    def add_trace(self, trace):
        data = read_store(self.runtime_file)
        traces = data.get("recent_traces", [])
        item = dict(trace or {})
        item.setdefault("time", now_text())
        traces.append(item)
        max_recent = int(data.get("max_recent_traces", 80))
        data["recent_traces"] = traces[-max_recent:]
        write_store(self.runtime_file, data)

    def format_status(self, limit=10):
        data = read_store(self.runtime_file)
        lines = [
            "【事件运行状态】",
            f"seen_message_keys：{len(data.get('seen_message_keys', []))}",
            f"recent_traces：{len(data.get('recent_traces', []))}",
            f"updated_at：{data.get('updated_at', '')}",
        ]

        recent = data.get("recent_traces", [])[-int(limit):]
        if recent:
            lines.append("【最近 trace】")
            for t in recent:
                lines.append(
                    f"- {t.get('time')} trace={t.get('trace_id')} "
                    f"dup={t.get('duplicate')} type={t.get('message_type')} "
                    f"user={t.get('user_id')} key={t.get('message_key')}"
                )

        return "\n".join(lines)
