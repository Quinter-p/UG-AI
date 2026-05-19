import json
from datetime import datetime, timedelta
from core.config import project_path


DEFAULT_STORE = {
    "created_at": "",
    "updated_at": "",
    "mode": "normal",
    "cooldown_seconds": 20,
    "last_reply_time_by_group": {},
    "last_reply_time_by_user": {},
    "recent_decisions": []
}


def now():
    return datetime.now()


def now_text():
    return now().strftime("%Y-%m-%d %H:%M:%S")


def parse_time(text):
    try:
        return datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


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

    for key, value in DEFAULT_STORE.items():
        data.setdefault(key, value)

    data.setdefault("last_reply_time_by_group", {})
    data.setdefault("last_reply_time_by_user", {})
    data.setdefault("recent_decisions", [])

    return data


def write_store(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_text()
    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class SpeakRuntimeStore:
    def __init__(self, runtime_file="memory_runtime/speak_runtime.json"):
        self.runtime_file = runtime_file

    def load(self):
        return read_store(self.runtime_file)

    def save(self, data):
        write_store(self.runtime_file, data)

    def get_mode(self):
        data = self.load()
        return str(data.get("mode", "normal") or "normal")

    def set_mode(self, mode):
        mode = str(mode or "").strip().lower()

        if mode not in ["quiet", "normal", "active"]:
            raise ValueError("mode must be quiet / normal / active")

        data = self.load()
        data["mode"] = mode
        self.save(data)
        return mode

    def set_cooldown(self, seconds):
        seconds = int(seconds)
        seconds = max(0, min(3600, seconds))
        data = self.load()
        data["cooldown_seconds"] = seconds
        self.save(data)
        return seconds

    def cooldown_seconds(self):
        data = self.load()
        try:
            return int(data.get("cooldown_seconds", 20))
        except Exception:
            return 20

    def is_group_in_cooldown(self, group_id):
        data = self.load()
        group_id = str(group_id or "")

        last = data.get("last_reply_time_by_group", {}).get(group_id)
        dt = parse_time(last)

        if not dt:
            return False, 0

        cooldown = self.cooldown_seconds()
        elapsed = (now() - dt).total_seconds()
        remain = cooldown - elapsed

        if remain > 0:
            return True, int(remain)

        return False, 0

    def mark_reply(self, group_id="", user_id=""):
        data = self.load()

        if group_id:
            data.setdefault("last_reply_time_by_group", {})[str(group_id)] = now_text()

        if user_id:
            data.setdefault("last_reply_time_by_user", {})[str(user_id)] = now_text()

        self.save(data)

    def add_decision(self, decision):
        data = self.load()
        items = data.setdefault("recent_decisions", [])
        item = dict(decision or {})
        item["time"] = now_text()
        items.append(item)
        data["recent_decisions"] = items[-30:]
        self.save(data)

    def format_status(self):
        data = self.load()
        lines = [
            "【主动发言状态】",
            f"模式：{data.get('mode', 'normal')}",
            f"冷却时间：{data.get('cooldown_seconds', 20)} 秒",
            f"群冷却记录数：{len(data.get('last_reply_time_by_group', {}))}",
            f"最近决策数：{len(data.get('recent_decisions', []))}",
        ]

        recent = data.get("recent_decisions", [])[-5:]

        if recent:
            lines.append("【最近决策】")
            for item in recent:
                lines.append(
                    f"- {item.get('time')} | {item.get('message_type')} | "
                    f"reply={item.get('should_reply')} | {item.get('reason')}"
                )

        return "\n".join(lines)
