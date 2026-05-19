import json
from datetime import datetime
from core.config import project_path


SCHEMA_VERSION = 2


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_STORE = {
    "schema_version": SCHEMA_VERSION,
    "created_at": "",
    "updated_at": "",
    "sessions": {}
}


def normalize_text(text, max_chars=None):
    text = str(text or "").strip()
    text = text.replace("\r", "")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


def read_store(path):
    full = project_path(path)

    if not full.exists():
        full.parent.mkdir(parents=True, exist_ok=True)
        data = dict(DEFAULT_STORE)
        data["created_at"] = now()
        data["updated_at"] = now()
        full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except Exception:
        data = dict(DEFAULT_STORE)

    data.setdefault("schema_version", 1)
    data.setdefault("sessions", {})

    if data.get("schema_version") != SCHEMA_VERSION:
        data = migrate_store(data)
        write_store(path, data)

    return data


def write_store(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = now()
    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def migrate_store(data):
    """
    v1:
      sessions[session_key] = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]

    v2:
      sessions[session_key] = {
        "rolling_summary": "",
        "last_user_message": "",
        "turns": [
          {"time": "...", "user": "...", "assistant_summary": "..."}
        ]
      }
    """
    sessions = data.get("sessions", {}) or {}
    new_sessions = {}

    for key, value in sessions.items():
        if isinstance(value, dict) and "turns" in value:
            session = value
            session.setdefault("rolling_summary", "")
            session.setdefault("last_user_message", find_last_user_from_turns(session.get("turns", [])))
            session.setdefault("turns", [])
            new_sessions[key] = session
            continue

        if not isinstance(value, list):
            new_sessions[key] = {
                "rolling_summary": "",
                "last_user_message": "",
                "turns": [],
            }
            continue

        turns = []
        pending_user = ""

        for item in value:
            role = item.get("role", "")
            content = normalize_text(item.get("content", ""))

            if role == "user":
                if pending_user:
                    turns.append({
                        "time": item.get("time", now()),
                        "user": pending_user,
                        "assistant_summary": ""
                    })
                pending_user = content

            elif role == "assistant":
                if pending_user:
                    turns.append({
                        "time": item.get("time", now()),
                        "user": pending_user,
                        "assistant_summary": make_summary(content)
                    })
                    pending_user = ""

        if pending_user:
            turns.append({
                "time": now(),
                "user": pending_user,
                "assistant_summary": ""
            })

        new_sessions[key] = {
            "rolling_summary": "",
            "last_user_message": find_last_user_from_turns(turns),
            "turns": turns[-12:],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": data.get("created_at", now()),
        "updated_at": now(),
        "sessions": new_sessions,
    }


def find_last_user_from_turns(turns):
    for turn in reversed(turns or []):
        text = normalize_text(turn.get("user", ""))
        if text:
            return text
    return ""


def build_session_key(message_type, user_id, group_id=None):
    if message_type == "group":
        return f"group_{group_id}_user_{user_id}"
    return f"private_{user_id}"


def make_summary(text, max_chars=96):
    text = normalize_text(text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


class SessionHistoryStore:
    def __init__(self, history_file="memory_runtime/session_history.json", max_turns=6):
        self.history_file = history_file
        self.max_turns = int(max_turns)

    def get_session(self, session_key):
        data = read_store(self.history_file)
        sessions = data.setdefault("sessions", {})

        session = sessions.get(session_key)

        if not isinstance(session, dict):
            session = {
                "rolling_summary": "",
                "last_user_message": "",
                "turns": [],
            }
            sessions[session_key] = session
            write_store(self.history_file, data)

        session.setdefault("rolling_summary", "")
        session.setdefault("last_user_message", find_last_user_from_turns(session.get("turns", [])))
        session.setdefault("turns", [])

        return session

    def save_session(self, session_key, session):
        data = read_store(self.history_file)
        sessions = data.setdefault("sessions", {})
        sessions[session_key] = session
        write_store(self.history_file, data)

    def get_turns(self, session_key):
        session = self.get_session(session_key)
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            return []
        return turns[-self.max_turns:]

    def append_turn(self, session_key, user_text, assistant_summary):
        user_text = normalize_text(user_text, max_chars=800)
        assistant_summary = make_summary(assistant_summary, max_chars=120)

        if not user_text and not assistant_summary:
            return

        session = self.get_session(session_key)
        turns = session.setdefault("turns", [])

        if user_text:
            turns.append({
                "time": now(),
                "user": user_text,
                "assistant_summary": assistant_summary,
            })
            session["last_user_message"] = user_text
        elif turns and assistant_summary:
            turns[-1]["assistant_summary"] = assistant_summary

        # Compact old turns into a simple rolling summary when too many turns accumulate.
        # 这里先用规则摘要，不额外调用模型，避免复杂化。
        keep = self.max_turns
        if len(turns) > keep:
            old = turns[:-keep]
            kept = turns[-keep:]
            compact_lines = []
            existing = normalize_text(session.get("rolling_summary", ""))

            if existing:
                compact_lines.append(existing)

            for t in old[-8:]:
                u = make_summary(t.get("user", ""), max_chars=64)
                a = make_summary(t.get("assistant_summary", ""), max_chars=64)
                if u:
                    compact_lines.append(f"用户曾说：{u}")
                if a:
                    compact_lines.append(f"UGAI曾回复：{a}")

            session["rolling_summary"] = make_summary("\n".join(compact_lines), max_chars=700)
            session["turns"] = kept

        self.save_session(session_key, session)

    def clear_session(self, session_key):
        session = {
            "rolling_summary": "",
            "last_user_message": "",
            "turns": [],
        }
        self.save_session(session_key, session)

    def format_session(self, session_key):
        session = self.get_session(session_key)
        turns = session.get("turns", [])

        if not turns and not session.get("rolling_summary"):
            return "当前会话暂无短期记忆。"

        lines = ["【短期对话记忆】"]

        if session.get("rolling_summary"):
            lines.append("【更早摘要】")
            lines.append(session["rolling_summary"])

        if turns:
            lines.append("【最近几轮】")

        for turn in turns[-self.max_turns:]:
            user = make_summary(turn.get("user", ""), max_chars=160)
            assistant = make_summary(turn.get("assistant_summary", ""), max_chars=120)

            if user:
                lines.append(f"用户：{user}")
            if assistant:
                lines.append(f"UGAI：{assistant}")

        return "\n".join(lines)
