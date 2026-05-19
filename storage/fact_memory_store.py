import json
import re
from datetime import datetime
from core.config import project_path


DEFAULT_STORE = {
    "schema_version": 1,
    "created_at": "",
    "updated_at": "",
    "next_id": 1,
    "facts": []
}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_tags(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    for sep in ["，", ";", "；"]:
        text = text.replace(sep, ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def clamp_float(value, default=0.5):
    try:
        value = float(value)
    except Exception:
        value = default
    return max(0.0, min(1.0, value))


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

    data.setdefault("schema_version", 1)
    data.setdefault("next_id", 1)
    data.setdefault("facts", [])
    data.setdefault("created_at", now_text())
    data.setdefault("updated_at", now_text())
    return data


def write_store(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_text()
    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def next_id(data):
    value = int(data.get("next_id", 1))
    data["next_id"] = value + 1
    return value


def tokenize(text):
    text = str(text or "").lower()
    words = set(re.findall(r"[a-zA-Z0-9_]+", text))
    chars = set(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    return words | chars


def score_fact(item, query):
    q = tokenize(query)
    if not q:
        return 0.0

    fields = [
        item.get("content", ""),
        item.get("subject", ""),
        " ".join(item.get("tags", []) or []),
        item.get("notes", ""),
        str((item.get("evidence") or {}).get("quote", "")),
    ]

    m = tokenize("\n".join(str(x) for x in fields))
    overlap = len(q & m)
    confidence = clamp_float(item.get("confidence", 0.6), 0.6)
    importance = clamp_float(item.get("importance", 0.5), 0.5)
    return overlap + confidence * 0.25 + importance * 0.25


class FactMemoryStore:
    """
    证据化事实记忆。

    和 reflection_memory 的区别：
    - fact_memory 保存“明确事实/偏好/关系设定”
    - 每条事实尽量带 evidence，方便以后追溯来源
    """
    def __init__(self, fact_file="memory_runtime/fact_memory.json"):
        self.fact_file = fact_file

    def add_fact(
        self,
        content,
        subject="",
        tags=None,
        evidence_event_id=None,
        evidence_quote="",
        confidence=0.75,
        importance=0.6,
        source="manual",
        notes="",
    ):
        content = str(content or "").strip()
        if not content:
            return None

        data = read_store(self.fact_file)

        item = {
            "id": next_id(data),
            "content": content,
            "subject": str(subject or "").strip(),
            "tags": parse_tags(tags),
            "confidence": clamp_float(confidence, 0.75),
            "importance": clamp_float(importance, 0.6),
            "status": "active",
            "source": str(source or "manual"),
            "evidence": {
                "event_id": evidence_event_id,
                "quote": str(evidence_quote or "").strip(),
            },
            "notes": str(notes or "").strip(),
            "created_at": now_text(),
            "updated_at": now_text(),
        }

        data["facts"].append(item)
        write_store(self.fact_file, data)
        return item

    def retire_fact(self, fact_id):
        data = read_store(self.fact_file)
        target = str(fact_id).strip()

        for item in data.get("facts", []):
            if str(item.get("id")) == target:
                item["status"] = "retired"
                item["updated_at"] = now_text()
                write_store(self.fact_file, data)
                return True

        return False

    def get_fact(self, fact_id):
        data = read_store(self.fact_file)
        target = str(fact_id).strip()

        for item in data.get("facts", []):
            if str(item.get("id")) == target:
                return item

        return None

    def list_facts(self, limit=30, include_retired=False):
        data = read_store(self.fact_file)
        items = data.get("facts", [])

        if not include_retired:
            items = [x for x in items if x.get("status", "active") == "active"]

        items = items[-int(limit):]

        if not items:
            return "暂无事实记忆。"

        lines = ["【事实记忆】"]
        for item in items:
            tags = ",".join(item.get("tags", []) or []) or "无"
            evidence = item.get("evidence") or {}
            ev = evidence.get("event_id")
            ev_text = f" event=#{ev}" if ev else ""
            content = item.get("content", "")
            if len(content) > 120:
                content = content[:120] + "..."
            lines.append(
                f"#{item.get('id')} [{item.get('status', 'active')}] "
                f"conf={item.get('confidence')} imp={item.get('importance')} tags={tags}{ev_text}\n"
                f"  {content}"
            )

        return "\n".join(lines)

    def retrieve(self, query, limit=6):
        data = read_store(self.fact_file)
        items = [x for x in data.get("facts", []) if x.get("status", "active") == "active"]

        scored = []
        for item in items:
            scored.append((score_fact(item, query), item))

        scored.sort(key=lambda x: (x[0], x[1].get("importance", 0.5), x[1].get("id", 0)), reverse=True)

        selected = [item for score, item in scored[:int(limit)] if score > 0.2]

        if not selected:
            selected = [item for score, item in scored[:min(3, int(limit))]]

        return selected

    def format_for_prompt(self, items):
        if not items:
            return "暂无事实记忆。"

        lines = []
        for item in items:
            evidence = item.get("evidence") or {}
            ev_id = evidence.get("event_id")
            ev = f"；证据事件#{ev_id}" if ev_id else ""
            subject = item.get("subject") or "未指定对象"
            tags = ",".join(item.get("tags", []) or [])
            tag_text = f"；标签={tags}" if tags else ""
            lines.append(
                f"- [事实#{item.get('id')}] {item.get('content', '')} "
                f"（对象={subject}；可信度={item.get('confidence', 0.75)}{tag_text}{ev}）"
            )

        return "\n".join(lines)
