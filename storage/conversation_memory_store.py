import json
import re
from datetime import datetime
from core.config import project_path


DEFAULT_STORE = {
    "schema_version": 1,
    "created_at": "",
    "updated_at": "",
    "next_id": 1,
    "memories": []
}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(text, max_chars=500):
    text = str(text or "").strip()
    text = text.replace("\r", "")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > max_chars:
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
    data.setdefault("next_id", 1)
    data.setdefault("memories", [])
    return data


def write_store(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now()
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


def score_memory(memory, query):
    q = tokenize(query)
    if not q:
        return 0

    fields = [
        memory.get("content", ""),
        memory.get("tags", ""),
        memory.get("category", ""),
        memory.get("subject", ""),
    ]
    m = tokenize(" ".join(str(x) for x in fields))
    return len(q & m)


def format_memory_item(memory):
    tags = memory.get("tags", "")
    category = memory.get("category", "general")
    subject = memory.get("subject", "")
    content = memory.get("content", "")

    prefix = f"[{category}]"
    if subject:
        prefix += f"[{subject}]"
    if tags:
        prefix += f"[{tags}]"
    return f"{prefix} {content}"


class ConversationMemoryStore:
    """
    长期个人记忆。

    只保存主人确认过的稳定信息，例如：
    - 主人的偏好
    - Agent 的长期互动经验
    - 主人对回答风格的要求
    - 稳定关系信息

    不保存临时事件，例如今天吃了什么、刚才下雨等。
    """

    def __init__(self, memory_file="memory_runtime/conversation_memory.json"):
        self.memory_file = memory_file

    def add_memory(
        self,
        content,
        category="personal",
        subject="",
        tags="",
        source_user_id="",
        source_name="",
    ):
        content = normalize_text(content)

        if not content:
            return None

        data = read_store(self.memory_file)

        item = {
            "id": next_id(data),
            "created_at": now(),
            "updated_at": now(),
            "category": str(category or "personal"),
            "subject": str(subject or ""),
            "tags": str(tags or ""),
            "content": content,
            "source_user_id": str(source_user_id or ""),
            "source_name": str(source_name or ""),
        }

        data["memories"].append(item)
        write_store(self.memory_file, data)
        return item

    def delete_memory(self, memory_id):
        data = read_store(self.memory_file)
        memory_id = int(memory_id)

        before = len(data.get("memories", []))
        data["memories"] = [
            item for item in data.get("memories", [])
            if int(item.get("id", -1)) != memory_id
        ]
        after = len(data["memories"])
        write_store(self.memory_file, data)

        return before != after

    def list_memories(self, limit=30):
        data = read_store(self.memory_file)
        items = data.get("memories", [])[-int(limit):]

        if not items:
            return "暂无长期个人记忆。"

        lines = ["【长期个人记忆】"]

        for item in items:
            lines.append(f"#{item.get('id')} {format_memory_item(item)}")

        return "\n".join(lines)

    def retrieve(self, query, limit=8):
        data = read_store(self.memory_file)
        memories = data.get("memories", [])

        scored = []
        for item in memories:
            scored.append((score_memory(item, query), item))

        scored.sort(key=lambda x: (x[0], x[1].get("id", 0)), reverse=True)

        selected = [item for score, item in scored[:int(limit)] if score > 0]

        if not selected:
            selected = [item for score, item in scored[:min(3, int(limit))]]

        return selected

    def format_for_prompt(self, memories):
        if not memories:
            return "暂无长期个人记忆。"

        lines = []
        for item in memories:
            lines.append("- " + format_memory_item(item))

        return "\n".join(lines)
