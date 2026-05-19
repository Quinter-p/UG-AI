import json
import re
from datetime import datetime
from core.config import project_path


DEFAULT_STORE = {
    "schema_version": 1,
    "created_at": "",
    "updated_at": "",
    "next_id": 1,
    "tasks": []
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
    data.setdefault("tasks", [])
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


def score_task(item, query):
    q = tokenize(query)
    if not q:
        return 0.0

    fields = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("owner", ""),
        item.get("source_text", ""),
        " ".join(item.get("tags", []) or []),
    ]
    m = tokenize("\n".join(str(x) for x in fields))
    overlap = len(q & m)
    priority_bonus = {"high": 0.8, "normal": 0.4, "low": 0.1}.get(item.get("priority", "normal"), 0.2)
    status_bonus = 0.4 if item.get("status") in ["pending", "active"] else 0.0
    return overlap + priority_bonus + status_bonus


class TaskStore:
    """
    v8.0 任务登记表。

    这还不是自动工具执行，只是 Agent 的任务底座：
    - 记录主人交代的任务
    - 跟踪 pending/done/cancelled
    - 后续可以接 tool bus / executor
    """
    def __init__(self, task_file="memory_runtime/task_registry.json"):
        self.task_file = task_file

    def add_task(
        self,
        title,
        description="",
        owner="",
        priority="normal",
        tags=None,
        source_text="",
        source_event_id=None,
        trace_id="",
        created_by="",
    ):
        title = str(title or "").strip()
        if not title:
            return None

        priority = str(priority or "normal").strip().lower()
        if priority not in ["low", "normal", "high"]:
            priority = "normal"

        data = read_store(self.task_file)
        item = {
            "id": next_id(data),
            "title": title,
            "description": str(description or "").strip(),
            "status": "pending",
            "priority": priority,
            "owner": str(owner or "").strip(),
            "tags": parse_tags(tags),
            "source_text": str(source_text or "").strip(),
            "source_event_id": source_event_id,
            "trace_id": str(trace_id or "").strip(),
            "created_by": str(created_by or "").strip(),
            "created_at": now_text(),
            "updated_at": now_text(),
            "completed_at": "",
            "cancelled_at": "",
            "notes": "",
        }
        data["tasks"].append(item)
        write_store(self.task_file, data)
        return item

    def update_status(self, task_id, status, notes=""):
        status = str(status or "").strip().lower()
        if status not in ["pending", "active", "done", "cancelled"]:
            raise ValueError("status must be pending/active/done/cancelled")

        data = read_store(self.task_file)
        target = str(task_id).strip()

        for item in data.get("tasks", []):
            if str(item.get("id")) == target:
                item["status"] = status
                item["updated_at"] = now_text()
                if notes:
                    item["notes"] = str(notes)
                if status == "done":
                    item["completed_at"] = now_text()
                if status == "cancelled":
                    item["cancelled_at"] = now_text()
                write_store(self.task_file, data)
                return item

        return None

    def get(self, task_id):
        data = read_store(self.task_file)
        target = str(task_id).strip()
        for item in data.get("tasks", []):
            if str(item.get("id")) == target:
                return item
        return None

    def list_tasks(self, status="pending", limit=30):
        data = read_store(self.task_file)
        items = data.get("tasks", [])

        status = str(status or "pending").strip().lower()
        if status != "all":
            items = [x for x in items if str(x.get("status", "pending")).lower() == status]

        items = items[-int(limit):]

        if not items:
            return f"暂无 {status} 任务。"

        title = "【任务列表】" if status == "all" else f"【{status} 任务】"
        lines = [title]
        for item in items:
            desc = item.get("description", "")
            if len(desc) > 80:
                desc = desc[:80] + "..."
            tags = ",".join(item.get("tags", []) or []) or "无"
            lines.append(
                f"#{item.get('id')} [{item.get('status')}/{item.get('priority')}] "
                f"{item.get('title')} tags={tags}\n"
                f"  {desc}"
            )

        return "\n".join(lines)

    def retrieve(self, query, limit=6):
        data = read_store(self.task_file)
        items = [
            x for x in data.get("tasks", [])
            if str(x.get("status", "pending")).lower() in ["pending", "active"]
        ]

        scored = []
        for item in items:
            scored.append((score_task(item, query), item))

        scored.sort(key=lambda x: (x[0], x[1].get("priority", ""), x[1].get("id", 0)), reverse=True)

        selected = [item for score, item in scored[:int(limit)] if score > 0.2]

        if not selected:
            selected = [item for score, item in scored[:min(3, int(limit))]]

        return selected

    def format_for_prompt(self, items):
        if not items:
            return "暂无待办任务。"

        lines = []
        for item in items:
            tags = ",".join(item.get("tags", []) or [])
            tag_text = f"；标签={tags}" if tags else ""
            lines.append(
                f"- [任务#{item.get('id')}] {item.get('title')} "
                f"（状态={item.get('status')}；优先级={item.get('priority')}{tag_text}）"
            )
            desc = str(item.get("description", "")).strip()
            if desc:
                lines.append(f"  说明：{desc}")

        return "\n".join(lines)
