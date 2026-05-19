import json
import shutil
from datetime import datetime
from core.config import project_path

DEFAULT_STORE = {
    "created_at": "",
    "updated_at": "",
    "users": {}
}

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def clamp(value, low=0, high=100):
    try:
        value = int(value)
    except Exception:
        value = low
    return max(low, min(high, value))

def parse_list(value):
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

def backup_file(path, reason="write"):
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}_backup_{reason}_{stamp()}{path.suffix}.bak")
    shutil.copy2(path, backup)
    return backup

def empty_store():
    data = dict(DEFAULT_STORE)
    data["created_at"] = now()
    data["updated_at"] = now()
    data["users"] = {}
    return data

def read_store(path):
    """
    安全读取 relationship store。

    v4.1.2 原则：
    - 读取失败不自动覆盖原文件
    - JSON 格式错误时直接抛错，避免把手写文件覆盖成默认关系
    - 文件不存在时返回空 store，但不创建文件；创建只能发生在明确写操作
    """
    full = project_path(path)

    if not full.exists():
        return empty_store()

    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except Exception as e:
        backup = backup_file(full, reason="invalid_json")
        raise RuntimeError(
            f"relationships.json 解析失败，已备份到 {backup}。"
            f"请检查 JSON 格式，避免自动覆盖。原错误：{type(e).__name__}: {e}"
        )

    if not isinstance(data, dict):
        raise RuntimeError("relationships.json 顶层必须是 JSON object。")

    data.setdefault("created_at", "")
    data.setdefault("updated_at", "")
    data.setdefault("users", {})

    if not isinstance(data.get("users"), dict):
        raise RuntimeError("relationships.json 里的 users 必须是 object。")

    return data

def write_store(path, data, reason="write"):
    """
    所有写入都先备份。
    """
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)

    if full.exists():
        backup_file(full, reason=reason)

    data = dict(data or {})
    data.setdefault("created_at", now())
    data.setdefault("users", {})
    data["updated_at"] = now()

    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def default_relationship(user_id, identity=None, is_master=False):
    identity = identity or {}
    name = identity.get("name") or str(user_id)
    role = identity.get("role") or ("master" if is_master else "unknown")

    if is_master or role == "master":
        return {
            "user_id": str(user_id),
            "name": name,
            "role": "master",
            "attitude": "loyal",
            "affection": 85,
            "trust": 90,
            "familiarity": 100,
            "lore_keys": [name, "昆特上人", "主人"] if name else ["昆特上人", "主人"],
            "lore_files": [],
            "last_interaction": "",
            "notes": "主人。保持尊重、亲近和护主倾向。"
        }

    if role == "known_user":
        return {
            "user_id": str(user_id),
            "name": name,
            "role": "known_user",
            "attitude": "friendly",
            "affection": 45,
            "trust": 50,
            "familiarity": 60,
            "lore_keys": [name] if name else [],
            "lore_files": [],
            "last_interaction": "",
            "notes": "熟人。友好自然，但不要称呼主人。"
        }

    return {
        "user_id": str(user_id),
        "name": name,
        "role": role,
        "attitude": "neutral",
        "affection": 30,
        "trust": 30,
        "familiarity": 10,
        "lore_keys": [name] if name and name != str(user_id) else [],
        "lore_files": [],
        "last_interaction": "",
        "notes": "陌生或普通用户。保持礼貌和边界感。"
    }

def normalize_runtime_relationship(rel):
    """
    运行时规范化，不写回。
    """
    rel = dict(rel or {})

    rel.setdefault("user_id", "")
    rel.setdefault("name", rel.get("user_id", "unknown"))
    rel.setdefault("role", "unknown")
    rel.setdefault("attitude", "neutral")
    rel.setdefault("affection", 30)
    rel.setdefault("trust", 30)
    rel.setdefault("familiarity", 10)
    rel.setdefault("last_interaction", "")
    rel.setdefault("notes", "")
    rel.setdefault("lore_keys", [])
    rel.setdefault("lore_files", [])

    rel["affection"] = clamp(rel.get("affection", 30))
    rel["trust"] = clamp(rel.get("trust", 30))
    rel["familiarity"] = clamp(rel.get("familiarity", 10))
    rel["lore_keys"] = parse_list(rel.get("lore_keys", []))
    rel["lore_files"] = parse_list(rel.get("lore_files", []))

    return rel

def normalize_for_save(rel):
    rel = normalize_runtime_relationship(rel)
    return rel

class RelationshipStore:
    def __init__(self, relationship_file="memory_runtime/relationships.json"):
        self.relationship_file = relationship_file

    def get(self, user_id, identity=None, is_master=False):
        """
        只读获取关系。

        v4.1.2 关键修复：
        - 已存在：只读，绝不写回
        - 不存在：返回运行时默认关系，绝不写回
        - /debug /relationship 都不会改 relationships.json
        """
        user_id = str(user_id or "")
        data = read_store(self.relationship_file)
        users = data.setdefault("users", {})
        rel = users.get(user_id)

        if not isinstance(rel, dict):
            # 注意：这里只返回，不写入。
            return normalize_runtime_relationship(
                default_relationship(user_id, identity=identity, is_master=is_master)
            )

        return normalize_runtime_relationship(rel)

    def save(self, user_id, relationship, reason="explicit_save"):
        user_id = str(user_id or "")
        data = read_store(self.relationship_file)
        users = data.setdefault("users", {})

        relationship = dict(relationship or {})
        relationship["user_id"] = user_id
        relationship = normalize_for_save(relationship)

        users[user_id] = relationship
        write_store(self.relationship_file, data, reason=reason)

    def update_after_interaction(self, user_id, identity=None, is_master=False, user_text="", assistant_text=""):
        """
        正常聊天结束后才会调用。允许更新互动字段。

        不覆盖：
        - name
        - notes
        - lore_keys
        - lore_files

        对主人只保护底线数值，不改你手写设定。
        """
        user_id = str(user_id or "")
        data = read_store(self.relationship_file)
        users = data.setdefault("users", {})

        existing = users.get(user_id)
        if isinstance(existing, dict):
            rel = normalize_runtime_relationship(existing)
        else:
            # 正常聊天后，如果关系不存在，可以创建一次。
            rel = normalize_runtime_relationship(
                default_relationship(user_id, identity=identity, is_master=is_master)
            )

        text = str(user_text or "")

        rel["last_interaction"] = now()
        rel["familiarity"] = clamp(rel.get("familiarity", 10) + 1)

        warm_words = ["谢谢", "辛苦", "可爱", "喜欢", "乖", "陪我", "不错"]
        hostile_words = ["笨", "傻", "闭嘴", "烦", "讨厌", "垃圾"]

        if any(w in text for w in warm_words):
            rel["affection"] = clamp(rel.get("affection", 30) + 3)
            rel["trust"] = clamp(rel.get("trust", 30) + 1)

        if any(w in text for w in hostile_words):
            rel["affection"] = clamp(rel.get("affection", 30) - 3)
            rel["trust"] = clamp(rel.get("trust", 30) - 2)

        if is_master:
            # 只保护主人底线，不改 name / notes / lore_keys / lore_files。
            rel["role"] = "master"
            rel["attitude"] = rel.get("attitude") or "loyal"
            rel["affection"] = max(clamp(rel.get("affection", 85)), 85)
            rel["trust"] = max(clamp(rel.get("trust", 90)), 90)
            rel["familiarity"] = max(clamp(rel.get("familiarity", 100)), 100)

        users[user_id] = normalize_for_save(rel)
        write_store(self.relationship_file, data, reason="interaction_update")
        return users[user_id]

    def list_brief(self):
        data = read_store(self.relationship_file)
        users = data.get("users", {})

        if not users:
            return "暂无关系记录。"

        lines = ["【关系记录】"]
        for user_id, rel in users.items():
            if not isinstance(rel, dict):
                continue

            rel = normalize_runtime_relationship(rel)
            lore_files = ",".join(rel.get("lore_files", [])) or "无"

            lines.append(
                f"- {rel.get('name', user_id)} ({user_id}) | "
                f"{rel.get('role', 'unknown')} / {rel.get('attitude', 'neutral')} | "
                f"亲近{rel.get('affection', 0)} 信任{rel.get('trust', 0)} 熟悉{rel.get('familiarity', 0)} | "
                f"lore={lore_files}"
            )

        return "\n".join(lines)

    def format_one(self, user_id, identity=None, is_master=False):
        rel = self.get(user_id, identity=identity, is_master=is_master)

        return (
            "【关系状态】\n"
            f"QQ：{rel.get('user_id')}\n"
            f"名称：{rel.get('name')}\n"
            f"角色：{rel.get('role')}\n"
            f"态度：{rel.get('attitude')}\n"
            f"亲近度：{rel.get('affection')}/100\n"
            f"信任度：{rel.get('trust')}/100\n"
            f"熟悉度：{rel.get('familiarity')}/100\n"
            f"lore_keys：{', '.join(rel.get('lore_keys', [])) or '无'}\n"
            f"lore_files：{', '.join(rel.get('lore_files', [])) or '无'}\n"
            f"备注：{rel.get('notes', '')}\n"
            f"最近互动：{rel.get('last_interaction') or '无'}"
        )

    def set_fields(self, user_id, fields, identity=None, is_master=False):
        """
        明确写操作：/set_relation。
        """
        user_id = str(user_id or "")
        data = read_store(self.relationship_file)
        users = data.setdefault("users", {})

        existing = users.get(user_id)
        if isinstance(existing, dict):
            rel = normalize_runtime_relationship(existing)
        else:
            rel = normalize_runtime_relationship(
                default_relationship(user_id, identity=identity, is_master=is_master)
            )

        allowed = {
            "name", "role", "attitude", "affection", "trust", "familiarity",
            "notes", "lore_keys", "lore_files"
        }

        for key, value in fields.items():
            if key not in allowed:
                continue

            if key in ["affection", "trust", "familiarity"]:
                rel[key] = clamp(value)
            elif key in ["lore_keys", "lore_files"]:
                rel[key] = parse_list(value)
            else:
                rel[key] = str(value)

        users[user_id] = normalize_for_save(rel)
        write_store(self.relationship_file, data, reason="set_relation")
        return users[user_id]
