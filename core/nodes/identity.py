from storage.relationship_store import RelationshipStore


def as_str_id(value):
    if value is None:
        return ""
    return str(value).strip()


def get_identity_config(config):
    data = config.get("identity", {}) or {}
    if not isinstance(data, dict):
        return {}
    return data


def get_master_ids(config):
    """
    兼容旧配置和新配置。

    新推荐：
    "identity": {
      "master_ids": ["674398417"]
    }

    兼容旧写法：
    "master_ids": [...]
    "identity": {
      "674398417": {"role": "master"}
    }
    "identity": {
      "users": {
        "674398417": {"role": "master"}
      }
    }
    """
    master_ids = set()

    root_master_ids = config.get("master_ids", [])
    if isinstance(root_master_ids, list):
        for item in root_master_ids:
            sid = as_str_id(item)
            if sid:
                master_ids.add(sid)

    identity_cfg = get_identity_config(config)

    ids = identity_cfg.get("master_ids", [])
    if isinstance(ids, list):
        for item in ids:
            sid = as_str_id(item)
            if sid:
                master_ids.add(sid)

    # 兼容 identity.users
    users = identity_cfg.get("users", {})
    if isinstance(users, dict):
        for uid, info in users.items():
            if isinstance(info, dict) and str(info.get("role", "")).lower() == "master":
                sid = as_str_id(uid)
                if sid:
                    master_ids.add(sid)

    # 兼容 identity 直接以 QQ号 为 key
    for uid, info in identity_cfg.items():
        if uid in ["master_ids", "users", "default"]:
            continue
        if isinstance(info, dict) and str(info.get("role", "")).lower() == "master":
            sid = as_str_id(uid)
            if sid:
                master_ids.add(sid)

    # 兼容 identities
    identities = config.get("identities", {})
    if isinstance(identities, dict):
        for uid, info in identities.items():
            if isinstance(info, dict) and str(info.get("role", "")).lower() == "master":
                sid = as_str_id(uid)
                if sid:
                    master_ids.add(sid)

    return master_ids


def get_legacy_identity(config, user_id):
    """
    读取旧 config identity，仅作为兜底，不再作为人物设定主源。
    """
    user_id = as_str_id(user_id)
    identity_cfg = get_identity_config(config)

    candidates = []

    users = identity_cfg.get("users", {})
    if isinstance(users, dict):
        candidates.append(users.get(user_id))

    candidates.append(identity_cfg.get(user_id))

    identities = config.get("identities", {})
    if isinstance(identities, dict):
        candidates.append(identities.get(user_id))

    for item in candidates:
        if isinstance(item, dict):
            return item

    return {}


def title_for(role, name, is_master):
    if is_master or role == "master":
        return "主人"
    if role in ["known_user", "friend"]:
        return name or "道友"
    if role in ["enemy", "hostile"]:
        return name or "敌对目标"
    return name or "对方"


def identity_node(state):
    """
    v5.1：身份从 relationships.json 派生。

    分工：
    - config.identity.master_ids：只负责最高权限识别
    - relationships.json：负责 name / role / notes 等人物身份信息
    - knowledge/*.md：负责详细世界观设定
    """
    config = state.get("config") or {}
    user_id = as_str_id(state.get("user_id", ""))

    rel_cfg = config.get("relationship", {}) or {}
    store = RelationshipStore(
        relationship_file=rel_cfg.get("relationship_file", "memory_runtime/relationships.json")
    )

    master_ids = get_master_ids(config)
    legacy = get_legacy_identity(config, user_id)

    # 先用 master_ids 和旧配置判断一次，给新用户默认关系兜底。
    legacy_role = str(legacy.get("role", "") or "").lower()
    legacy_is_master = user_id in master_ids or legacy_role == "master"

    # 读取 relationships.json。这个 get() 是 read-only，不会因为 /debug 覆盖文件。
    rel = store.get(
        user_id=user_id,
        identity=legacy,
        is_master=legacy_is_master,
    )

    role = str(rel.get("role") or legacy.get("role") or "unknown").strip()
    name = str(rel.get("name") or legacy.get("name") or user_id).strip()
    notes = str(rel.get("notes") or legacy.get("description") or "").strip()

    # v5.1 后，relationship 里的 role=master 也可判定主人。
    is_master = (
        user_id in master_ids
        or role.lower() == "master"
        or legacy_role == "master"
    )

    if is_master:
        role = "master"

    identity = {
        "user_id": user_id,
        "name": name,
        "role": role,
        "title": legacy.get("title") or title_for(role.lower(), name, is_master),
        "description": notes,
        "source": "relationships.json",
        "legacy_config_used": bool(legacy),
    }

    return {
        "identity": identity,
        "is_master": bool(is_master),
    }
