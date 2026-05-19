# check_ugai_project_health.py
# 用法：
# python check_ugai_project_health.py
#
# 作用：
# 检查 v5.2 后几个关键文件是否存在、配置是否合理。

from pathlib import Path
import json


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


def ok(msg):
    print("[OK]", msg)


def warn(msg):
    print("[WARN]", msg)


def fail(msg):
    print("[FAIL]", msg)


def main():
    root = Path.cwd()

    print("UGAI Project Health Check")
    print("=" * 60)

    config_path = root / "config.json"
    if not config_path.exists():
        fail("缺少 config.json")
        return

    cfg = read_json(config_path)
    if "__error__" in cfg:
        fail(f"config.json 解析失败：{cfg['__error__']}")
        return

    ok("config.json 可读取")

    identity = cfg.get("identity", {})
    if not isinstance(identity, dict):
        fail("config.identity 不是 object")
    else:
        master_ids = identity.get("master_ids", [])
        if isinstance(master_ids, list) and master_ids:
            ok(f"identity.master_ids = {master_ids}")
        else:
            fail("identity.master_ids 缺失或为空，主人命令可能不能用")

        source = identity.get("source")
        if source == "relationships.json":
            ok("identity.source = relationships.json")
        else:
            warn("identity.source 不是 relationships.json；可能还没完成 v5.2 清理")

    rel_file = cfg.get("relationship", {}).get("relationship_file", "memory_runtime/relationships.json")
    rel_path = root / rel_file
    if rel_path.exists():
        rel = read_json(rel_path)
        if "__error__" in rel:
            fail(f"relationships.json 解析失败：{rel['__error__']}")
        elif isinstance(rel.get("users"), dict):
            ok(f"relationships.json users = {len(rel.get('users', {}))} entries")
        else:
            fail("relationships.json 缺少 users object")
    else:
        warn(f"找不到 relationships.json: {rel_file}")

    lore_dir = cfg.get("long_memory", {}).get("world_lore_dir", "knowledge")
    lore_path = root / lore_dir
    if lore_path.exists() and lore_path.is_dir():
        files = sorted(list(lore_path.glob("*.md")) + list(lore_path.glob("*.txt")))
        if files:
            ok(f"knowledge 文件数 = {len(files)}")
            for p in files[:10]:
                print("   -", p.name)
        else:
            warn("knowledge 文件夹存在，但没有 .md/.txt 世界观文件")
    else:
        warn(f"找不到 knowledge 文件夹：{lore_dir}")

    conv_file = cfg.get("long_memory", {}).get("conversation_memory_file", "memory_runtime/conversation_memory.json")
    conv_path = root / conv_file
    if conv_path.exists():
        conv = read_json(conv_path)
        if "__error__" in conv:
            fail(f"conversation_memory.json 解析失败：{conv['__error__']}")
        else:
            memories = conv.get("memories", [])
            ok(f"conversation_memory memories = {len(memories) if isinstance(memories, list) else 'unknown'}")
    else:
        warn(f"长期个人记忆文件还不存在：{conv_file}，首次 /remember 后会创建")

    session_path = root / "memory_runtime" / "session_history.json"
    if session_path.exists():
        ok("session_history.json 存在")
    else:
        warn("session_history.json 不存在，首次聊天后会创建")

    emotion_path = root / "memory_runtime" / "emotion_state.json"
    if emotion_path.exists():
        ok("emotion_state.json 存在")
    else:
        warn("emotion_state.json 不存在，首次 /status 或聊天后会创建")

    print("=" * 60)
    print("检查完成。")


if __name__ == "__main__":
    main()
