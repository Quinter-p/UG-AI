from storage.emotion_store import EmotionStore
from storage.session_store import SessionHistoryStore, build_session_key
from storage.relationship_store import RelationshipStore
from storage.conversation_memory_store import ConversationMemoryStore
from storage.world_lore_store import WorldLoreStore
from storage.speak_runtime_store import SpeakRuntimeStore


def command_node(state):
    text = str(state.get("clean_text") or "").strip()
    config = state.get("config") or {}

    if not text.startswith("/"):
        return {}

    lower = text.lower()

    if lower in ["/help", "/菜单", "/commands"]:
        return {"route": "command_reply", "final_reply": help_text()}

    if lower in ["/status", "/心情", "/life"]:
        emotion_cfg = config.get("emotion", {})
        store = EmotionStore(emotion_cfg.get("state_file", "memory_runtime/emotion_state.json"))
        return {"route": "command_reply", "final_reply": store.format_status()}

    if lower in ["/speak_status", "/speak"]:
        store = speak_runtime_store(config)
        return {"route": "command_reply", "final_reply": store.format_status()}

    if lower.startswith("/speak_mode "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}

        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/speak_mode quiet|normal|active"}

        store = speak_runtime_store(config)
        try:
            mode = store.set_mode(parts[1])
        except Exception:
            return {"route": "command_reply", "final_reply": "模式只能是 quiet / normal / active。"}

        return {"route": "command_reply", "final_reply": f"主动发言模式已切换为：{mode}"}

    if lower.startswith("/speak_cooldown "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}

        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/speak_cooldown 秒数"}

        store = speak_runtime_store(config)
        try:
            seconds = store.set_cooldown(int(parts[1]))
        except Exception:
            return {"route": "command_reply", "final_reply": "请输入 0-3600 的秒数。"}

        return {"route": "command_reply", "final_reply": f"主动发言冷却时间已设为：{seconds} 秒"}

    if lower in ["/history", "/短期记忆"]:
        memory_cfg = config.get("short_memory", {}) or {}
        store = SessionHistoryStore(
            history_file=memory_cfg.get("history_file", "memory_runtime/session_history.json"),
            max_turns=int(memory_cfg.get("max_turns", 6)),
        )
        session_key = build_session_key(
            message_type=state.get("message_type", "private"),
            user_id=state.get("user_id", ""),
            group_id=state.get("group_id"),
        )
        return {"route": "command_reply", "final_reply": store.format_session(session_key)}

    if lower in ["/memory", "/memories", "/长期记忆"]:
        mem_store = conversation_memory_store(config)
        return {"route": "command_reply", "final_reply": mem_store.list_memories()}

    if lower.startswith("/remember "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "长期记忆只能由主人写入。"}

        content = text.split(maxsplit=1)[1].strip()
        mem_store = conversation_memory_store(config)
        item = mem_store.add_memory(
            content=content,
            category="personal",
            subject=(state.get("identity") or {}).get("name", ""),
            tags="manual",
            source_user_id=state.get("user_id", ""),
            source_name=(state.get("identity") or {}).get("name", ""),
        )
        return {
            "route": "command_reply",
            "final_reply": f"已写入长期个人记忆：#{item.get('id')} {item.get('content')}" if item else "没有识别到要记住的内容。"
        }

    if lower.startswith("/forget "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}

        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/forget 记忆ID"}

        mem_store = conversation_memory_store(config)
        ok = mem_store.delete_memory(parts[1])
        return {"route": "command_reply", "final_reply": f"已删除长期记忆 #{parts[1]}" if ok else "没有找到这条长期记忆。"}

    if lower in ["/lore", "/world_lore", "/世界观"]:
        lore_store = world_lore_store(config)
        return {"route": "command_reply", "final_reply": lore_store.summary()}

    if lower in ["/reload_lore", "/重载世界观"]:
        lore_store = world_lore_store(config)
        return {"route": "command_reply", "final_reply": "世界观文件会在每轮对话中从 knowledge 文件夹重新读取。\n" + lore_store.summary()}

    if lower in ["/relationship", "/relation", "/关系"]:
        rel_store = relationship_store(config)
        return {
            "route": "command_reply",
            "final_reply": rel_store.format_one(
                user_id=state.get("user_id", ""),
                identity=state.get("identity") or {},
                is_master=bool(state.get("is_master", False)),
            ),
        }

    if lower in ["/relationships", "/relations", "/关系列表"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        rel_store = relationship_store(config)
        return {"route": "command_reply", "final_reply": rel_store.list_brief()}

    if lower.startswith("/set_relation "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return handle_set_relation(text, state, config)

    if lower in ["/reset_status", "/reset_state", "/重置状态"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}

        emotion_cfg = config.get("emotion", {})
        emotion_store = EmotionStore(emotion_cfg.get("state_file", "memory_runtime/emotion_state.json"))
        emotion_store.reset()

        memory_cfg = config.get("short_memory", {}) or {}
        history_store = SessionHistoryStore(
            history_file=memory_cfg.get("history_file", "memory_runtime/session_history.json"),
            max_turns=int(memory_cfg.get("max_turns", 6)),
        )
        session_key = build_session_key(
            message_type=state.get("message_type", "private"),
            user_id=state.get("user_id", ""),
            group_id=state.get("group_id"),
        )
        history_store.clear_session(session_key)
        return {"route": "command_reply", "final_reply": "状态和当前会话短期记忆已重置。"}

    if lower in ["/debug", "/debug_state"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": debug_text(state)}

    return {}


def speak_runtime_store(config):
    speak_cfg = config.get("speak_decision", {}) or {}
    return SpeakRuntimeStore(
        runtime_file=speak_cfg.get("runtime_file", "memory_runtime/speak_runtime.json")
    )


def conversation_memory_store(config):
    mem_cfg = config.get("long_memory", {}) or {}
    return ConversationMemoryStore(
        memory_file=mem_cfg.get("conversation_memory_file", "memory_runtime/conversation_memory.json")
    )


def world_lore_store(config):
    mem_cfg = config.get("long_memory", {}) or {}
    return WorldLoreStore(
        lore_dir=mem_cfg.get("world_lore_dir", "knowledge"),
        max_chars=int(mem_cfg.get("lore_max_chars", 3500)),
    )


def relationship_store(config):
    rel_cfg = config.get("relationship", {}) or {}
    return RelationshipStore(
        relationship_file=rel_cfg.get("relationship_file", "memory_runtime/relationships.json")
    )


def parse_kv_pairs(text):
    fields = {}
    parts = text.split()
    for item in parts:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def handle_set_relation(text, state, config):
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        return {
            "route": "command_reply",
            "final_reply": (
                "格式：/set_relation QQ号 name=名字 role=known_user attitude=friendly affection=50\n"
                "绑定世界观例：/set_relation 1151208708 name=6m lore_keys=6m,六米 lore_files=characters_6m.md\n"
                "敌人例：/set_relation 123456789 name=黑沐 role=enemy attitude=hostile affection=0 trust=0 familiarity=80"
            )
        }

    user_id = parts[1].strip()
    fields = parse_kv_pairs(parts[2])
    if not user_id or not fields:
        return {"route": "command_reply", "final_reply": "没有识别到 QQ号 或字段。"}

    rel_store = relationship_store(config)
    rel = rel_store.set_fields(user_id, fields)
    return {
        "route": "command_reply",
        "final_reply": (
            "关系已更新：\n"
            f"{rel.get('name')} ({rel.get('user_id')})\n"
            f"role={rel.get('role')} attitude={rel.get('attitude')}\n"
            f"亲近={rel.get('affection')} 信任={rel.get('trust')} 熟悉={rel.get('familiarity')}\n"
            f"lore_keys={','.join(rel.get('lore_keys', [])) or '无'}\n"
            f"lore_files={','.join(rel.get('lore_files', [])) or '无'}"
        )
    }


def help_text():
    return (
        "【UGAI Agent v6】\n"
        "普通聊天：私聊直接说；群聊根据主动发言策略判断是否回复。\n"
        "唤醒：@机器人、/ugai、施耐德、器灵\n"
        "状态：/status 或 /心情\n"
        "短期记忆：/history\n"
        "长期个人记忆：/memory\n"
        "世界观文件：/lore\n"
        "关系：/relationship\n"
        "主动发言状态：/speak_status\n"
        "主动发言模式：/speak_mode quiet|normal|active（主人）\n"
        "主动发言冷却：/speak_cooldown 秒数（主人）\n"
        "调试：/debug\n"
        "当前版本：主动发言决策。"
    )


def debug_text(state):
    identity = state.get("identity") or {}
    emotion = state.get("emotion_state") or {}
    meta = state.get("prompt_meta") or {}
    usage = state.get("usage_metadata") or {}
    policy = state.get("reply_policy") or {}
    rel = state.get("relationship_state") or {}
    relp = state.get("relationship_policy") or {}
    expr = state.get("expression_style") or {}
    speak = state.get("speak_decision") or {}

    return (
        "【UGAI Debug】\n"
        f"route：{state.get('route')}\n"
        f"message_type：{state.get('message_type')}\n"
        f"user_id：{state.get('user_id')}\n"
        f"group_id：{state.get('group_id')}\n"
        f"identity：{identity.get('name')} / {identity.get('role')} / {identity.get('source')}\n"
        f"relationship：{rel.get('name')} / {rel.get('role')} / {rel.get('attitude')}\n"
        f"lore_keys：{rel.get('lore_keys')}\n"
        f"lore_files：{rel.get('lore_files')}\n"
        f"relationship_policy：{relp.get('stance')} / {relp.get('reason')}\n"
        f"speak_decision：reply={speak.get('should_reply')} reason={speak.get('reason')} mode={speak.get('mode')} priority={speak.get('priority')}\n"
        f"mood：{emotion.get('mood_text')} ({emotion.get('mood')})\n"
        f"reply_mode：{policy.get('mode')} / {policy.get('reason')}\n"
        f"reply_budget：{policy.get('max_chars')}字 / {policy.get('max_paragraphs')}段\n"
        f"expression_tone：{expr.get('tone')}\n"
        f"expression_distance：{expr.get('distance')}\n"
        f"catness：{expr.get('catness')}\n"
        f"builder_strategy：{meta.get('builder_strategy')}\n"
        f"history_turns_used：{meta.get('history_turns_used')}\n"
        f"world_lore_files_used：{meta.get('world_lore_files_used')}\n"
        f"conversation_memory_items_used：{meta.get('conversation_memory_items_used')}\n"
        f"prompt_eval_count：{usage.get('prompt_eval_count')}\n"
        f"eval_count：{usage.get('eval_count')}\n"
        f"clean_text：{state.get('clean_text')}"
    )
