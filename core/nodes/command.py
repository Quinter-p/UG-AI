from storage.emotion_store import EmotionStore
from storage.session_store import SessionHistoryStore, build_session_key
from storage.relationship_store import RelationshipStore
from storage.conversation_memory_store import ConversationMemoryStore
from storage.world_lore_store import WorldLoreStore
from storage.speak_runtime_store import SpeakRuntimeStore
from storage.event_log_store import EventLogStore
from storage.event_dedup_store import EventDedupStore
from storage.reflection_memory_store import ReflectionMemoryStore
from storage.fact_memory_store import FactMemoryStore
from storage.task_store import TaskStore
from storage.tool_bus_store import ToolBusStore
from core.nodes.reflection import run_reflection_now
from core.nodes.tool_executor import execute_tool_call, execute_approved_calls
import json


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

    if lower in ["/tools", "/tool_list", "/工具"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": tool_bus_store(config).list_tools()}

    if lower in ["/tool_policy", "/approval_policy", "/工具策略"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": tool_bus_store(config).format_policy()}

    if lower in ["/tool_calls", "/tool_status", "/工具状态"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": tool_bus_store(config).list_calls(status="open", limit=20)}

    if lower in ["/tool_calls_all", "/工具全部"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": tool_bus_store(config).list_calls(status="all", limit=30)}

    if lower.startswith("/tool_request "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return handle_tool_request(text, state, config)

    if lower.startswith("/tool_approve "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_approve 调用ID"}
        actor = (state.get("identity") or {}).get("name", "")
        item, err = tool_bus_store(config).approve_call(parts[1].strip(), actor=actor)
        if not item:
            return {"route": "command_reply", "final_reply": "没有找到这个工具请求。"}
        if err:
            return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 无法批准：{err}。"}
        return {
            "route": "command_reply",
            "final_reply": (
                f"工具请求 #{item.get('id')} 已批准：{item.get('tool_name')}。\n"
                f"执行：/tool_execute {item.get('id')}"
            )
        }

    if lower.startswith("/tool_execute "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_execute 调用ID"}
        result = execute_tool_call(config, parts[1].strip())
        return {"route": "command_reply", "final_reply": result.get("message", "执行完成。")}

    if lower.startswith("/tool_execute_all"):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        result = execute_approved_calls(config, limit=5)
        return {"route": "command_reply", "final_reply": result.get("message", "执行完成。")}

    if lower.startswith("/tool_reject "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_reject 调用ID [原因]"}
        reason = parts[2] if len(parts) >= 3 else ""
        actor = (state.get("identity") or {}).get("name", "")
        item, err = tool_bus_store(config).reject_call(parts[1].strip(), actor=actor, reason=reason)
        if not item:
            return {"route": "command_reply", "final_reply": "没有找到这个工具请求。"}
        if err:
            return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 无法拒绝：{err}。"}
        return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 已拒绝。"}

    if lower.startswith("/tool_done "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_done 调用ID [结果]"}
        result = parts[2] if len(parts) >= 3 else ""
        item = tool_bus_store(config).update_call(parts[1], "done", result=result)
        if not item:
            return {"route": "command_reply", "final_reply": "没有找到这个工具请求。"}
        return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 已标记完成。"}

    if lower.startswith("/tool_cancel "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_cancel 调用ID [原因]"}
        result = parts[2] if len(parts) >= 3 else ""
        item = tool_bus_store(config).update_call(parts[1], "cancelled", result=result)
        if not item:
            return {"route": "command_reply", "final_reply": "没有找到这个工具请求。"}
        return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 已取消。"}

    if lower.startswith("/tool_enable "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_enable 工具名"}
        item = tool_bus_store(config).set_tool_enabled(parts[1], True)
        return {"route": "command_reply", "final_reply": f"工具 {parts[1]} 已启用。" if item else "没有找到这个工具。"}

    if lower.startswith("/tool_disable "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/tool_disable 工具名"}
        item = tool_bus_store(config).set_tool_enabled(parts[1], False)
        return {"route": "command_reply", "final_reply": f"工具 {parts[1]} 已禁用。" if item else "没有找到这个工具。"}

    if lower.startswith("/tool_risk "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 3:
            return {"route": "command_reply", "final_reply": "格式：/tool_risk 工具名 low|medium|high"}
        try:
            item = tool_bus_store(config).set_tool_risk(parts[1], parts[2])
        except Exception:
            return {"route": "command_reply", "final_reply": "风险等级只能是 low / medium / high。"}
        return {"route": "command_reply", "final_reply": f"工具 {parts[1]} 风险等级已设为 {parts[2]}。" if item else "没有找到这个工具。"}

    if lower in ["/event_status", "/trace_status", "/事件状态"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        store = event_dedup_store(config)
        return {"route": "command_reply", "final_reply": store.format_status(limit=10)}

    if lower in ["/speak_status", "/speak"]:
        return {"route": "command_reply", "final_reply": speak_runtime_store(config).format_status()}

    if lower.startswith("/speak_mode "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/speak_mode quiet|normal|active"}
        try:
            mode = speak_runtime_store(config).set_mode(parts[1])
        except Exception:
            return {"route": "command_reply", "final_reply": "模式只能是 quiet / normal / active。"}
        return {"route": "command_reply", "final_reply": f"主动发言模式已切换为：{mode}"}

    if lower.startswith("/speak_cooldown "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/speak_cooldown 秒数"}
        try:
            seconds = speak_runtime_store(config).set_cooldown(int(parts[1]))
        except Exception:
            return {"route": "command_reply", "final_reply": "请输入 0-3600 的秒数。"}
        return {"route": "command_reply", "final_reply": f"主动发言冷却时间已设为：{seconds} 秒"}

    if lower in ["/tasks", "/task", "/任务"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": task_store(config).list_tasks(status="pending", limit=30)}

    if lower in ["/tasks_all", "/task_all", "/全部任务"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": task_store(config).list_tasks(status="all", limit=50)}

    if lower.startswith("/task_add "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        content = text.split(maxsplit=1)[1].strip()
        if not content:
            return {"route": "command_reply", "final_reply": "格式：/task_add 任务标题 | 可选说明"}
        title, desc = split_title_desc(content)
        ev = state.get("event_context") or {}
        identity = state.get("identity") or {}
        item = task_store(config).add_task(
            title=title,
            description=desc,
            owner=identity.get("name", ""),
            priority="normal",
            tags=["manual"],
            source_text=content,
            trace_id=ev.get("trace_id", ""),
            source_event_id=ev.get("source_event_id", ""),
            created_by=identity.get("name", ""),
        )
        return {"route": "command_reply", "final_reply": f"已登记任务 #{item.get('id')}：{item.get('title')}"}

    if lower.startswith("/task_done "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/task_done 任务ID [备注]"}
        notes = parts[2] if len(parts) >= 3 else ""
        item = task_store(config).update_status(parts[1], "done", notes=notes)
        return {"route": "command_reply", "final_reply": f"任务 #{item.get('id')} 已完成：{item.get('title')}" if item else "没有找到这个任务。"}

    if lower.startswith("/task_cancel "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/task_cancel 任务ID [原因]"}
        notes = parts[2] if len(parts) >= 3 else ""
        item = task_store(config).update_status(parts[1], "cancelled", notes=notes)
        return {"route": "command_reply", "final_reply": f"任务 #{item.get('id')} 已取消：{item.get('title')}" if item else "没有找到这个任务。"}

    if lower in ["/events", "/event_log", "/事件日志"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": event_log_store(config).format_recent(limit=10)}

    if lower in ["/facts", "/fact", "/事实记忆"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": fact_memory_store(config).list_facts(limit=30)}

    if lower.startswith("/fact_add "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        content = text.split(maxsplit=1)[1].strip()
        if not content:
            return {"route": "command_reply", "final_reply": "格式：/fact_add 要写入的事实"}
        identity = state.get("identity") or {}
        item = fact_memory_store(config).add_fact(
            content=content,
            subject=identity.get("name", ""),
            tags=["manual"],
            confidence=0.8,
            importance=0.6,
            source="manual_fact_add",
        )
        return {"route": "command_reply", "final_reply": f"已写入事实记忆 #{item.get('id')}：{item.get('content')}"}

    if lower.startswith("/fact_from_event "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split(maxsplit=2)
        if len(parts) < 3:
            return {"route": "command_reply", "final_reply": "格式：/fact_from_event 事件ID 要写入的事实"}
        event_id = parts[1].strip()
        content = parts[2].strip()
        ev = event_log_store(config).get_by_id(event_id)
        if not ev:
            return {"route": "command_reply", "final_reply": f"没有找到事件 #{event_id}。先用 /events 查看最近事件。"}
        quote = f"user: {ev.get('user_text', '')}\nai: {ev.get('assistant_reply', '')}"
        item = fact_memory_store(config).add_fact(
            content=content,
            subject=ev.get("speaker_name", "") or ev.get("user_id", ""),
            tags=["from_event"],
            evidence_event_id=event_id,
            evidence_quote=quote[:500],
            confidence=0.85,
            importance=0.7,
            source="manual_fact_from_event",
        )
        return {"route": "command_reply", "final_reply": f"已从事件 #{event_id} 写入事实记忆 #{item.get('id')}：{item.get('content')}"}

    if lower.startswith("/forget_fact "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/forget_fact 事实ID"}
        ok = fact_memory_store(config).retire_fact(parts[1])
        return {"route": "command_reply", "final_reply": f"已停用事实记忆 #{parts[1]}" if ok else "没有找到这条事实记忆。"}

    if lower in ["/reflections", "/reflection", "/反思记忆"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return {"route": "command_reply", "final_reply": reflection_memory_store(config).list_reflections(limit=20)}

    if lower.startswith("/reflect"):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        try:
            result = run_reflection_now(config, source="manual_reflect")
            return {"route": "command_reply", "final_reply": result.get("message", "反思完成。")}
        except Exception as e:
            return {"route": "command_reply", "final_reply": f"反思失败：{type(e).__name__}: {e}"}

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
        return {"route": "command_reply", "final_reply": conversation_memory_store(config).list_memories()}

    if lower.startswith("/remember "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "长期记忆只能由主人写入。"}
        content = text.split(maxsplit=1)[1].strip()
        identity = state.get("identity") or {}
        item = conversation_memory_store(config).add_memory(
            content=content,
            category="personal",
            subject=identity.get("name", ""),
            tags="manual",
            source_user_id=state.get("user_id", ""),
            source_name=identity.get("name", ""),
        )
        return {"route": "command_reply", "final_reply": f"已写入长期个人记忆：#{item.get('id')} {item.get('content')}" if item else "没有识别到要记住的内容。"}

    if lower.startswith("/forget "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        parts = text.split()
        if len(parts) < 2:
            return {"route": "command_reply", "final_reply": "格式：/forget 记忆ID"}
        ok = conversation_memory_store(config).delete_memory(parts[1])
        return {"route": "command_reply", "final_reply": f"已删除长期记忆 #{parts[1]}" if ok else "没有找到这条长期记忆。"}

    if lower in ["/lore", "/world_lore", "/世界观"]:
        return {"route": "command_reply", "final_reply": world_lore_store(config).summary()}

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
        return {"route": "command_reply", "final_reply": relationship_store(config).list_brief()}

    if lower.startswith("/set_relation "):
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        return handle_set_relation(text, state, config)

    if lower in ["/reset_status", "/reset_state", "/重置状态"]:
        if not state.get("is_master", False):
            return {"route": "command_reply", "final_reply": "这条命令只有主人能用。"}
        emotion_cfg = config.get("emotion", {})
        EmotionStore(emotion_cfg.get("state_file", "memory_runtime/emotion_state.json")).reset()
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


def handle_tool_request(text, state, config):
    parts = text.split(maxsplit=2)
    if len(parts) < 2:
        return {"route": "command_reply", "final_reply": "格式：/tool_request 工具名 {JSON参数}"}

    tool_name = parts[1].strip()
    args = {}
    if len(parts) >= 3:
        raw = parts[2].strip()
        try:
            args = json.loads(raw)
        except Exception:
            args = {"text": raw}

    ev = state.get("event_context") or {}
    item = tool_bus_store(config).add_call(
        tool_name=tool_name,
        args=args,
        reason="主人手动登记工具请求",
        proposed_by="master",
        source_text=text,
        trace_id=ev.get("trace_id", ""),
        requires_approval=True,
    )

    if item.get("status") in ["blocked", "rejected"]:
        return {"route": "command_reply", "final_reply": f"工具请求 #{item.get('id')} 未进入执行队列：{item.get('error')}。"}

    return {
        "route": "command_reply",
        "final_reply": (
            f"已登记工具请求 #{item.get('id')}：{item.get('tool_name')}，状态={item.get('status')}。\n"
            f"批准：/tool_approve {item.get('id')}\n"
            f"执行：/tool_execute {item.get('id')}"
        )
    }


def split_title_desc(content):
    content = str(content or "").strip()
    for sep in [" | ", "|", "；", ";"]:
        if sep in content:
            a, b = content.split(sep, 1)
            return a.strip(), b.strip()
    return content, ""


def tool_bus_store(config):
    tool_cfg = config.get("tool_bus", {}) or {}
    return ToolBusStore(
        registry_file=tool_cfg.get("registry_file", "memory_runtime/tool_registry.json"),
        runtime_file=tool_cfg.get("runtime_file", "memory_runtime/tool_calls.json"),
    )


def task_store(config):
    task_cfg = config.get("task_registry", {}) or {}
    return TaskStore(task_file=task_cfg.get("task_file", "memory_runtime/task_registry.json"))


def event_dedup_store(config):
    event_cfg = config.get("event_model", {}) or {}
    return EventDedupStore(runtime_file=event_cfg.get("runtime_file", "memory_runtime/event_runtime.json"))


def speak_runtime_store(config):
    speak_cfg = config.get("speak_decision", {}) or {}
    return SpeakRuntimeStore(runtime_file=speak_cfg.get("runtime_file", "memory_runtime/speak_runtime.json"))


def event_log_store(config):
    event_cfg = config.get("event_log", {}) or {}
    return EventLogStore(event_file=event_cfg.get("event_file", "memory_runtime/event_log.jsonl"))


def reflection_memory_store(config):
    refl_cfg = config.get("reflection", {}) or {}
    return ReflectionMemoryStore(reflection_file=refl_cfg.get("reflection_file", "memory_runtime/reflection_memory.json"))


def fact_memory_store(config):
    fact_cfg = config.get("fact_memory", {}) or {}
    return FactMemoryStore(fact_file=fact_cfg.get("fact_file", "memory_runtime/fact_memory.json"))


def conversation_memory_store(config):
    mem_cfg = config.get("long_memory", {}) or {}
    return ConversationMemoryStore(memory_file=mem_cfg.get("conversation_memory_file", "memory_runtime/conversation_memory.json"))


def world_lore_store(config):
    mem_cfg = config.get("long_memory", {}) or {}
    return WorldLoreStore(
        lore_dir=mem_cfg.get("world_lore_dir", "knowledge"),
        max_chars=int(mem_cfg.get("lore_max_chars", 3500)),
    )


def relationship_store(config):
    rel_cfg = config.get("relationship", {}) or {}
    return RelationshipStore(relationship_file=rel_cfg.get("relationship_file", "memory_runtime/relationships.json"))


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
    rel = relationship_store(config).set_fields(user_id, fields)
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
        "【UGAI Agent v8.3】\n"
        "工具请求：/tool_request 工具名 {JSON参数}\n"
        "批准工具：/tool_approve 调用ID\n"
        "执行工具：/tool_execute 调用ID\n"
        "执行所有 approved：/tool_execute_all\n"
        "工具状态：/tool_status\n"
        "工具列表：/tools\n"
        "当前可执行工具：task_registry / memory_write\n"
        "仍不执行：local_shell / browser\n"
        "任务：/tasks /task_add /task_done /task_cancel\n"
        "记忆：/facts /events /reflections /memory /history\n"
        "关系：/relationship\n"
        "调试：/debug"
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
    ev = state.get("event_context") or {}

    return (
        "【UGAI Debug】\n"
        f"trace_id：{ev.get('trace_id')}\n"
        f"message_key：{ev.get('message_key')}\n"
        f"source_event_id：{ev.get('source_event_id')}\n"
        f"duplicate：{ev.get('duplicate')}\n"
        f"route：{state.get('route')}\n"
        f"message_type：{state.get('message_type')}\n"
        f"user_id：{state.get('user_id')}\n"
        f"group_id：{state.get('group_id')}\n"
        f"identity：{identity.get('name')} / {identity.get('role')} / {identity.get('source')}\n"
        f"relationship：{rel.get('name')} / {rel.get('role')} / {rel.get('attitude')}\n"
        f"relationship_policy：{relp.get('stance')} / {relp.get('reason')}\n"
        f"speak_decision：reply={speak.get('should_reply')} reason={speak.get('reason')} mode={speak.get('mode')} priority={speak.get('priority')}\n"
        f"mood：{emotion.get('mood_text')} ({emotion.get('mood')})\n"
        f"reply_mode：{policy.get('mode')} / {policy.get('reason')}\n"
        f"expression_tone：{expr.get('tone')}\n"
        f"builder_strategy：{meta.get('builder_strategy')}\n"
        f"history_turns_used：{meta.get('history_turns_used')}\n"
        f"world_lore_files_used：{meta.get('world_lore_files_used')}\n"
        f"conversation_memory_items_used：{meta.get('conversation_memory_items_used')}\n"
        f"fact_memory_items_used：{meta.get('fact_memory_items_used')}\n"
        f"reflection_memory_items_used：{meta.get('reflection_memory_items_used')}\n"
        f"task_items_used：{meta.get('task_items_used')}\n"
        f"tool_call_items_used：{meta.get('tool_call_items_used')}\n"
        f"event_log_id：{state.get('event_log_id')}\n"
        f"prompt_eval_count：{usage.get('prompt_eval_count')}\n"
        f"eval_count：{usage.get('eval_count')}\n"
        f"clean_text：{state.get('clean_text')}"
    )
