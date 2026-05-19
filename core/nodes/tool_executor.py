import json
from storage.tool_bus_store import ToolBusStore
from storage.task_store import TaskStore
from storage.fact_memory_store import FactMemoryStore
from storage.conversation_memory_store import ConversationMemoryStore


def get_tool_store(config):
    tool_cfg = config.get("tool_bus", {}) or {}
    return ToolBusStore(
        registry_file=tool_cfg.get("registry_file", "memory_runtime/tool_registry.json"),
        runtime_file=tool_cfg.get("runtime_file", "memory_runtime/tool_calls.json"),
    )


def get_task_store(config):
    task_cfg = config.get("task_registry", {}) or {}
    return TaskStore(
        task_file=task_cfg.get("task_file", "memory_runtime/task_registry.json")
    )


def get_fact_store(config):
    fact_cfg = config.get("fact_memory", {}) or {}
    return FactMemoryStore(
        fact_file=fact_cfg.get("fact_file", "memory_runtime/fact_memory.json")
    )


def get_long_memory_store(config):
    mem_cfg = config.get("long_memory", {}) or {}
    return ConversationMemoryStore(
        memory_file=mem_cfg.get("conversation_memory_file", "memory_runtime/conversation_memory.json")
    )


def ensure_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def execute_task_registry(config, call):
    args = ensure_dict(call.get("args", {}))
    action = str(args.get("action", "list") or "list").strip().lower()
    store = get_task_store(config)

    if action == "list":
        status = str(args.get("status", "pending") or "pending")
        return store.list_tasks(status=status, limit=int(args.get("limit", 30)))

    if action == "add":
        title = str(args.get("title", "") or "").strip()
        description = str(args.get("description", "") or "").strip()
        priority = str(args.get("priority", "normal") or "normal").strip()
        tags = args.get("tags", ["tool"])
        if not title:
            raise ValueError("task_registry.add 缺少 title")
        item = store.add_task(
            title=title,
            description=description,
            priority=priority,
            tags=tags,
            source_text=json.dumps(args, ensure_ascii=False),
            trace_id=call.get("trace_id", ""),
            created_by="tool_executor",
        )
        return f"已添加任务 #{item.get('id')}：{item.get('title')}"

    if action == "done":
        task_id = args.get("task_id")
        if not task_id:
            raise ValueError("task_registry.done 缺少 task_id")
        item = store.update_status(task_id, "done", notes=str(args.get("notes", "")))
        if not item:
            raise ValueError(f"没有找到任务 #{task_id}")
        return f"任务 #{item.get('id')} 已完成：{item.get('title')}"

    if action == "cancel":
        task_id = args.get("task_id")
        if not task_id:
            raise ValueError("task_registry.cancel 缺少 task_id")
        item = store.update_status(task_id, "cancelled", notes=str(args.get("reason", "")))
        if not item:
            raise ValueError(f"没有找到任务 #{task_id}")
        return f"任务 #{item.get('id')} 已取消：{item.get('title')}"

    raise ValueError(f"未知 task_registry action：{action}")


def execute_memory_write(config, call):
    args = ensure_dict(call.get("args", {}))
    memory_type = str(args.get("memory_type", "fact") or "fact").strip().lower()
    content = str(args.get("content", "") or "").strip()

    if not content:
        raise ValueError("memory_write 缺少 content")

    if memory_type == "fact":
        store = get_fact_store(config)
        item = store.add_fact(
            content=content,
            subject=str(args.get("subject", "") or ""),
            tags=args.get("tags", ["tool"]),
            evidence_event_id=args.get("evidence_event_id"),
            evidence_quote=str(args.get("evidence_quote", "") or ""),
            confidence=float(args.get("confidence", 0.8)),
            importance=float(args.get("importance", 0.6)),
            source="tool_executor_memory_write",
            notes=str(args.get("notes", "") or ""),
        )
        return f"已写入事实记忆 #{item.get('id')}：{item.get('content')}"

    if memory_type in ["long", "conversation"]:
        store = get_long_memory_store(config)
        item = store.add_memory(
            content=content,
            category=str(args.get("category", "personal") or "personal"),
            subject=str(args.get("subject", "") or ""),
            tags=args.get("tags", "tool"),
            source_user_id=str(args.get("source_user_id", "") or ""),
            source_name="tool_executor",
        )
        return f"已写入长期记忆 #{item.get('id')}：{item.get('content')}"

    raise ValueError("memory_write 目前只支持 memory_type=fact 或 long")


def execute_internal_tool(config, call):
    tool_name = str(call.get("tool_name", "") or "").strip()

    if tool_name == "task_registry":
        return execute_task_registry(config, call)

    if tool_name == "memory_write":
        return execute_memory_write(config, call)

    if tool_name in ["local_shell", "browser"]:
        raise PermissionError(f"{tool_name} 是外部/高风险工具，v8.3 不执行。")

    raise ValueError(f"未知或未实现工具：{tool_name}")


def execute_tool_call(config, call_id):
    store = get_tool_store(config)
    call = store.get_call(call_id)

    if not call:
        return {
            "ok": False,
            "message": f"没有找到工具请求 #{call_id}。"
        }

    if call.get("status") != "approved":
        return {
            "ok": False,
            "message": f"工具请求 #{call_id} 当前状态是 {call.get('status')}，必须先 approved 才能执行。"
        }

    try:
        result = execute_internal_tool(config, call)
        item = store.update_call(call_id, "done", result=str(result))
        return {
            "ok": True,
            "message": f"工具请求 #{call_id} 已执行完成：{result}",
            "call": item,
        }
    except Exception as e:
        item = store.update_call(call_id, "failed", result=f"{type(e).__name__}: {e}")
        return {
            "ok": False,
            "message": f"工具请求 #{call_id} 执行失败：{type(e).__name__}: {e}",
            "call": item,
        }


def execute_approved_calls(config, limit=5):
    store = get_tool_store(config)
    calls = store.retrieve_open_calls(limit=100)
    approved = [c for c in calls if c.get("status") == "approved"][:int(limit)]

    if not approved:
        return {
            "ok": True,
            "message": "暂无 approved 工具请求。",
            "results": [],
        }

    results = []
    for call in approved:
        results.append(execute_tool_call(config, call.get("id")))

    lines = ["【工具执行结果】"]
    for r in results:
        lines.append(r.get("message", ""))

    return {
        "ok": all(r.get("ok") for r in results),
        "message": "\n".join(lines),
        "results": results,
    }
