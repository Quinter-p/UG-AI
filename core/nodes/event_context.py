import hashlib
from datetime import datetime
from storage.event_dedup_store import EventDedupStore


def now_compact():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def short_hash(text, n=10):
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:n]


def get_raw_message(raw_event):
    return (
        raw_event.get("raw_message")
        or raw_event.get("message")
        or raw_event.get("text")
        or ""
    )


def get_source_event_id(raw_event):
    for key in ["message_id", "id", "event_id", "message_seq"]:
        value = raw_event.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def build_message_key(state):
    raw_event = state.get("raw_event") or {}
    message_type = str(state.get("message_type") or raw_event.get("message_type") or "")
    group_id = str(state.get("group_id") or raw_event.get("group_id") or "")
    user_id = str(state.get("user_id") or raw_event.get("user_id") or "")
    source_event_id = get_source_event_id(raw_event)
    text = str(state.get("raw_message") or get_raw_message(raw_event) or "")

    if source_event_id:
        body = f"id:{source_event_id}"
    else:
        t = raw_event.get("time", "")
        body = f"hash:{short_hash(str(t) + '|' + message_type + '|' + group_id + '|' + user_id + '|' + text, 16)}"

    return f"{message_type}:{group_id}:{user_id}:{body}"


def event_context_node(state):
    """
    v7.2 统一事件上下文。

    给每条进入 LangGraph 的消息建立：
    - trace_id：本轮处理链路 ID
    - message_key：去重 key
    - source_event_id：OneBot 原始 message_id 等
    """
    config = state.get("config") or {}
    event_cfg = config.get("event_model", {}) or {}

    raw_event = state.get("raw_event") or {}
    message_key = build_message_key(state)
    source_event_id = get_source_event_id(raw_event)
    trace_id = f"tr_{now_compact()}_{short_hash(message_key, 8)}"

    ctx = {
        "trace_id": trace_id,
        "source_event_id": source_event_id,
        "message_key": message_key,
        "message_type": state.get("message_type", ""),
        "group_id": state.get("group_id", ""),
        "user_id": state.get("user_id", ""),
        "duplicate": False,
        "dedup_enabled": bool(event_cfg.get("dedup_enabled", True)),
    }

    store = EventDedupStore(
        runtime_file=event_cfg.get("runtime_file", "memory_runtime/event_runtime.json")
    )

    if ctx["dedup_enabled"]:
        if store.is_seen(message_key):
            ctx["duplicate"] = True
            store.add_trace(ctx)
            return {
                "route": "ignore",
                "should_reply": False,
                "event_context": ctx,
            }

        store.mark_seen(message_key)

    store.add_trace(ctx)

    return {
        "event_context": ctx,
    }
