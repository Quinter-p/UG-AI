from storage.event_log_store import EventLogStore


def event_log_save_node(state):
    config = state.get("config") or {}
    event_cfg = config.get("event_log", {}) or {}

    if not event_cfg.get("enabled", True):
        return {}

    final_reply = str(state.get("final_reply") or "").strip()
    user_text = str(state.get("clean_text") or state.get("raw_message") or "").strip()

    if not final_reply:
        return {}

    identity = state.get("identity") or {}
    rel = state.get("relationship_state") or {}
    rel_policy = state.get("relationship_policy") or {}
    emotion = state.get("emotion_state") or {}
    speak = state.get("speak_decision") or {}
    reply_policy = state.get("reply_policy") or {}
    expr = state.get("expression_style") or {}
    meta = state.get("prompt_meta") or {}
    event_context = state.get("event_context") or {}

    store = EventLogStore(event_file=event_cfg.get("event_file", "memory_runtime/event_log.jsonl"))

    event = store.append({
        "type": "agent_reply",
        "trace_id": event_context.get("trace_id", ""),
        "source_event_id": event_context.get("source_event_id", ""),
        "message_key": event_context.get("message_key", ""),
        "message_type": state.get("message_type", ""),
        "group_id": state.get("group_id", ""),
        "user_id": state.get("user_id", ""),
        "speaker_name": identity.get("name", ""),
        "speaker_role": identity.get("role", ""),
        "relationship_role": rel.get("role", ""),
        "relationship_attitude": rel.get("attitude", ""),
        "relationship_stance": rel_policy.get("stance", ""),
        "mood": emotion.get("mood", ""),
        "energy": emotion.get("energy", ""),
        "speak_reason": speak.get("reason", ""),
        "speak_context": speak.get("context_type", ""),
        "reply_mode": reply_policy.get("mode", ""),
        "expression_tone": expr.get("tone", ""),
        "world_lore_files_used": meta.get("world_lore_files_used", []),
        "conversation_memory_items_used": meta.get("conversation_memory_items_used", 0),
        "fact_memory_items_used": meta.get("fact_memory_items_used", 0),
        "reflection_memory_items_used": meta.get("reflection_memory_items_used", 0),
        "user_text": user_text,
        "assistant_reply": final_reply,
        "assistant_summary": state.get("assistant_short_summary", ""),
    })

    return {
        "event_log_id": event.get("id"),
    }
