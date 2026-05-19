from storage.session_store import SessionHistoryStore, build_session_key


def history_load_node(state):
    config = state.get("config") or {}
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

    session = store.get_session(session_key)

    return {
        "session_key": session_key,
        "history_turns": session.get("turns", [])[-int(memory_cfg.get("max_turns", 6)):],
        "rolling_summary": session.get("rolling_summary", ""),
        "last_user_message": session.get("last_user_message", ""),
    }


def history_save_node(state):
    config = state.get("config") or {}
    memory_cfg = config.get("short_memory", {}) or {}

    store = SessionHistoryStore(
        history_file=memory_cfg.get("history_file", "memory_runtime/session_history.json"),
        max_turns=int(memory_cfg.get("max_turns", 6)),
    )

    session_key = state.get("session_key") or build_session_key(
        message_type=state.get("message_type", "private"),
        user_id=state.get("user_id", ""),
        group_id=state.get("group_id"),
    )

    store.append_turn(
        session_key=session_key,
        user_text=state.get("clean_text") or state.get("raw_message") or "",
        assistant_summary=state.get("assistant_short_summary") or state.get("final_reply") or "",
    )

    return {}
