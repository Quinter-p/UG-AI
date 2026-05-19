from storage.emotion_store import EmotionStore

def emotion_node(state):
    config = state.get("config") or {}; emotion_cfg = config.get("emotion", {})
    store = EmotionStore(emotion_cfg.get("state_file", "memory_runtime/emotion_state.json"))
    data = store.update_by_message(text=state.get("clean_text") or state.get("raw_message") or "", is_master=bool(state.get("is_master", False)))
    return {"emotion_state": data}
