from storage.relationship_store import RelationshipStore

def relationship_load_node(state):
    config = state.get("config") or {}
    rel_cfg = config.get("relationship", {}) or {}
    store = RelationshipStore(
        relationship_file=rel_cfg.get("relationship_file", "memory_runtime/relationships.json")
    )
    rel = store.get(
        user_id=state.get("user_id", ""),
        identity=state.get("identity") or {},
        is_master=bool(state.get("is_master", False)),
    )
    return {"relationship_state": rel}

def relationship_save_node(state):
    config = state.get("config") or {}
    rel_cfg = config.get("relationship", {}) or {}
    store = RelationshipStore(
        relationship_file=rel_cfg.get("relationship_file", "memory_runtime/relationships.json")
    )
    rel = store.update_after_interaction(
        user_id=state.get("user_id", ""),
        identity=state.get("identity") or {},
        is_master=bool(state.get("is_master", False)),
        user_text=state.get("clean_text") or state.get("raw_message") or "",
        assistant_text=state.get("final_reply") or "",
    )
    return {"relationship_state": rel}
