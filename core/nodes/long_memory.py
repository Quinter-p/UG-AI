from storage.conversation_memory_store import ConversationMemoryStore
from storage.world_lore_store import WorldLoreStore


def long_memory_load_node(state):
    config = state.get("config") or {}
    mem_cfg = config.get("long_memory", {}) or {}

    if not mem_cfg.get("enabled", True):
        return {
            "world_lore_text": "长期记忆未启用。",
            "world_lore_files": [],
            "conversation_memory_items": [],
            "conversation_memory_text": "长期记忆未启用。",
        }

    relationship = state.get("relationship_state") or {}
    lore_keys = relationship.get("lore_keys", []) or []
    lore_files = relationship.get("lore_files", []) or []

    query = " ".join([
        str(state.get("clean_text") or state.get("raw_message") or ""),
        str((state.get("identity") or {}).get("name", "")),
        str(relationship.get("name", "")),
        str(relationship.get("role", "")),
        " ".join(str(x) for x in lore_keys),
    ])

    lore_store = WorldLoreStore(
        lore_dir=mem_cfg.get("world_lore_dir", "knowledge"),
        max_chars=int(mem_cfg.get("lore_max_chars", 3500)),
    )

    lore_items = lore_store.retrieve(
        query=query,
        limit=int(mem_cfg.get("lore_file_limit", 6)),
        force_files=lore_files,
        force_keys=lore_keys,
    )

    memory_store = ConversationMemoryStore(
        memory_file=mem_cfg.get("conversation_memory_file", "memory_runtime/conversation_memory.json")
    )

    memory_items = memory_store.retrieve(
        query=query,
        limit=int(mem_cfg.get("conversation_memory_limit", 8)),
    )

    return {
        "world_lore_text": lore_store.format_for_prompt(lore_items),
        "world_lore_files": [item.get("filename", "") for item in lore_items],
        "conversation_memory_items": memory_items,
        "conversation_memory_text": memory_store.format_for_prompt(memory_items),
    }
