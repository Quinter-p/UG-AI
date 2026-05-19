from storage.fact_memory_store import FactMemoryStore


def fact_memory_load_node(state):
    config = state.get("config") or {}
    fact_cfg = config.get("fact_memory", {}) or {}

    if not fact_cfg.get("enabled", True):
        return {
            "fact_memory_items": [],
            "fact_memory_text": "事实记忆未启用。",
        }

    store = FactMemoryStore(
        fact_file=fact_cfg.get("fact_file", "memory_runtime/fact_memory.json")
    )

    rel = state.get("relationship_state") or {}
    identity = state.get("identity") or {}

    query = " ".join([
        str(state.get("clean_text") or state.get("raw_message") or ""),
        str(identity.get("name", "")),
        str(identity.get("role", "")),
        str(rel.get("name", "")),
        str(rel.get("role", "")),
        str(rel.get("attitude", "")),
        " ".join(rel.get("lore_keys", []) or []),
    ])

    items = store.retrieve(query=query, limit=int(fact_cfg.get("prompt_fact_limit", 6)))

    return {
        "fact_memory_items": items,
        "fact_memory_text": store.format_for_prompt(items),
    }
