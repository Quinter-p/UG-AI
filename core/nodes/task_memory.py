from storage.task_store import TaskStore


def task_memory_load_node(state):
    config = state.get("config") or {}
    task_cfg = config.get("task_registry", {}) or {}

    if not task_cfg.get("enabled", True):
        return {
            "task_items": [],
            "task_memory_text": "任务登记表未启用。",
        }

    store = TaskStore(
        task_file=task_cfg.get("task_file", "memory_runtime/task_registry.json")
    )

    rel = state.get("relationship_state") or {}
    identity = state.get("identity") or {}

    query = " ".join([
        str(state.get("clean_text") or state.get("raw_message") or ""),
        str(identity.get("name", "")),
        str(identity.get("role", "")),
        str(rel.get("name", "")),
        str(rel.get("role", "")),
        " ".join(rel.get("lore_keys", []) or []),
    ])

    items = store.retrieve(query=query, limit=int(task_cfg.get("prompt_task_limit", 5)))

    return {
        "task_items": items,
        "task_memory_text": store.format_for_prompt(items),
    }
