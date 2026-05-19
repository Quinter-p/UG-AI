from storage.tool_bus_store import ToolBusStore


def tool_bus_load_node(state):
    config = state.get("config") or {}
    tool_cfg = config.get("tool_bus", {}) or {}

    if not tool_cfg.get("enabled", True):
        return {
            "tool_call_items": [],
            "tool_bus_text": "工具总线未启用。",
        }

    store = ToolBusStore(
        registry_file=tool_cfg.get("registry_file", "memory_runtime/tool_registry.json"),
        runtime_file=tool_cfg.get("runtime_file", "memory_runtime/tool_calls.json"),
    )

    calls = store.retrieve_open_calls(limit=int(tool_cfg.get("prompt_tool_call_limit", 5)))

    return {
        "tool_call_items": calls,
        "tool_bus_text": store.format_for_prompt(calls),
    }
