from adapters.ollama_client import OllamaClient


def expression_node(state):
    config = state.get("config") or {}
    client = OllamaClient(config)

    messages = state.get("prompt_messages") or []

    if not messages:
        # 兼容兜底
        messages = [
            {"role": "system", "content": state.get("prompt") or ""},
            {"role": "user", "content": state.get("clean_text") or ""},
        ]

    reply, usage = client.chat_messages(messages)

    return {
        "route": "llm_reply",
        "llm_output": reply,
        "final_reply": reply,
        "usage_metadata": usage,
    }
