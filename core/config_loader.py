# core/config_loader.py

import json
import os
from copy import deepcopy


CONFIG_FILE = "config.json"


DEFAULT_CONFIG = {
    "model": {
        "chat_model": "qwen2.5:7b"
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "keep_alive": "10m"
    },
    "generation": {
        "temperature": 0.45,
        "top_p": 0.8,
        "num_ctx": 4096,
        "num_predict": 512
    },
    "paths": {
        "knowledge_dir": "knowledge",
        "log_dir": "logs",
        "memory_file": "memory.json",
        "memory_notes_file": "memory_notes.txt"
    },
    "limits": {
        "max_history_chars": 5000,
        "max_read_chars": 20000
    },
    "memory": {
        "enable_auto_memory": False
    },
    "history": {
        "enable_persistent_history": True,
        "history_file": "memory/chat_history.json",
        "max_saved_turns": 50,
        "load_recent_turns": 12
    },
    "rag": {
        "enable_auto_rag": True,
        "embedding_model": "nomic-embed-text",
        "index_file": "memory/rag_index.json",
        "top_k": 4,
        "min_score": 0.25,
        "chunk_size": 700,
        "chunk_overlap": 120
    },
    "debug": {
        "debug_prompt": False,
        "debug_tool_router": False
    },
    "style": {
        "use_style_examples": True
    }
}


def deep_merge(default, user):
    result = deepcopy(default)

    for key, value in user.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def ensure_config_file(config_path=CONFIG_FILE):
    if os.path.exists(config_path):
        return

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)


def load_config(config_path=CONFIG_FILE):
    ensure_config_file(config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
    except Exception:
        user_config = {}

    return deep_merge(DEFAULT_CONFIG, user_config)


def pretty_config(config):
    return json.dumps(config, ensure_ascii=False, indent=2)
