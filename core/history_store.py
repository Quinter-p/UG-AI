# core/history_store.py

import json
import os
from datetime import datetime


DEFAULT_HISTORY_FILE = "memory/chat_history.json"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_history_items(history_file=DEFAULT_HISTORY_FILE):
    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict) and isinstance(data.get("turns"), list):
            return data["turns"]

        return []
    except Exception:
        return []


def save_history_items(items, history_file=DEFAULT_HISTORY_FILE, max_saved_turns=50):
    ensure_parent_dir(history_file)

    if max_saved_turns and max_saved_turns > 0:
        items = items[-max_saved_turns:]

    data = {
        "updated_at": now_str(),
        "turns": items
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_history_turn(
    user_text,
    assistant_text,
    history_file=DEFAULT_HISTORY_FILE,
    max_saved_turns=50,
    source="chat"
):
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()

    if not user_text or not assistant_text:
        return

    items = load_history_items(history_file)

    items.append({
        "time": now_str(),
        "source": source,
        "user": user_text,
        "assistant": assistant_text
    })

    save_history_items(
        items,
        history_file=history_file,
        max_saved_turns=max_saved_turns
    )


def clear_history(history_file=DEFAULT_HISTORY_FILE):
    ensure_parent_dir(history_file)

    data = {
        "updated_at": now_str(),
        "turns": []
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_history_text(
    history_file=DEFAULT_HISTORY_FILE,
    load_recent_turns=12,
    max_chars=5000
):
    items = load_history_items(history_file)

    if load_recent_turns and load_recent_turns > 0:
        items = items[-load_recent_turns:]

    lines = []

    for item in items:
        user = item.get("user", "").strip()
        assistant = item.get("assistant", "").strip()

        if not user or not assistant:
            continue

        lines.append(f"用户：{user}")
        lines.append(f"AI：{assistant}")

    text = "\n".join(lines).strip()

    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]

    return text


def history_status(history_file=DEFAULT_HISTORY_FILE):
    items = load_history_items(history_file)

    if not os.path.exists(history_file):
        exists = False
        size_kb = 0
    else:
        exists = True
        size_kb = os.path.getsize(history_file) / 1024

    lines = []
    lines.append("====== 对话历史状态 ======")
    lines.append(f"历史文件：{history_file}")
    lines.append(f"文件存在：{exists}")
    lines.append(f"文件大小：{size_kb:.1f} KB")
    lines.append(f"保存轮数：{len(items)}")

    if items:
        last = items[-1]
        lines.append(f"最后更新时间：{last.get('time', '未知')}")
        lines.append(f"最后来源：{last.get('source', '未知')}")

    lines.append("=========================")

    return "\n".join(lines)


def format_recent_history(history_file=DEFAULT_HISTORY_FILE, recent_turns=10):
    items = load_history_items(history_file)

    if not items:
        return "暂无保存的对话历史。"

    items = items[-recent_turns:]

    lines = []
    lines.append("====== 最近对话历史 ======")

    for i, item in enumerate(items, 1):
        lines.append("")
        lines.append(f"[{i}] {item.get('time', '未知')} | {item.get('source', 'chat')}")
        lines.append(f"用户：{item.get('user', '')}")
        lines.append(f"AI：{item.get('assistant', '')}")

    lines.append("")
    lines.append("=========================")

    return "\n".join(lines)
