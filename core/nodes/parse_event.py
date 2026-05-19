import re
CQ_AT_PATTERN = re.compile(r"\[CQ:at,qq=(\d+)\]")
CQ_CODE_PATTERN = re.compile(r"\[CQ:[^\]]+\]")

def parse_event_node(state):
    event = state.get("raw_event") or {}; config = state.get("config") or {}; qq_cfg = config.get("qq", {})
    sender = event.get("sender") or {}
    message_type = event.get("message_type") or ""
    user_id = str(event.get("user_id") or sender.get("user_id") or "")
    group_id = event.get("group_id"); group_id = str(group_id) if group_id is not None else None
    self_id = str(event.get("self_id") or event.get("_self_id") or "")
    raw_message = str(event.get("raw_message") or event.get("message") or "").strip()
    max_len = int(qq_cfg.get("max_message_chars", 1000))
    if max_len and len(raw_message) > max_len: raw_message = raw_message[:max_len]
    clean_text = CQ_AT_PATTERN.sub("", raw_message); clean_text = CQ_CODE_PATTERN.sub("", clean_text).strip()
    return {"message_type": message_type, "user_id": user_id, "group_id": group_id, "self_id": self_id, "raw_message": raw_message, "clean_text": clean_text, "should_reply": False, "route": "continue"}
