import re
CQ_AT_PATTERN = re.compile(r"\[CQ:at,qq=(\d+)\]")
def wakeup_node(state):
    config = state.get("config") or {}; qq_cfg = config.get("qq", {})
    message_type = state.get("message_type"); user_id = str(state.get("user_id") or ""); group_id = state.get("group_id"); self_id = str(state.get("self_id") or "")
    raw_message = state.get("raw_message") or ""; clean_text = state.get("clean_text") or ""
    if message_type == "private":
        if not qq_cfg.get("enable_private", True): return {"route": "ignore", "should_reply": False}
        whitelist = qq_cfg.get("private_whitelist", []) or []
        if whitelist and int(user_id) not in [int(x) for x in whitelist]: return {"route": "ignore", "should_reply": False}
        return {"should_reply": True, "route": "continue"}
    if message_type == "group":
        if not qq_cfg.get("enable_group", True): return {"route": "ignore", "should_reply": False}
        group_whitelist = qq_cfg.get("group_whitelist", []) or []
        if group_whitelist and int(group_id) not in [int(x) for x in group_whitelist]: return {"route": "ignore", "should_reply": False}
        mentioned = False
        if self_id:
            for qq in CQ_AT_PATTERN.findall(raw_message):
                if str(qq) == str(self_id): mentioned = True; break
        prefixes = qq_cfg.get("group_prefixes", []) or []; matched_prefix = False; final_text = clean_text
        for prefix in prefixes:
            prefix = str(prefix)
            if final_text.startswith(prefix):
                final_text = final_text[len(prefix):].strip(" ：:，,。"); matched_prefix = True; break
        require_prefix = bool(qq_cfg.get("require_prefix_in_group", True))
        if require_prefix and not mentioned and not matched_prefix: return {"route": "ignore", "should_reply": False}
        return {"clean_text": final_text, "should_reply": bool(final_text), "route": "continue" if final_text else "ignore"}
    return {"route": "ignore", "should_reply": False}
