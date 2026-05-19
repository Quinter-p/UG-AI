import re

def match_rule(text, message_type, rule):
    if not rule.get("enabled", True): return None
    if message_type == "group" and not rule.get("bypass_group_wakeup", False): return None
    raw = str(text or "").strip()
    if not raw: return None
    if rule.get("ignore_commands", True) and raw.startswith("/"): return None
    reply = str(rule.get("reply", "") or "").strip()
    if not reply: return None
    triggers = rule.get("triggers", []) or []
    if isinstance(triggers, str): triggers = [triggers]
    mode = str(rule.get("match_mode", "contains")).lower(); case_sensitive = bool(rule.get("case_sensitive", False))
    raw_match = raw if case_sensitive else raw.lower()
    for trigger in triggers:
        trigger = str(trigger or "")
        if not trigger: continue
        trig = trigger if case_sensitive else trigger.lower()
        if mode == "exact" and raw_match == trig: return reply
        if mode == "word" and re.search(r"(?<![a-zA-Z0-9_])" + re.escape(trig) + r"(?![a-zA-Z0-9_])", raw_match): return reply
        if mode == "regex":
            if re.search(trigger, raw, flags=(0 if case_sensitive else re.IGNORECASE)): return reply
        if mode not in ["exact", "word", "regex"] and trig in raw_match: return reply
    return None

def reflex_node(state):
    config = state.get("config") or {}; auto_cfg = config.get("auto_reactions", {}) or {}
    if not auto_cfg.get("enabled", True): return {}
    rules = auto_cfg.get("rules", []) or []; text = state.get("raw_message") or ""; message_type = state.get("message_type") or "private"
    for rule in rules:
        if not isinstance(rule, dict): continue
        reply = match_rule(text, message_type, rule)
        if reply: return {"route": "auto_reply", "final_reply": reply}
    return {}
