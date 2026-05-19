from storage.speak_runtime_store import SpeakRuntimeStore


def text_contains_any(text, words):
    text = str(text or "")
    return any(str(w) and str(w) in text for w in words)


def lower_contains_any(text, words):
    text = str(text or "").lower()
    return any(str(w).lower() in text for w in words if str(w))


def strip_wake_prefix(text, wake_prefixes):
    text = str(text or "").strip()

    for prefix in wake_prefixes:
        prefix = str(prefix or "").strip()
        if not prefix:
            continue

        if text.lower().startswith(prefix.lower()):
            rest = text[len(prefix):].strip()
            # 去掉常见分隔
            rest = rest.lstrip(" ：:，,")
            return rest or text

    return text


def is_at_self(raw_message, self_id):
    raw = str(raw_message or "")
    self_id = str(self_id or "")

    if not self_id:
        return False

    patterns = [
        f"[CQ:at,qq={self_id}]",
        f"[CQ:at,qq={self_id},",
        f"@{self_id}",
    ]

    return any(p in raw for p in patterns)


def make_decision(
    should_reply,
    reason,
    mode,
    priority="normal",
    stripped_text=None,
    cooldown_applied=False,
    cooldown_remaining=0,
):
    data = {
        "should_reply": bool(should_reply),
        "reason": reason,
        "mode": mode,
        "priority": priority,
        "cooldown_applied": bool(cooldown_applied),
        "cooldown_remaining": int(cooldown_remaining or 0),
    }

    if stripped_text is not None:
        data["stripped_text"] = stripped_text

    return data


def speak_decision_node(state):
    """
    v6 主动发言决策。

    目标：
    - 私聊默认回复
    - 群聊不乱插嘴
    - @ / /ugai / 叫她名字时回复
    - 提到主人、施耐德、器灵时可回复
    - 危险词更敏感
    - 群聊自动回复受 cooldown 限制
    """
    config = state.get("config") or {}
    speak_cfg = config.get("speak_decision", {}) or {}

    runtime = SpeakRuntimeStore(
        runtime_file=speak_cfg.get("runtime_file", "memory_runtime/speak_runtime.json")
    )

    runtime_mode = runtime.get_mode()
    default_mode = str(speak_cfg.get("default_mode", "normal") or "normal").lower()
    mode = runtime_mode or default_mode

    if mode not in ["quiet", "normal", "active"]:
        mode = "normal"

    message_type = str(state.get("message_type", "private") or "private")
    user_id = str(state.get("user_id", "") or "")
    group_id = str(state.get("group_id", "") or "")
    self_id = str(state.get("self_id", "") or "")

    raw_message = str(state.get("raw_message", "") or "")
    clean_text = str(state.get("clean_text", "") or "").strip()

    identity = state.get("identity") or {}
    rel = state.get("relationship_state") or {}
    rel_policy = state.get("relationship_policy") or {}
    emotion = state.get("emotion_state") or {}

    wake_prefixes = speak_cfg.get("wake_prefixes") or ["/ugai", "ugai", "UGAI"]
    wake_words = speak_cfg.get("wake_words") or ["施耐德", "器灵", "本喵", "小猫", "猫娘"]
    owner_words = speak_cfg.get("owner_words") or ["主人", "昆特上人", "上人"]
    danger_words = speak_cfg.get("danger_words") or ["黑沐", "敌人", "入侵", "危险", "攻击", "偷袭"]
    self_lore_keys = speak_cfg.get("self_lore_keys") or ["施耐德", "UGAI", "ugai", "器灵"]

    relationship_lore_keys = rel.get("lore_keys", []) or []
    relationship_name = str(rel.get("name", "") or "")
    relationship_role = str(rel.get("role", "") or "")

    # private 默认回复
    if message_type == "private":
        decision = make_decision(
            True,
            "private_message",
            mode,
            priority="direct",
        )
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": True,
            "reason": decision["reason"],
        })
        return {
            "should_reply": True,
            "speak_decision": decision,
        }

    # group
    if message_type != "group":
        decision = make_decision(False, "unknown_message_type", mode)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {
            "route": "ignore",
            "should_reply": False,
            "speak_decision": decision,
        }

    # blocked 在 relationship_policy 已经会拦，但这里再兜底。
    if rel_policy.get("blocked") or rel_policy.get("force_reply_mode") == "ignore":
        decision = make_decision(False, "relationship_blocked", mode)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {
            "route": "ignore",
            "should_reply": False,
            "speak_decision": decision,
        }

    at_self = is_at_self(raw_message, self_id)
    has_wake_prefix = lower_contains_any(clean_text, wake_prefixes) and any(
        clean_text.lower().startswith(str(p).lower()) for p in wake_prefixes
    )
    has_wake_word = text_contains_any(clean_text, wake_words)
    mentions_self = text_contains_any(clean_text, self_lore_keys)
    mentions_owner = text_contains_any(clean_text, owner_words)
    mentions_danger = text_contains_any(clean_text, danger_words)
    mentions_bound_lore = text_contains_any(clean_text, relationship_lore_keys)
    mentions_relationship_name = bool(relationship_name and relationship_name in clean_text)

    # @ 或 /ugai 属于直接唤醒，不受 cooldown。
    if at_self or has_wake_prefix:
        stripped = strip_wake_prefix(clean_text, wake_prefixes)
        decision = make_decision(
            True,
            "direct_wakeup_at_or_prefix",
            mode,
            priority="direct",
            stripped_text=stripped,
        )
        runtime.mark_reply(group_id=group_id, user_id=user_id)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": True,
            "reason": decision["reason"],
        })
        return {
            "should_reply": True,
            "clean_text": stripped,
            "speak_decision": decision,
        }

    # quiet 模式下，只有直接唤醒和危险词让她说话。
    if mode == "quiet":
        if mentions_danger:
            in_cd, remain = runtime.is_group_in_cooldown(group_id)
            if in_cd:
                decision = make_decision(False, "quiet_danger_cooldown", mode, cooldown_applied=True, cooldown_remaining=remain)
                runtime.add_decision({
                    "message_type": message_type,
                    "group_id": group_id,
                    "user_id": user_id,
                    "should_reply": False,
                    "reason": decision["reason"],
                })
                return {"route": "ignore", "should_reply": False, "speak_decision": decision}

            decision = make_decision(True, "quiet_danger_keyword", mode, priority="danger")
            runtime.mark_reply(group_id=group_id, user_id=user_id)
            runtime.add_decision({
                "message_type": message_type,
                "group_id": group_id,
                "user_id": user_id,
                "should_reply": True,
                "reason": decision["reason"],
            })
            return {"should_reply": True, "speak_decision": decision}

        decision = make_decision(False, "quiet_no_direct_wakeup", mode)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {"route": "ignore", "should_reply": False, "speak_decision": decision}

    # normal / active 模式下的候选插话。
    reason = None
    priority = "normal"

    if mentions_danger:
        reason = "danger_keyword"
        priority = "danger"
    elif has_wake_word or mentions_self:
        reason = "mentions_agent"
        priority = "high"
    elif mentions_owner:
        reason = "mentions_owner"
        priority = "high"
    elif mentions_bound_lore and mode == "active":
        reason = "mentions_bound_lore_active"
        priority = "normal"
    elif mentions_relationship_name and mode == "active":
        reason = "mentions_relationship_name_active"
        priority = "normal"
    else:
        reason = ""

    # active 模式额外：熟人发言如果是问句，可以主动回复。
    if not reason and mode == "active":
        if relationship_role in ["master", "known_user", "friend"] and ("?" in clean_text or "？" in clean_text):
            reason = "active_known_user_question"
            priority = "normal"

    if not reason:
        decision = make_decision(False, "group_no_wakeup", mode)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {
            "route": "ignore",
            "should_reply": False,
            "speak_decision": decision,
        }

    # 插话类受冷却限制。危险词可以使用更短冷却，但先用统一 cooldown。
    in_cd, remain = runtime.is_group_in_cooldown(group_id)
    if in_cd and priority != "danger":
        decision = make_decision(
            False,
            reason + "_cooldown",
            mode,
            priority=priority,
            cooldown_applied=True,
            cooldown_remaining=remain,
        )
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {
            "route": "ignore",
            "should_reply": False,
            "speak_decision": decision,
        }

    # 精神太低时，非危险、非直接唤醒少插话。
    try:
        energy = int(emotion.get("energy", 70))
    except Exception:
        energy = 70

    if energy < 20 and priority not in ["danger", "direct"]:
        decision = make_decision(False, reason + "_low_energy", mode, priority=priority)
        runtime.add_decision({
            "message_type": message_type,
            "group_id": group_id,
            "user_id": user_id,
            "should_reply": False,
            "reason": decision["reason"],
        })
        return {
            "route": "ignore",
            "should_reply": False,
            "speak_decision": decision,
        }

    decision = make_decision(True, reason, mode, priority=priority)
    runtime.mark_reply(group_id=group_id, user_id=user_id)
    runtime.add_decision({
        "message_type": message_type,
        "group_id": group_id,
        "user_id": user_id,
        "should_reply": True,
        "reason": decision["reason"],
    })

    return {
        "should_reply": True,
        "speak_decision": decision,
    }
