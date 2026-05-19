def contains_any(text, words):
    return any(w in text for w in words)


def apply_relationship_budget(policy, relationship_policy):
    relationship_policy = relationship_policy or {}

    override_chars = relationship_policy.get("max_chars_override")
    override_paragraphs = relationship_policy.get("max_paragraphs_override")
    force_mode = relationship_policy.get("force_reply_mode")

    if force_mode in ["hostile", "cautious"]:
        policy["mode"] = force_mode
        policy["reason"] = "relationship_" + str(force_mode)
        policy["allow_question"] = False

    if override_chars is not None:
        try:
            policy["max_chars"] = min(int(policy.get("max_chars", 999)), int(override_chars))
        except Exception:
            policy["max_chars"] = int(override_chars)

    if override_paragraphs is not None:
        try:
            policy["max_paragraphs"] = min(int(policy.get("max_paragraphs", 99)), int(override_paragraphs))
        except Exception:
            policy["max_paragraphs"] = int(override_paragraphs)

    policy["relationship_stance"] = relationship_policy.get("stance", "neutral")
    policy["relationship_tone_hint"] = relationship_policy.get("tone_hint", "")
    policy["allow_owner_info"] = bool(relationship_policy.get("allow_owner_info", False))
    policy["allow_internal_info"] = bool(relationship_policy.get("allow_internal_info", False))
    policy["allow_call_master"] = bool(relationship_policy.get("allow_call_master", False))

    return policy


def reply_policy_node(state):
    config = state.get("config") or {}
    policy_cfg = config.get("reply_policy", {}) or {}
    relationship_policy = state.get("relationship_policy") or {}

    text = str(state.get("clean_text") or state.get("raw_message") or "").strip()
    emotion = state.get("emotion_state") or {}

    detailed_words = policy_cfg.get("detailed_words") or [
        "详细", "展开", "解释", "为什么", "怎么做", "步骤", "方案", "分析", "总结",
        "代码", "脚本", "报错", "修复", "设计", "架构", "文档", "报告", "论文",
        "帮我写", "写一份", "完整", "系统地", "讲讲", "推导", "证明"
    ]

    comfort_words = policy_cfg.get("comfort_words") or [
        "累", "难受", "烦", "emo", "焦虑", "压力", "不开心", "崩溃", "困", "失眠",
        "害怕", "难过", "委屈"
    ]

    memory_words = policy_cfg.get("memory_words") or [
        "刚刚", "刚才", "之前", "记得", "还记得", "我说了什么", "我们聊到哪"
    ]

    short_words = policy_cfg.get("short_words") or [
        "嗯", "哦", "好", "ok", "OK", "行", "可以", "继续", "然后呢", "在吗", "你好"
    ]

    mood = emotion.get("mood", "calm")

    policy = {
        "mode": "casual",
        "reason": "default_casual",
        "max_chars": int(policy_cfg.get("casual_max_chars", 180)),
        "max_paragraphs": int(policy_cfg.get("casual_max_paragraphs", 1)),
        "target_sentences": policy_cfg.get("casual_sentences", "1-2句"),
        "allow_question": bool(policy_cfg.get("casual_allow_question", False)),
        "allow_stage_direction": False,
    }

    # 敌对/低信任关系不允许 detailed 展开优先级太高。
    rel_stance = relationship_policy.get("stance", "neutral")
    if rel_stance in ["hostile", "cautious"]:
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    if contains_any(text, memory_words):
        policy.update({
            "mode": "memory_answer",
            "reason": "memory_query",
            "max_chars": int(policy_cfg.get("memory_max_chars", 160)),
            "max_paragraphs": 1,
            "target_sentences": "1-2句",
            "allow_question": False,
        })
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    if contains_any(text, detailed_words):
        policy.update({
            "mode": "detailed",
            "reason": "detailed_request",
            "max_chars": int(policy_cfg.get("detailed_max_chars", 900)),
            "max_paragraphs": int(policy_cfg.get("detailed_max_paragraphs", 6)),
            "target_sentences": "按需要展开，但结构清楚",
            "allow_question": True,
            "allow_stage_direction": True,
        })
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    if contains_any(text, comfort_words) or mood in ["hurt", "warm"]:
        policy.update({
            "mode": "comfort",
            "reason": "comfort_or_warm_mood",
            "max_chars": int(policy_cfg.get("comfort_max_chars", 240)),
            "max_paragraphs": int(policy_cfg.get("comfort_max_paragraphs", 2)),
            "target_sentences": "2-3句",
            "allow_question": True,
        })
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    if text in short_words or len(text) <= 4:
        policy.update({
            "mode": "short_chat",
            "reason": "short_input",
            "max_chars": int(policy_cfg.get("short_max_chars", 120)),
            "max_paragraphs": 1,
            "target_sentences": "1句",
            "allow_question": False,
        })
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    if mood == "alert":
        policy.update({
            "mode": "alert",
            "reason": "alert_mood",
            "max_chars": int(policy_cfg.get("alert_max_chars", 160)),
            "max_paragraphs": 1,
            "target_sentences": "1-2句",
            "allow_question": False,
        })
        policy = apply_relationship_budget(policy, relationship_policy)
        return {"reply_policy": policy}

    policy = apply_relationship_budget(policy, relationship_policy)
    return {"reply_policy": policy}
