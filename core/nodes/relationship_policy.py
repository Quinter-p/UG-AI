def relationship_policy_node(state):
    """
    把关系状态转成可执行的行为策略。

    v3: relationship 只是 prompt 信息
    v3.1: relationship_policy 参与路由和回复预算
    """
    rel = state.get("relationship_state") or {}
    is_master = bool(state.get("is_master", False))

    role = str(rel.get("role", "unknown") or "unknown").lower()
    attitude = str(rel.get("attitude", "neutral") or "neutral").lower()

    try:
        trust = int(rel.get("trust", 30))
    except Exception:
        trust = 30

    try:
        affection = int(rel.get("affection", 30))
    except Exception:
        affection = 30

    policy = {
        "stance": "neutral",
        "reason": "default",
        "allow_owner_info": False,
        "allow_internal_info": False,
        "allow_warm_tone": False,
        "allow_call_master": False,
        "tone_hint": "礼貌、中立、有边界",
        "max_chars_override": None,
        "max_paragraphs_override": None,
        "force_reply_mode": None,
        "blocked": False,
    }

    # 主人优先级最高
    if is_master or role == "master":
        policy.update({
            "stance": "loyal",
            "reason": "master",
            "allow_owner_info": True,
            "allow_internal_info": True,
            "allow_warm_tone": True,
            "allow_call_master": True,
            "tone_hint": "亲近、自然、尊重，可以称呼主人，但不要过度撒娇",
        })
        return {"relationship_policy": policy}

    # blocked：直接忽略普通消息和命令
    if role == "blocked" or attitude in ["blocked", "avoid"]:
        policy.update({
            "stance": "blocked",
            "reason": "blocked_or_avoid",
            "blocked": True,
            "force_reply_mode": "ignore",
        })
        return {
            "relationship_policy": policy,
            "route": "ignore",
        }

    # 敌人 / 敌对态度：短、冷、警觉
    if role in ["enemy", "hostile"] or attitude in ["hostile", "enemy"]:
        policy.update({
            "stance": "hostile",
            "reason": "enemy_or_hostile",
            "allow_owner_info": False,
            "allow_internal_info": False,
            "allow_warm_tone": False,
            "allow_call_master": False,
            "tone_hint": "警觉、冷淡、短句，不透露主人信息，不亲近，不解释内部设定",
            "max_chars_override": 90,
            "max_paragraphs_override": 1,
            "force_reply_mode": "hostile",
        })
        return {"relationship_policy": policy}

    # 低信任：谨慎
    if trust < 25:
        policy.update({
            "stance": "cautious",
            "reason": "low_trust",
            "allow_owner_info": False,
            "allow_internal_info": False,
            "allow_warm_tone": False,
            "allow_call_master": False,
            "tone_hint": "谨慎、简短、礼貌，不透露主人信息或内部设定",
            "max_chars_override": 120,
            "max_paragraphs_override": 1,
            "force_reply_mode": "cautious",
        })
        return {"relationship_policy": policy}

    # 熟人/朋友
    if role in ["known_user", "friend"] or attitude in ["friendly", "friend"]:
        policy.update({
            "stance": "friendly",
            "reason": "known_or_friendly",
            "allow_owner_info": False,
            "allow_internal_info": trust >= 60,
            "allow_warm_tone": affection >= 40,
            "allow_call_master": False,
            "tone_hint": "友好自然，但有边界；不要称呼对方为主人",
        })
        return {"relationship_policy": policy}

    # 默认陌生人
    if role in ["stranger", "unknown"] or attitude in ["neutral", "cautious"]:
        policy.update({
            "stance": "neutral",
            "reason": "stranger_or_neutral",
            "allow_owner_info": False,
            "allow_internal_info": False,
            "allow_warm_tone": False,
            "allow_call_master": False,
            "tone_hint": "礼貌、中立、简短，保持边界感",
            "max_chars_override": 160,
            "max_paragraphs_override": 1,
        })
        return {"relationship_policy": policy}

    return {"relationship_policy": policy}
