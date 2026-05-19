def build_relationship_policy_fragment(policy):
    policy = policy or {}

    return (
        f"关系立场：{policy.get('stance', 'neutral')}\n"
        f"策略原因：{policy.get('reason', '')}\n"
        f"语气倾向：{policy.get('tone_hint', '礼貌、中立')}\n"
        f"可否称呼对方为主人：{policy.get('allow_call_master', False)}\n"
        f"可否透露主人相关信息：{policy.get('allow_owner_info', False)}\n"
        f"可否透露内部设定/系统信息：{policy.get('allow_internal_info', False)}\n"
        "以上是行为约束，不要直接念给用户。"
    )
