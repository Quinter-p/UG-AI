def build_speak_decision_fragment(decision):
    decision = decision or {}

    return (
        f"是否应该发言：{decision.get('should_reply')}\n"
        f"发言原因：{decision.get('reason', '')}\n"
        f"主动性模式：{decision.get('mode', 'normal')}\n"
        f"优先级：{decision.get('priority', 'normal')}\n"
        f"是否因冷却限制：{decision.get('cooldown_applied', False)}\n"
        "这是内部发言决策。不要直接向用户解释这些字段。"
    )
