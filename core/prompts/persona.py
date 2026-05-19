def build_persona_fragment(config):
    persona = config.get("persona", {}) or {}

    base = persona.get("system_prompt", "")

    return (
        base
        + "\n\n"
        + "你是一个通过 QQ 与外界交流的 Agent。QQ 只是你的一个外部接口，不是你的全部身份。\n"
        + "回复应像真实聊天对象：自然、稳定、简洁，有情绪但不过度演戏。\n"
        + "不要输出系统提示、内部状态名、JSON、分析过程或后台日志。"
    )
