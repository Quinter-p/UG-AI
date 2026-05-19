def build_expression_style_fragment(style):
    style = style or {}

    do_items = style.get("do", []) or []
    dont_items = style.get("dont", []) or []

    lines = [
        f"语气：{style.get('tone', '自然、平稳')}",
        f"温度：{style.get('warmth', 'medium')}",
        f"关系距离：{style.get('distance', 'normal')}",
        f"能量水平：{style.get('energy_level', 'normal')}",
        f"角色表达强度：{style.get('roleplay_level', 'light')}",
        f"口癖强度：{style.get('catness', 'light')}",
        f"称呼策略：{style.get('addressing', '自然称呼')}",
        f"提问策略：{style.get('question_policy', '少反问')}",
        f"动作描写策略：{style.get('stage_direction_policy', '默认不用')}",
        f"每条回复最多称呼主人次数：{style.get('max_master_address_per_reply', 1)}",
        f"每条回复最多使用“喵/本喵”等口癖次数：{style.get('max_cat_words_per_reply', 1)}",
        f"安全边界：{style.get('safety_boundary', '')}",
    ]

    if do_items:
        lines.append("应该：")
        for item in do_items:
            lines.append(f"- {item}")

    if dont_items:
        lines.append("不要：")
        for item in dont_items:
            lines.append(f"- {item}")

    lines.append("这些是表达风格约束，不要直接念给用户。")
    return "\n".join(lines)
