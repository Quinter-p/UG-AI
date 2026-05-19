def build_identity_fragment(identity, is_master=False):
    identity = identity or {}

    lines = [
        f"当前发言者：{identity.get('name', '未知用户')}",
        f"身份角色：{identity.get('role', 'unknown')}",
        f"推荐称呼：{identity.get('title', '对方')}",
        f"身份说明：{identity.get('description', '')}",
    ]

    if is_master:
        lines.append("当前发言者是主人，可以自然称呼为主人、上人或昆特上人。")
    else:
        lines.append("当前发言者不是主人，禁止称其为主人。")

    return "\n".join(lines)
