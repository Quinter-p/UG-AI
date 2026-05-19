def build_relationship_fragment(relationship):
    relationship = relationship or {}
    return (
        f"关系名称：{relationship.get('name', '未知用户')}\n"
        f"关系角色：{relationship.get('role', 'unknown')}\n"
        f"当前态度：{relationship.get('attitude', 'neutral')}\n"
        f"亲近度：{relationship.get('affection', 30)}/100\n"
        f"信任度：{relationship.get('trust', 30)}/100\n"
        f"熟悉度：{relationship.get('familiarity', 10)}/100\n"
        f"关系备注：{relationship.get('notes', '')}\n"
        "关系状态只影响称呼、边界感和语气。不要把这些数值直接说出来。"
    )
