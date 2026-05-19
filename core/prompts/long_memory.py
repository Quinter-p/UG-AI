def build_world_lore_fragment(world_lore_text):
    text = str(world_lore_text or "").strip()

    if not text:
        text = "暂无世界观设定。"

    return (
        text
        + "\n\n以上是主人编写的世界观/角色/规则设定，优先级高于普通对话记忆。"
        + "相关时自然使用，不要逐条背诵。"
    )


def build_conversation_memory_fragment(conversation_memory_text):
    text = str(conversation_memory_text or "").strip()

    if not text:
        text = "暂无长期个人记忆。"

    return (
        text
        + "\n\n以上是你和主人长期互动沉淀出的稳定记忆。"
        + "只在相关时自然参考，不要把记忆条目逐条念给用户。"
    )
