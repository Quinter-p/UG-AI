def build_fact_memory_fragment(fact_memory_text):
    text = str(fact_memory_text or "").strip()

    if not text:
        text = "暂无事实记忆。"

    return (
        text
        + "\n\n以上是带证据来源的事实记忆。"
        + "它比临时聊天历史更稳定，但如果与用户当前明确说法冲突，应以当前用户修正为准。"
        + "不要逐字背诵事实编号，除非用户询问记忆来源。"
    )
