def clip(text, limit):
    text = str(text or "").strip().replace("\n", " ")
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def build_memory_fragment(rolling_summary, turns, last_user_message):
    blocks = []

    if rolling_summary:
        blocks.append("更早对话摘要：\n" + clip(rolling_summary, 700))

    if turns:
        lines = []
        for turn in turns[-6:]:
            user = clip(turn.get("user", ""), 140)
            assistant = clip(turn.get("assistant_summary", ""), 90)

            if user:
                lines.append(f"用户：{user}")
            if assistant:
                lines.append(f"UGAI：{assistant}")

        if lines:
            blocks.append("最近对话提要：\n" + "\n".join(lines))

    if last_user_message:
        blocks.append(
            "上一条用户消息："
            + clip(last_user_message, 160)
        )

    if not blocks:
        return "暂无历史。"

    return "\n\n".join(blocks)
