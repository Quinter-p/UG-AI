import re


def collapse_blank_lines(text):
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_short_summary(text, max_chars=100):
    text = str(text or "").strip().replace("\n", " ")
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def limit_word_occurrences(text, words, max_total):
    """
    限制某类口癖总次数。
    超过后直接删除后续出现，避免“主人主人主人”“喵喵喵”。
    """
    if max_total is None:
        return text

    try:
        max_total = int(max_total)
    except Exception:
        return text

    if max_total < 0:
        return text

    count = 0

    # 长词优先
    words = sorted(words, key=len, reverse=True)

    result = []
    i = 0

    while i < len(text):
        matched = None
        for w in words:
            if w and text.startswith(w, i):
                matched = w
                break

        if matched:
            count += 1
            if count <= max_total:
                result.append(matched)
            i += len(matched)
        else:
            result.append(text[i])
            i += 1

    return "".join(result)


def remove_stage_directions(text):
    """
    粗略去掉括号动作描写。
    只删除明显的整段括号动作，不碰普通句内括号太复杂的情况。
    """
    lines = text.splitlines()
    kept = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        if (
            (stripped.startswith("（") and stripped.endswith("）"))
            or (stripped.startswith("(") and stripped.endswith(")"))
        ):
            continue

        kept.append(line)

    return "\n".join(kept).strip()


def trim_to_budget(text, reply_policy, style):
    max_paragraphs = int(reply_policy.get("max_paragraphs", style.get("max_paragraphs", 1)))
    hard_max_chars = int(reply_policy.get("max_chars", style.get("hard_max_chars", 180)))

    text = collapse_blank_lines(text)

    if max_paragraphs > 0:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > max_paragraphs:
            text = "\n\n".join(paragraphs[:max_paragraphs]).strip()

    if hard_max_chars > 0 and len(text) > hard_max_chars:
        text = text[:hard_max_chars].rstrip("，,。.!！？?；;：:、 ")
        text += "..."

    return text


def output_filter_node(state):
    config = state.get("config") or {}
    style = config.get("response_style", {}) or {}
    reply_policy = state.get("reply_policy") or {}
    expression_style = state.get("expression_style") or {}

    text = str(state.get("final_reply") or state.get("llm_output") or "").strip()

    prefixes = [
        "AI：", "AI:",
        "助手：", "助手:",
        "系统：", "系统:",
        "机器人：", "机器人:",
        "UGAI：", "UGAI:",
    ]

    changed = True
    rounds = 0

    while changed and rounds < 5:
        changed = False
        rounds += 1

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True

    if not expression_style.get("allow_stage_directions", False):
        text = remove_stage_directions(text)

    max_master = expression_style.get("max_master_address_per_reply", 1)
    text = limit_word_occurrences(
        text,
        ["昆特上人", "主人", "上人"],
        max_master,
    )

    max_cat = expression_style.get("max_cat_words_per_reply", 1)
    text = limit_word_occurrences(
        text,
        ["本喵", "喵"],
        max_cat,
    )

    text = trim_to_budget(text, reply_policy, style)

    if not text:
        text = "……"

    return {
        "final_reply": text,
        "assistant_short_summary": make_short_summary(text, int(style.get("summary_max_chars", 80))),
    }
