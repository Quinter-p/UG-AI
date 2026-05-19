def build_tool_bus_fragment(tool_bus_text):
    text = str(tool_bus_text or "").strip()

    if not text:
        text = "暂无待处理工具请求。"

    return (
        text
        + "\n\n以上是工具总线状态。"
        + "当前阶段只有 task_registry 和 memory_write 可以在主人批准后执行。"
        + "approved 表示主人允许执行，但不等于已经完成。"
        + "local_shell 与 browser 仍不可执行。"
        + "不要声称已经在后台执行；不要承诺稍后完成。"
    )
