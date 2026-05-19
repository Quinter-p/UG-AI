def build_task_memory_fragment(task_memory_text):
    text = str(task_memory_text or "").strip()

    if not text:
        text = "暂无待办任务。"

    return (
        text
        + "\n\n以上是当前待办任务登记表。"
        + "它用于提醒你当前未完成的任务，但不要主动承诺后台完成。"
        + "除非用户询问任务进度或要求处理任务，否则只在相关时轻量参考。"
    )
