# core/intent_router.py

import re


def normalize_text(text: str) -> str:
    return text.strip()


def extract_quoted_text(text: str):
    """
    提取引号中的内容。
    支持：
    "xxx"
    “xxx”
    'xxx'
    """
    patterns = [
        r"“(.+?)”",
        r"\"(.+?)\"",
        r"'(.+?)'",
        r"‘(.+?)’"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None


def extract_file_like_path(text: str):
    """
    尽量从自然语言中提取文件路径。
    不做系统命令，不做通配符执行，只返回文本路径。
    """
    quoted = extract_quoted_text(text)
    if quoted:
        return quoted

    # Windows 绝对路径，如 D:\xxx\version0.04.py
    match = re.search(r"[A-Za-z]:\\[^\s，。；;]+", text)
    if match:
        return match.group(0).strip()

    # 相对路径，如 ./version0.04.py ../x/a.md tools/file_tools.py
    match = re.search(r"(\.{1,2}[\\/][^\s，。；;]+)", text)
    if match:
        return match.group(1).strip()

    # 常见文件名，如 version0.04.py config.json README.md
    match = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fff./\\]+?\.(?:py|md|txt|json|csv|log|lsf|m|html|css|js))",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    return None


def extract_folder_path(text: str):
    quoted = extract_quoted_text(text)
    if quoted:
        return quoted

    if "knowledge" in text or "知识库" in text:
        return "knowledge"

    if "logs" in text or "日志" in text:
        return "logs"

    if "当前文件夹" in text or "项目根目录" in text or "根目录" in text:
        return "."

    # Windows 绝对路径
    match = re.search(r"[A-Za-z]:\\[^\s，。；;]+", text)
    if match:
        return match.group(0).strip()

    # 相对目录
    match = re.search(r"(\.{1,2}[\\/][^\s，。；;]+)", text)
    if match:
        return match.group(1).strip()

    return None


def clean_query(text: str, remove_words):
    query = text

    for word in remove_words:
        query = query.replace(word, " ")

    query = re.sub(r"\s+", " ", query)
    query = query.strip(" ：:，,。？?！!；;")

    return query.strip()


def extract_after_keywords(text: str, keywords):
    """
    找到最后一个关键词后面的内容。
    """
    best_pos = -1
    best_keyword = None

    for kw in keywords:
        pos = text.rfind(kw)
        if pos > best_pos:
            best_pos = pos
            best_keyword = kw

    if best_pos >= 0 and best_keyword:
        return text[best_pos + len(best_keyword):].strip(" ：:，,。？?！!；;")

    return ""


def parse_tool_intent(user_text: str):
    """
    轻量工具识别器。

    原则：
    1. 只识别明确的本地安全工具。
    2. 不执行系统命令。
    3. 不确定就返回 None，让模型正常聊天。
    4. 斜杠命令由 version0.04.py 原有逻辑处理，这里不处理。
    """
    text = normalize_text(user_text)

    if not text:
        return None

    if text.startswith("/"):
        return None

    compact = re.sub(r"\s+", "", text)

    # ======================
    # 查看配置 / 日志 / 记忆 / 场景
    # ======================

    if compact in ["查看配置", "当前配置", "配置是什么", "看看配置"]:
        return {"action": "config_view"}

    if any(k in compact for k in ["日志在哪", "日志位置", "日志文件在哪", "logs在哪"]):
        return {"action": "logs_view"}

    if compact in ["你记住了什么", "查看记忆", "当前记忆", "长期记忆", "看看记忆"]:
        return {"action": "memory_show"}

    if compact in ["我们在哪", "现在在哪", "当前在哪", "当前场景", "查看场景", "看看场景", "现在什么地方"]:
        return {"action": "scene_view"}

    if compact in ["清除场景", "清空场景", "退出场景", "回到默认场景"]:
        return {"action": "scene_clear"}

    if compact in ["rag状态", "查看rag状态", "索引状态", "查看索引状态"]:
        return {"action": "rag_status"}

    # ======================
    # 添加长期记忆
    # ======================

    memory_prefixes = [
        "记住",
        "帮我记住",
        "你记住",
        "记一下",
        "长期记住",
        "加入记忆",
        "添加记忆",
    ]

    for prefix in memory_prefixes:
        if text.startswith(prefix):
            fact = text.replace(prefix, "", 1).strip(" ：:，,。")
            if fact:
                return {
                    "action": "memory_add",
                    "fact": fact
                }

    # ======================
    # 添加场景笔记
    # ======================

    scene_note_prefixes = [
        "场景记一下",
        "场景笔记",
        "记下场景",
        "记录场景",
    ]

    for prefix in scene_note_prefixes:
        if text.startswith(prefix):
            note = text.replace(prefix, "", 1).strip(" ：:，,。")
            if note:
                return {
                    "action": "scene_note",
                    "note": note
                }

    # ======================
    # 手动设置场景
    # ======================

    set_scene_prefixes = [
        "设置场景为",
        "当前场景是",
        "现在场景是",
        "我们现在在",
        "我们当前在",
    ]

    for prefix in set_scene_prefixes:
        if text.startswith(prefix):
            scene = text.replace(prefix, "", 1).strip(" ：:，,。")
            if scene:
                return {
                    "action": "scene_set",
                    "scene": scene
                }

    # ======================
    # 文件夹列表
    # ======================

    list_words = ["列出", "看看", "查看", "显示"]
    folder_words = ["文件夹", "目录", "里面有什么", "有哪些文件"]

    if any(w in text for w in list_words) and any(w in text for w in folder_words):
        path = extract_folder_path(text)
        if path:
            return {
                "action": "file_list",
                "path": path
            }

    # 更口语的：knowledge里有什么
    if ("knowledge" in text or "知识库" in text) and any(w in text for w in ["有什么", "有哪些", "文件"]):
        return {
            "action": "file_list",
            "path": "knowledge"
        }

    # ======================
    # 读取 / 总结文件
    # ======================

    read_words = ["读一下", "读取", "看看", "总结", "概括", "分析一下", "打开"]
    file_words = ["文件", ".py", ".md", ".txt", ".json", ".csv", ".log", ".lsf", ".m"]

    if any(w in text for w in read_words) and any(w in text for w in file_words):
        path = extract_file_like_path(text)
        if path:
            return {
                "action": "file_read",
                "path": path
            }

    # ======================
    # RAG 检索
    # ======================

    if any(k in compact.lower() for k in ["rag搜索", "rag检索", "向量检索", "向量搜索"]):
        query = clean_query(
            text,
            ["RAG", "rag", "搜索", "检索", "向量", "查一下", "帮我", "用"]
        )
        if query:
            return {
                "action": "rag_search",
                "query": query
            }

    # ======================
    # 知识库搜索
    # ======================

    knowledge_markers = ["知识库", "knowledge"]
    search_markers = ["搜索", "查一下", "查找", "找一下", "有没有", "相关资料", "资料"]

    if any(k in text for k in knowledge_markers) and any(k in text for k in search_markers):
        query = clean_query(
            text,
            ["知识库", "knowledge", "里", "有没有", "搜索", "查一下", "查找", "找一下", "相关资料", "资料", "帮我", "看看"]
        )
        if query:
            return {
                "action": "knowledge_search",
                "query": query
            }

    # “搜一下 xxx” 默认走知识库关键词搜索，避免每次都走 RAG
    if text.startswith("搜一下") or text.startswith("搜索一下") or text.startswith("查一下"):
        query = clean_query(text, ["搜一下", "搜索一下", "查一下", "帮我"])
        if query:
            return {
                "action": "knowledge_search",
                "query": query
            }

    # ======================
    # RAG 更新
    # ======================

    if compact in ["更新知识库索引", "更新rag", "更新RAG", "增量更新rag", "增量更新RAG"]:
        return {"action": "rag_update"}

    if compact in ["重建知识库索引", "重建rag", "重建RAG", "全量重建rag", "全量重建RAG"]:
        return {"action": "rag_rebuild"}

    return None
