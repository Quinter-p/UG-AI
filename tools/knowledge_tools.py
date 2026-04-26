# tools/knowledge_tools.py

import os
from tools.file_tools import read_text_file


def search_knowledge(keyword, knowledge_dir="knowledge", max_results=5, max_read_chars=20000):
    keyword = keyword.strip()

    if not keyword:
        return "请输入关键词。"

    if not os.path.exists(knowledge_dir):
        return "knowledge 文件夹不存在。"

    results = []

    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext not in [".txt", ".md", ".py", ".json", ".csv", ".lsf", ".m"]:
                continue

            path = os.path.join(root, file)

            content, error = read_text_file(path, max_read_chars=max_read_chars)

            if error or not content:
                continue

            lower_content = content.lower()
            lower_keyword = keyword.lower()

            if lower_keyword in lower_content:
                index = lower_content.find(lower_keyword)
                start = max(0, index - 300)
                end = min(len(content), index + 500)
                snippet = content[start:end]

                results.append({
                    "path": path,
                    "snippet": snippet
                })

    if not results:
        return f"没有在 knowledge 文件夹中找到关键词：{keyword}"

    output = []

    for i, item in enumerate(results[:max_results], 1):
        output.append(
            f"结果 {i}\n文件：{item['path']}\n片段：\n{item['snippet']}\n"
        )

    return "\n" + "\n".join(output)


def summarize_search_stream(keyword, result, llm_client, write_log=None):
    prompt = f"""
你是一个本地知识库助手。

用户搜索关键词：
{keyword}

以下是 knowledge 文件夹中的搜索结果：
{result}

请基于搜索结果，用中文简洁总结有用信息。
如果搜索结果为空，就说明没有找到。
"""

    print("AI：", end="", flush=True)
    reply = llm_client.generate_stream(prompt, temperature=0.3)
    print("\n")

    if write_log:
        write_log("TOOL_SEARCH", result)
        write_log("AI", reply)

    return reply
