# tools/file_tools.py

import os


ALLOWED_TEXT_EXT = [
    ".txt", ".py", ".md", ".json", ".csv", ".log",
    ".lsf", ".m", ".html", ".css", ".js"
]


def list_files(path):
    path = path.strip().strip('"')

    if not os.path.exists(path):
        return f"路径不存在：{path}"

    if not os.path.isdir(path):
        return f"这不是文件夹：{path}"

    items = os.listdir(path)

    if not items:
        return "文件夹为空。"

    lines = []

    for item in items:
        full_path = os.path.join(path, item)

        if os.path.isdir(full_path):
            lines.append(f"[DIR]  {item}")
        else:
            size_kb = os.path.getsize(full_path) / 1024
            lines.append(f"[FILE] {item}  ({size_kb:.1f} KB)")

    return "\n".join(lines)


def read_text_file(path, max_read_chars=20000):
    path = path.strip().strip('"')

    if not os.path.exists(path):
        return None, f"文件不存在：{path}"

    if not os.path.isfile(path):
        return None, f"这不是文件：{path}"

    ext = os.path.splitext(path)[1].lower()

    if ext not in ALLOWED_TEXT_EXT:
        return None, f"暂时只支持读取文本类文件：{ALLOWED_TEXT_EXT}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                content = f.read()
        except Exception as e:
            return None, f"文件编码读取失败：{e}"
    except Exception as e:
        return None, f"读取失败：{e}"

    if len(content) > max_read_chars:
        content = content[:max_read_chars] + "\n\n[内容过长，已截断]"

    return content, None


def summarize_file_stream(
    path,
    llm_client,
    write_log=None,
    user_instruction="请总结这个文件的主要内容。",
    max_read_chars=20000
):
    content, error = read_text_file(path, max_read_chars=max_read_chars)

    if error:
        print(f"AI：{error}\n")
        if write_log:
            write_log("AI", error)
        return error

    prompt = f"""
你是一个本地文件阅读助手。

用户要求：
{user_instruction}

文件路径：
{path}

文件内容：
{content}

请根据文件内容回答，不要编造。
"""

    print("AI：", end="", flush=True)
    result = llm_client.generate_stream(prompt, temperature=0.3)
    print("\n")

    if write_log:
        write_log("AI", result)

    return result
