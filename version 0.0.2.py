#version0.02 改动如下
#/files 文件夹路径      列出文件夹内容
#/read 文件路径         读取文件内容并让 AI 总结
#/search 关键词         搜索 knowledge 文件夹
#/logs                 查看日志文件夹位置

import json
import os
import requests
from datetime import datetime
from tools.prompt_loader import load_pinned_system_prompt
# ======================
# 基本配置
# ======================

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

MEMORY_FILE = "memory.json"
MEMORY_NOTES_FILE = "memory_notes.txt"

KNOWLEDGE_DIR = "knowledge"
LOG_DIR = "logs"

MAX_HISTORY_CHARS = 20000
MAX_READ_CHARS = 20000

session = requests.Session()
session.trust_env = False


# ======================
# 初始化文件夹
# ======================

def ensure_dirs():
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


# ======================
# 日志系统
# ======================

def get_log_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{today}_chat.txt")


def write_log(role, text):
    log_file = get_log_file()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{time_str}] {role}:\n{text}\n\n")


# ======================
# 读取 / 保存记忆
# ======================

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "facts": [],
            "summary": "",
            "updated_at": ""
        }

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "facts": [],
            "summary": "",
            "updated_at": ""
        }


def save_memory(memory):
    memory["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def load_memory_notes():
    if not os.path.exists(MEMORY_NOTES_FILE):
        return ""

    with open(MEMORY_NOTES_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def add_fact(memory, fact):
    fact = fact.strip()

    if not fact:
        return memory

    if fact not in memory["facts"]:
        memory["facts"].append(fact)

    return memory


def show_memory(memory):
    print("\n====== 当前长期记忆 ======")

    if memory["summary"]:
        print("\n[总结]")
        print(memory["summary"])

    if memory["facts"]:
        print("\n[事实]")
        for i, fact in enumerate(memory["facts"], 1):
            print(f"{i}. {fact}")
    else:
        print("暂无长期记忆。")

    print("========================\n")


def clear_memory():
    memory = {
        "facts": [],
        "summary": "",
        "updated_at": ""
    }
    save_memory(memory)
    return memory


# ======================
# 调用 Ollama：一次性输出
# 主要给自动记忆用
# ======================

def ask_ollama_once(prompt, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096
        }
    }

    response = session.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    data = response.json()

    return data["response"].strip()


# ======================
# 调用 Ollama：流式输出
# 以后默认聊天都用这个
# ======================

def ask_ollama_stream(prompt, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096
        }
    }

    response = session.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        stream=True,
        timeout=180
    )

    response.raise_for_status()

    full_text = ""

    for line in response.iter_lines():
        if not line:
            continue

        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        piece = data.get("response", "")

        print(piece, end="", flush=True)
        full_text += piece

        if data.get("done", False):
            break

    return full_text.strip()

def ask_ollama_chat_stream(messages, temperature=0.4):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "num_ctx": 4096
        }
    }

    response = session.post(
        OLLAMA_CHAT_URL,
        json=payload,
        stream=True,
        timeout=180
    )

    response.raise_for_status()

    full_text = ""

    for line in response.iter_lines():
        if not line:
            continue

        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        message = data.get("message", {})
        piece = message.get("content", "")

        print(piece, end="", flush=True)
        full_text += piece

        if data.get("done", False):
            break

    return full_text.strip()
# ======================
# 文件工具
# ======================

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


def read_text_file(path):
    path = path.strip().strip('"')

    if not os.path.exists(path):
        return None, f"文件不存在：{path}"

    if not os.path.isfile(path):
        return None, f"这不是文件：{path}"

    ext = os.path.splitext(path)[1].lower()

    allowed_ext = [
        ".txt", ".py", ".md", ".json", ".csv", ".log",
        ".lsf", ".m", ".html", ".css", ".js"
    ]

    if ext not in allowed_ext:
        return None, f"暂时只支持读取文本类文件：{allowed_ext}"

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

    if len(content) > MAX_READ_CHARS:
        content = content[:MAX_READ_CHARS] + "\n\n[内容过长，已截断]"

    return content, None


def summarize_file_stream(path, user_instruction="请总结这个文件的主要内容。"):
    content, error = read_text_file(path)

    if error:
        print(f"AI：{error}\n")
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
    result = ask_ollama_stream(prompt, temperature=0.3)
    print("\n")

    write_log("AI", result)

    return result


# ======================
# knowledge 搜索
# ======================

def search_knowledge(keyword):
    keyword = keyword.strip()

    if not keyword:
        return "请输入关键词。"

    if not os.path.exists(KNOWLEDGE_DIR):
        return "knowledge 文件夹不存在。"

    results = []

    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext not in [".txt", ".md", ".py", ".json", ".csv", ".lsf", ".m"]:
                continue

            path = os.path.join(root, file)

            content, error = read_text_file(path)

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

    for i, item in enumerate(results[:5], 1):
        output.append(
            f"结果 {i}\n文件：{item['path']}\n片段：\n{item['snippet']}\n"
        )

    return "\n" + "\n".join(output)


def summarize_search_stream(keyword, result):
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
    reply = ask_ollama_stream(prompt, temperature=0.3)
    print("\n")

    write_log("TOOL_SEARCH", result)
    write_log("AI", reply)

    return reply


# ======================
# 自动提取长期记忆
# ======================

def extract_memory_candidate(user_text, assistant_text):
    prompt = f"""
你是一个记忆提取器。你的任务是判断下面这轮对话中，是否有值得长期记住的用户信息。

只记住这些类型：
1. 用户长期偏好
2. 用户长期项目
3. 用户长期设备/环境
4. 用户明确要求你记住的信息
5. 对未来帮助明显有用的信息

不要记住：
1. 临时问题
2. 一次性报错
3. 隐私敏感信息
4. 没有长期价值的闲聊

如果没有值得记住的内容，只输出：
NONE

如果有，只输出一条简洁事实，不要解释。

用户说：
{user_text}

助手回答：
{assistant_text}

输出：
"""

    result = ask_ollama_once(prompt, temperature=0.2).strip()

    if result.upper() == "NONE":
        return None

    if len(result) > 200:
        return None

    return result


# ======================
# 构造记忆文本
# ======================

def build_memory_text(memory, manual_notes):
    memory_text = ""

    if manual_notes:
        memory_text += f"\n[用户手动背景资料]\n{manual_notes}\n"

    if memory["summary"]:
        memory_text += f"\n[长期记忆总结]\n{memory['summary']}\n"

    if memory["facts"]:
        memory_text += "\n[长期记忆事实]\n"
        for fact in memory["facts"]:
            memory_text += f"- {fact}\n"

    return memory_text


# ======================
# 主聊天循环
# ======================

def main():
    ensure_dirs()

    memory = load_memory()
    manual_notes = load_memory_notes()

    history = ""

    print("本地 AI 助手已启动。")
    print("默认：流式输出。")
    print("输入 exit 退出。")
    print("输入 /memory 查看长期记忆。")
    print("输入 /remember 你的内容，手动添加记忆。")
    print("输入 /clear_memory 清空长期记忆。")
    print("输入 /files 文件夹路径，列出文件。")
    print("输入 /read 文件路径，读取并总结文件。")
    print("输入 /search 关键词，搜索 knowledge 文件夹。")
    print("输入 /logs 查看日志文件位置。\n")

    system_prompt = load_pinned_system_prompt()


    while True:
        user_text = input("你：").strip()

        if not user_text:
            continue

        write_log("USER", user_text)

        # ======================
        # 退出
        # ======================

        if user_text.lower() in ["exit", "quit", "q", "退出"]:
            print("AI：下次见。")
            write_log("AI", "下次见。")
            break

        # ======================
        # 记忆命令
        # ======================

        if user_text == "/memory":
            show_memory(memory)
            continue

        if user_text == "/clear_memory":
            memory = clear_memory()
            print("AI：长期记忆已清空。\n")
            write_log("AI", "长期记忆已清空。")
            continue

        if user_text.startswith("/remember"):
            fact = user_text.replace("/remember", "", 1).strip()

            if not fact:
                print("AI：请在 /remember 后面输入要记住的内容。\n")
                continue

            memory = add_fact(memory, fact)
            save_memory(memory)

            reply = f"我记住了：{fact}"
            print(f"AI：{reply}\n")
            write_log("AI", reply)
            continue

        # ======================
        # 文件工具命令
        # ======================

        if user_text.startswith("/files"):
            path = user_text.replace("/files", "", 1).strip()

            if not path:
                path = "."

            result = list_files(path)
            print(f"\n{result}\n")
            write_log("TOOL_FILES", result)
            continue

        if user_text.startswith("/read"):
            path = user_text.replace("/read", "", 1).strip()

            if not path:
                reply = "请提供文件路径。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            summarize_file_stream(path)
            continue

        if user_text.startswith("/search"):
            keyword = user_text.replace("/search", "", 1).strip()

            if not keyword:
                reply = "请提供搜索关键词。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            result = search_knowledge(keyword)
            print(result)

            summarize_search_stream(keyword, result)
            continue

        if user_text == "/logs":
            log_path = os.path.abspath(LOG_DIR)
            reply = f"日志保存在：{log_path}"
            print(f"AI：{reply}\n")
            write_log("AI", reply)
            continue

        # ======================
        # 普通聊天：默认流式输出
        # ======================

        memory_text = build_memory_text(memory, manual_notes)


        chat_system_prompt = f"""
        {system_prompt}

        {memory_text}
        """

        chat_user_prompt = f"""
        [最近对话]
        {history}

        [当前用户输入]
        {user_text}
        """

        messages = [
            {
                "role": "system",
                "content": chat_system_prompt
            },
            {
                "role": "user",
                "content": chat_user_prompt
            }
        ]

        try:
            print("AI：", end="", flush=True)
            assistant_text = ask_ollama_chat_stream(messages, temperature=0.35)
            print("\n")

            write_log("AI", assistant_text)

            history += f"\n用户：{user_text}\nAI：{assistant_text}\n"

            if len(history) > MAX_HISTORY_CHARS:
                history = history[-MAX_HISTORY_CHARS:]

            # 自动提取长期记忆
            try:
                candidate = extract_memory_candidate(user_text, assistant_text)

                if candidate:
                    memory = add_fact(memory, candidate)
                    save_memory(memory)

                    memory_msg = f"[已自动记忆] {candidate}"
                    print(f"{memory_msg}\n")
                    write_log("MEMORY", candidate)

            except Exception:
                pass

        except Exception as e:
            err = f"调用 Ollama 出错：{type(e)} {e}"
            print(err)
            write_log("ERROR", err)

def debug_print_prompt(system_prompt, final_prompt):
    print("\n" + "=" * 80)
    print("[DEBUG] SYSTEM PROMPT")
    print("=" * 80)

    if not system_prompt.strip():
        print("[警告] system_prompt 是空的！")
    else:
        print(system_prompt[:3000])
        if len(system_prompt) > 3000:
            print(f"\n[system_prompt 过长，已截断显示。总长度：{len(system_prompt)} 字符]")

    print("\n" + "=" * 80)
    print("[DEBUG] FINAL PROMPT SENT TO OLLAMA")
    print("=" * 80)

    print(final_prompt[:5000])
    if len(final_prompt) > 5000:
        print(f"\n[final_prompt 过长，已截断显示。总长度：{len(final_prompt)} 字符]")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()