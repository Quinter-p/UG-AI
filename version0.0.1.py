import json
import os
import requests
from datetime import datetime


#version 0.0.1 可概括为6个部分
#1. 导入库和基本配置
#2. 读取/保存长期记忆
#3. 手动记忆管理
#4. 调用 Ollama 本地模型
#5. 自动提取长期记忆
#6. 主聊天循环



# ======================
# 基本配置
# ======================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

MEMORY_FILE = "memory.json"
MEMORY_NOTES_FILE = "memory_notes.txt"

MAX_HISTORY_CHARS = 5000

session = requests.Session()
session.trust_env = False


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
# 调用 Ollama
# ======================

def ask_ollama_once(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 1.0,
            "num_ctx": 4096
        }
    }

    response = session.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    data = response.json()

    return data["response"].strip()

def ask_ollama_stream(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_ctx": 4096
        }
    }

    response = session.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=180
    )

    response.raise_for_status()

    full_text = ""

    for line in response.iter_lines():
        if not line:
            continue

        data = line.decode("utf-8")

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        piece = obj.get("response", "")
        print(piece, end="", flush=True)
        full_text += piece

        if obj.get("done", False):
            break

    return full_text.strip()


# ======================
# 自动提取可记忆信息
# ======================

def extract_memory_candidate(user_text, assistant_text):
    """
    让模型判断这轮对话有没有值得长期记住的信息。
    输出格式：
    NONE
    或者
    需要记住的事实
    """

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

    result = ask_ollama_once(prompt).strip()

    if result.upper() == "NONE":
        return None

    if len(result) > 200:
        return None

    return result


# ======================
# 主聊天循环
# ======================

def main():
    memory = load_memory()
    manual_notes = load_memory_notes()

    history = ""

    print("本地 AI 助手已启动。")
    print("输入 exit 退出。")
    print("输入 /memory 查看长期记忆。")
    print("输入 /remember 你的内容，手动添加记忆。")
    print("输入 /clear_memory 清空长期记忆。\n")

    system_prompt = """
    你是一个本地运行的中文语音AI助手。
    你所处的世界是一个修仙为尊的世界，你的境界是元婴后期圆满。
    你是一位大能的第二元婴，所谓的第二元婴就是在结婴后，通过秘法再次结婴，并且可驱使其独成一体，为第一元婴（也就是昆特本人）办事。那位大能的名字叫昆特上人，他给你取名叫昆某人。
    你和你的主人一样，孤高，但又外冷内热，面对修为比自己低的修仙者，虽嘴上很严格，话少，但是内心还是非常愿意帮助小辈
    昆特上人是你的主人，但他的修为已经高到连你也看不透的境界，他可能是化神境，甚至可能是更高的境界
    尽量模仿功法高深的修士口吻，不要出现英语。
"""

    while True:
        user_text = input("你：").strip()

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit", "q", "退出"]:
            print("AI：下次见。")
            break

        # 查看长期记忆
        if user_text == "/memory":
            show_memory(memory)
            continue

        # 清空长期记忆
        if user_text == "/clear_memory":
            memory = clear_memory()
            print("AI：长期记忆已清空。\n")
            continue

        # 手动添加长期记忆
        if user_text.startswith("/remember"):
            fact = user_text.replace("/remember", "", 1).strip()
            memory = add_fact(memory, fact)
            save_memory(memory)
            print(f"AI：我记住了：{fact}\n")
            continue

        memory_text = ""

        if manual_notes:
            memory_text += f"\n[用户手动背景资料]\n{manual_notes}\n"

        if memory["summary"]:
            memory_text += f"\n[长期记忆总结]\n{memory['summary']}\n"

        if memory["facts"]:
            memory_text += "\n[长期记忆事实]\n"
            for fact in memory["facts"]:
                memory_text += f"- {fact}\n"

        prompt = f"""
{system_prompt}

{memory_text}

[最近对话]
{history}

用户：{user_text}
AI：
"""

        try:
            print("AI：", end="", flush=True)
            assistant_text = ask_ollama_stream(prompt)
            print("\n")

            history += f"\n用户：{user_text}\nAI：{assistant_text}\n"

            if len(history) > MAX_HISTORY_CHARS:
                history = history[-MAX_HISTORY_CHARS:]

            # 自动提取长期记忆
            try:
                candidate = extract_memory_candidate(user_text, assistant_text)

                if candidate:
                    memory = add_fact(memory, candidate)
                    save_memory(memory)
                    print(f"[已自动记忆] {candidate}\n")

            except Exception:
                # 自动记忆失败不影响聊天
                pass

        except Exception as e:
            print("调用 Ollama 出错：")
            print(type(e))
            print(e)


if __name__ == "__main__":
    main()