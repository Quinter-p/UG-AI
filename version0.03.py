# version0.03.py

# /files 文件夹路径       列出文件夹内容
# /read 文件路径          读取文件内容并让 AI 总结
# /search 关键词          搜索 knowledge 文件夹
# /rag_build             增量更新 RAG 向量索引
# /rag_update            增量更新 RAG 向量索引
# /rag_rebuild           全量重建 RAG 向量索引
# /rag_search 关键词      测试 RAG 检索
# /rag_status            查看 RAG 索引状态
# /logs                  查看日志文件夹位置
# /memory                查看长期记忆
# /remember 内容         手动添加长期记忆
# /clear_memory          清空长期记忆
# /state                 查看当前角色/场景状态
# /scene                 查看当前场景状态
# /set_scene 场景描述     手动设置当前场景
# /scene_note 内容        添加场景笔记
# /clear_scene           清除当前场景状态
# /config                查看当前配置
# /tool_debug on/off/status  查看或切换自然语言工具识别调试

import json
import os
from datetime import datetime

from core.config_loader import load_config, pretty_config
from core.intent_router import parse_tool_intent

from tools.prompt_loader import load_pinned_system_prompt
from tools.file_tools import list_files, summarize_file_stream
from tools.knowledge_tools import search_knowledge, summarize_search_stream
from tools.rag_tools import (
    build_rag_index,
    update_rag_index,
    search_rag,
    format_rag_results,
    format_rag_context_for_prompt,
    rag_status
)
from pipelines.llm_ollama import OllamaClient
from character.style_prompts import STYLE_GUIDE, STYLE_EXAMPLES

from core.session_state import (
    load_state,
    save_state,
    update_state_from_user_input,
    build_state_text,
    reset_scene_state,
    set_scene_manually,
    add_scene_note,
    format_scene_view
)


# ======================
# 加载配置
# ======================

CONFIG = load_config()

MODEL_NAME = CONFIG["model"]["chat_model"]

OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"]
OLLAMA_KEEP_ALIVE = CONFIG["ollama"]["keep_alive"]

TEMPERATURE = CONFIG["generation"]["temperature"]
NUM_CTX = CONFIG["generation"]["num_ctx"]
NUM_PREDICT = CONFIG["generation"]["num_predict"]

MEMORY_FILE = CONFIG["paths"]["memory_file"]
MEMORY_NOTES_FILE = CONFIG["paths"]["memory_notes_file"]
KNOWLEDGE_DIR = CONFIG["paths"]["knowledge_dir"]
LOG_DIR = CONFIG["paths"]["log_dir"]

MAX_HISTORY_CHARS = CONFIG["limits"]["max_history_chars"]
MAX_READ_CHARS = CONFIG["limits"]["max_read_chars"]

ENABLE_AUTO_MEMORY = CONFIG["memory"]["enable_auto_memory"]

ENABLE_AUTO_RAG = CONFIG["rag"]["enable_auto_rag"]
RAG_EMBEDDING_MODEL = CONFIG["rag"]["embedding_model"]
RAG_INDEX_FILE = CONFIG["rag"]["index_file"]
RAG_TOP_K = CONFIG["rag"]["top_k"]
RAG_MIN_SCORE = CONFIG["rag"]["min_score"]
RAG_CHUNK_SIZE = CONFIG["rag"]["chunk_size"]
RAG_CHUNK_OVERLAP = CONFIG["rag"]["chunk_overlap"]

DEBUG_PROMPT = CONFIG["debug"].get("debug_prompt", False)
DEBUG_TOOL_ROUTER_DEFAULT = CONFIG["debug"].get("debug_tool_router", False)

USE_STYLE_EXAMPLES = CONFIG["style"]["use_style_examples"]


# ======================
# 初始化文件夹
# ======================

def ensure_dirs():
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("memory", exist_ok=True)


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
# 自动提取长期记忆
# 默认关闭
# ======================

def extract_memory_candidate(llm_client, user_text, assistant_text):
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

    result = llm_client.generate_once(prompt, temperature=0.2).strip()

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
# 构造普通聊天 prompt
# ======================

def build_chat_prompt(system_prompt, state_text, memory_text, rag_context, history, user_text):
    examples_text = STYLE_EXAMPLES if USE_STYLE_EXAMPLES else ""

    prompt = f"""
{system_prompt}

{STYLE_GUIDE}

{examples_text}

{state_text}

{memory_text}

{rag_context}

[最近对话]
{history}

[当前用户输入]
用户：{user_text}

[回答要求]
只回答当前用户输入。
必须给出实质内容，不能用“有何吩咐”这类空话代替回答。
如果用户的问题很短，也要根据当前身份和上下文自然回应。
如果当前状态显示场景未激活，不要延续旧场景。
如果当前状态显示场景激活，可以结合地点、危险等级、当前目标和场景摘要回答。
如果知识库检索结果中有相关内容，优先结合知识库回答。
如果知识库检索结果无相关片段，不要假装查到了资料。
回答要自然，像贴身器灵正在给主人回话。

AI：
"""
    return prompt


# ======================
# prompt 调试
# ======================

def debug_print_prompt(prompt):
    if not DEBUG_PROMPT:
        return

    print("\n" + "=" * 80)
    print("[DEBUG] FINAL PROMPT")
    print("=" * 80)
    print(prompt[:5000])

    if len(prompt) > 5000:
        print(f"\n[Prompt 过长，已截断显示。总长度：{len(prompt)} 字符]")

    print("=" * 80 + "\n")


# ======================
# 工具路由调试
# ======================

def debug_print_tool_router(user_text, intent, debug_tool_router):
    if not debug_tool_router:
        return

    print("\n" + "-" * 80)
    print("[TOOL_ROUTER_DEBUG]")
    print(f"输入：{user_text}")

    if intent:
        print("结果：命中工具")
        print(f"action：{intent.get('action')}")
        params = {k: v for k, v in intent.items() if k != "action"}
        print(f"params：{json.dumps(params, ensure_ascii=False)}")
    else:
        print("结果：未命中工具，将进入普通聊天")

    print("-" * 80 + "\n")


def handle_tool_debug_command(user_text, current_value):
    """
    返回：
    handled: bool
    new_value: bool
    """
    text = user_text.strip().lower()

    if text not in ["/tool_debug", "/tool_debug on", "/tool_debug off", "/tool_debug status"]:
        return False, current_value

    if text == "/tool_debug on":
        print("AI：自然语言工具识别调试已开启。\n")
        return True, True

    if text == "/tool_debug off":
        print("AI：自然语言工具识别调试已关闭。\n")
        return True, False

    status = "开启" if current_value else "关闭"
    print(f"AI：自然语言工具识别调试当前为：{status}\n")
    return True, current_value


# ======================
# 自动 RAG 检索
# ======================

def build_auto_rag_context(llm_client, user_text):
    if not ENABLE_AUTO_RAG:
        return "[知识库检索结果]\n自动 RAG 已关闭。"

    if not os.path.exists(RAG_INDEX_FILE):
        return "[知识库检索结果]\nRAG 索引尚未构建。可输入 /rag_build 构建索引。"

    try:
        results = search_rag(
            query=user_text,
            llm_client=llm_client,
            index_file=RAG_INDEX_FILE,
            embedding_model=RAG_EMBEDDING_MODEL,
            top_k=RAG_TOP_K,
            min_score=RAG_MIN_SCORE
        )

        rag_context = format_rag_context_for_prompt(results)

        if results:
            write_log("RAG", rag_context)

        return rag_context

    except Exception as e:
        return f"[知识库检索结果]\nRAG 检索失败：{type(e).__name__}: {e}"


def print_rag_result_summary(result):
    mode = result.get("mode", "unknown")

    if mode == "incremental_update":
        reply = (
            "RAG 增量更新完成。\n"
            f"新增/修改文件数：{result.get('changed_files', 0)}\n"
            f"删除文件数：{result.get('removed_files', 0)}\n"
            f"复用旧知识块：{result.get('kept_chunks', 0)}\n"
            f"新生成知识块：{result.get('new_chunks', 0)}\n"
            f"当前总知识块：{result.get('total_chunks', 0)}\n"
            f"索引文件：{result.get('index_file')}"
        )
    else:
        reply = (
            "RAG 全量索引构建完成。\n"
            f"处理文件数：{result.get('files', 0)}\n"
            f"切块数：{result.get('chunks', 0)}\n"
            f"索引文件：{result.get('index_file')}"
        )

    print(f"\nAI：{reply}\n")
    write_log("RAG_BUILD", reply)


# ======================
# 自然语言工具调用
# ======================

def handle_natural_tool_intent(intent, user_text, llm_client, memory, state):
    """
    返回：
    handled: bool
    memory: dict
    state: dict
    history_reply: str or None
    """
    if not intent:
        return False, memory, state, None

    action = intent.get("action")

    # 查看配置
    if action == "config_view":
        config_text = pretty_config(CONFIG)
        print("\n====== 当前配置 ======")
        print(config_text)
        print("====================\n")
        write_log("CONFIG", config_text)
        return True, memory, state, None

    # 日志位置
    if action == "logs_view":
        log_path = os.path.abspath(LOG_DIR)
        reply = f"日志保存在：{log_path}"
        print(f"AI：{reply}\n")
        write_log("AI", reply)
        return True, memory, state, reply

    # 查看记忆
    if action == "memory_show":
        show_memory(memory)
        return True, memory, state, None

    # 添加记忆
    if action == "memory_add":
        fact = intent.get("fact", "").strip()

        if not fact:
            return False, memory, state, None

        memory = add_fact(memory, fact)
        save_memory(memory)

        reply = f"我记住了：{fact}"
        print(f"AI：{reply}\n")
        write_log("AI", reply)
        return True, memory, state, reply

    # 查看场景
    if action == "scene_view":
        scene_text = format_scene_view(state)
        print("\n" + scene_text + "\n")
        write_log("SCENE", scene_text)
        return True, memory, state, None

    # 清除场景
    if action == "scene_clear":
        state = reset_scene_state(state)
        save_state(state)

        reply = "当前场景状态已清除。"
        print(f"AI：{reply}\n")
        write_log("AI", reply)
        return True, memory, state, reply

    # 设置场景
    if action == "scene_set":
        scene_desc = intent.get("scene", "").strip()

        if not scene_desc:
            return False, memory, state, None

        state = set_scene_manually(state, scene_desc)
        save_state(state)

        scene_text = format_scene_view(state)
        print("\n" + scene_text + "\n")
        write_log("SCENE_SET", scene_text)
        return True, memory, state, None

    # 场景笔记
    if action == "scene_note":
        note = intent.get("note", "").strip()

        if not note:
            return False, memory, state, None

        state = add_scene_note(state, note)
        save_state(state)

        reply = f"场景笔记已添加：{note}"
        print(f"AI：{reply}\n")
        write_log("SCENE_NOTE", note)
        return True, memory, state, reply

    # 列文件夹
    if action == "file_list":
        path = intent.get("path") or "."
        result = list_files(path)
        print(f"\n{result}\n")
        write_log("TOOL_FILES", result)
        return True, memory, state, None

    # 读文件
    if action == "file_read":
        path = intent.get("path")

        if not path:
            return False, memory, state, None

        summarize_file_stream(
            path=path,
            llm_client=llm_client,
            write_log=write_log,
            max_read_chars=MAX_READ_CHARS
        )
        return True, memory, state, None

    # 关键词知识库搜索
    if action == "knowledge_search":
        query = intent.get("query", "").strip()

        if not query:
            return False, memory, state, None

        result = search_knowledge(
            keyword=query,
            knowledge_dir=KNOWLEDGE_DIR,
            max_results=5,
            max_read_chars=MAX_READ_CHARS
        )

        print(result)

        summarize_search_stream(
            keyword=query,
            result=result,
            llm_client=llm_client,
            write_log=write_log
        )
        return True, memory, state, None

    # RAG 检索
    if action == "rag_search":
        query = intent.get("query", "").strip()

        if not query:
            return False, memory, state, None

        try:
            results = search_rag(
                query=query,
                llm_client=llm_client,
                index_file=RAG_INDEX_FILE,
                embedding_model=RAG_EMBEDDING_MODEL,
                top_k=RAG_TOP_K,
                min_score=RAG_MIN_SCORE
            )

            output = format_rag_results(results)

            if not output:
                output = "没有检索到高相似度片段。"

            print("\n" + output + "\n")
            write_log("RAG_SEARCH", output)

        except Exception as e:
            err = (
                f"RAG 检索失败：{type(e).__name__}: {e}\n"
                f"请确认已构建索引，并安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
            )
            print(f"AI：{err}\n")
            write_log("ERROR", err)

        return True, memory, state, None

    # RAG 状态
    if action == "rag_status":
        status = rag_status(RAG_INDEX_FILE)
        print("\n" + status + "\n")
        write_log("RAG_STATUS", status)
        return True, memory, state, None

    # RAG 增量更新
    if action == "rag_update":
        try:
            print("AI：开始增量更新 RAG 索引。\n")
            result = update_rag_index(
                llm_client=llm_client,
                knowledge_dir=KNOWLEDGE_DIR,
                index_file=RAG_INDEX_FILE,
                embedding_model=RAG_EMBEDDING_MODEL,
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP,
                max_read_chars=MAX_READ_CHARS
            )
            print_rag_result_summary(result)
        except Exception as e:
            err = (
                f"RAG 增量更新失败：{type(e).__name__}: {e}\n"
                f"请确认已安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
            )
            print(f"AI：{err}\n")
            write_log("ERROR", err)

        return True, memory, state, None

    # RAG 全量重建
    if action == "rag_rebuild":
        try:
            print("AI：开始全量重建 RAG 索引。知识库多时会比较慢。\n")
            result = build_rag_index(
                llm_client=llm_client,
                knowledge_dir=KNOWLEDGE_DIR,
                index_file=RAG_INDEX_FILE,
                embedding_model=RAG_EMBEDDING_MODEL,
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP,
                max_read_chars=MAX_READ_CHARS
            )
            print_rag_result_summary(result)
        except Exception as e:
            err = (
                f"RAG 全量重建失败：{type(e).__name__}: {e}\n"
                f"请确认已安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
            )
            print(f"AI：{err}\n")
            write_log("ERROR", err)

        return True, memory, state, None

    return False, memory, state, None


# ======================
# 主聊天循环
# ======================

def main():
    ensure_dirs()

    llm_client = OllamaClient(
        model_name=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        keep_alive=OLLAMA_KEEP_ALIVE
    )

    memory = load_memory()
    manual_notes = load_memory_notes()
    state = load_state()

    history = ""
    debug_tool_router = DEBUG_TOOL_ROUTER_DEFAULT

    print("本地 AI 助手已启动。")
    print(f"当前模型：{MODEL_NAME}")
    print(f"RAG：{'开启' if ENABLE_AUTO_RAG else '关闭'}")
    print(f"工具识别调试：{'开启' if debug_tool_router else '关闭'}")
    print("默认：流式输出。")
    print("输入 exit 退出。")
    print("输入 /tool_debug on 开启自然语言工具识别调试。")
    print("输入 /tool_debug off 关闭自然语言工具识别调试。")
    print("输入 /config 查看当前配置。")
    print("输入 /rag_build 或 /rag_update 增量更新 RAG 索引。")
    print("输入 /rag_rebuild 全量重建 RAG 索引。")
    print("输入 /rag_search 关键词，测试 RAG 检索。")
    print("输入 /rag_status 查看 RAG 索引状态。")
    print("输入 /memory 查看长期记忆。")
    print("输入 /remember 你的内容，手动添加记忆。")
    print("输入 /clear_memory 清空长期记忆。")
    print("输入 /state 查看当前角色/场景状态。")
    print("输入 /scene 查看当前场景状态。")
    print("输入 /set_scene 场景描述，手动设置当前场景。")
    print("输入 /scene_note 内容，添加场景笔记。")
    print("输入 /clear_scene 清除当前场景状态。")
    print("输入 /files 文件夹路径，列出文件。")
    print("输入 /read 文件路径，读取并总结文件。")
    print("输入 /search 关键词，搜索 knowledge 文件夹。")
    print("现在也支持部分自然语言工具调用，比如：读一下 version0.03.py、看看 knowledge 文件夹、记住 xxx、我们在哪。\n")

    system_prompt = load_pinned_system_prompt()

    if not system_prompt.strip():
        print("警告：system_prompt 为空。请检查 tools/prompt_loader.py 和 knowledge 路径。\n")

    while True:
        user_text = input("你：").strip()

        if not user_text:
            continue

        write_log("USER", user_text)

        # 每轮先更新状态
        state = update_state_from_user_input(state, user_text)
        save_state(state)

        # ======================
        # 退出
        # ======================

        if user_text.lower() in ["exit", "quit", "q", "退出"]:
            reply = "下次见。"
            print(f"AI：{reply}")
            write_log("AI", reply)
            break

        # ======================
        # 工具识别调试命令
        # ======================

        handled_debug, debug_tool_router = handle_tool_debug_command(user_text, debug_tool_router)
        if handled_debug:
            continue

        # ======================
        # 配置命令
        # ======================

        if user_text == "/config":
            config_text = pretty_config(CONFIG)
            print("\n====== 当前配置 ======")
            print(config_text)
            print("====================\n")
            write_log("CONFIG", config_text)
            continue

        # ======================
        # RAG 命令
        # ======================

        if user_text in ["/rag_build", "/rag_update", "/reg_build"]:
            try:
                print("AI：开始增量更新 RAG 索引。\n")
                result = update_rag_index(
                    llm_client=llm_client,
                    knowledge_dir=KNOWLEDGE_DIR,
                    index_file=RAG_INDEX_FILE,
                    embedding_model=RAG_EMBEDDING_MODEL,
                    chunk_size=RAG_CHUNK_SIZE,
                    chunk_overlap=RAG_CHUNK_OVERLAP,
                    max_read_chars=MAX_READ_CHARS
                )
                print_rag_result_summary(result)

            except Exception as e:
                err = (
                    f"RAG 增量更新失败：{type(e).__name__}: {e}\n"
                    f"请确认已安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
                )
                print(f"AI：{err}\n")
                write_log("ERROR", err)

            continue

        if user_text == "/rag_rebuild":
            try:
                print("AI：开始全量重建 RAG 索引。知识库多时会比较慢。\n")
                result = build_rag_index(
                    llm_client=llm_client,
                    knowledge_dir=KNOWLEDGE_DIR,
                    index_file=RAG_INDEX_FILE,
                    embedding_model=RAG_EMBEDDING_MODEL,
                    chunk_size=RAG_CHUNK_SIZE,
                    chunk_overlap=RAG_CHUNK_OVERLAP,
                    max_read_chars=MAX_READ_CHARS
                )
                print_rag_result_summary(result)

            except Exception as e:
                err = (
                    f"RAG 全量重建失败：{type(e).__name__}: {e}\n"
                    f"请确认已安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
                )
                print(f"AI：{err}\n")
                write_log("ERROR", err)

            continue

        if user_text.startswith("/rag_search"):
            query = user_text.replace("/rag_search", "", 1).strip()

            if not query:
                reply = "请在 /rag_search 后面输入检索内容。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            try:
                results = search_rag(
                    query=query,
                    llm_client=llm_client,
                    index_file=RAG_INDEX_FILE,
                    embedding_model=RAG_EMBEDDING_MODEL,
                    top_k=RAG_TOP_K,
                    min_score=RAG_MIN_SCORE
                )

                output = format_rag_results(results)

                if not output:
                    output = "没有检索到高相似度片段。"

                print("\n" + output + "\n")
                write_log("RAG_SEARCH", output)

            except Exception as e:
                err = (
                    f"RAG 检索失败：{type(e).__name__}: {e}\n"
                    f"请确认已构建索引，并安装 embedding 模型：ollama pull {RAG_EMBEDDING_MODEL}"
                )
                print(f"AI：{err}\n")
                write_log("ERROR", err)

            continue

        if user_text == "/rag_status":
            status = rag_status(RAG_INDEX_FILE)
            print("\n" + status + "\n")
            write_log("RAG_STATUS", status)
            continue

        # ======================
        # 状态 / 场景命令
        # ======================

        if user_text == "/state":
            state_text = build_state_text(state)
            print("\n" + state_text + "\n")
            write_log("STATE", state_text)
            continue

        if user_text == "/scene":
            scene_text = format_scene_view(state)
            print("\n" + scene_text + "\n")
            write_log("SCENE", scene_text)
            continue

        if user_text.startswith("/set_scene"):
            scene_desc = user_text.replace("/set_scene", "", 1).strip()

            if not scene_desc:
                reply = "请在 /set_scene 后面输入场景描述。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            state = set_scene_manually(state, scene_desc)
            save_state(state)

            scene_text = format_scene_view(state)
            print("\n" + scene_text + "\n")
            write_log("SCENE_SET", scene_text)
            continue

        if user_text.startswith("/scene_note"):
            note = user_text.replace("/scene_note", "", 1).strip()

            if not note:
                reply = "请在 /scene_note 后面输入要添加的场景笔记。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            state = add_scene_note(state, note)
            save_state(state)

            reply = f"场景笔记已添加：{note}"
            print(f"AI：{reply}\n")
            write_log("SCENE_NOTE", note)
            continue

        if user_text == "/clear_scene":
            state = reset_scene_state(state)
            save_state(state)

            reply = "当前场景状态已清除。"
            print(f"AI：{reply}\n")
            write_log("AI", reply)
            continue

        # ======================
        # 记忆命令
        # ======================

        if user_text == "/memory":
            show_memory(memory)
            continue

        if user_text == "/clear_memory":
            memory = clear_memory()
            reply = "长期记忆已清空。"
            print(f"AI：{reply}\n")
            write_log("AI", reply)
            continue

        if user_text.startswith("/remember"):
            fact = user_text.replace("/remember", "", 1).strip()

            if not fact:
                reply = "请在 /remember 后面输入要记住的内容。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
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

            summarize_file_stream(
                path=path,
                llm_client=llm_client,
                write_log=write_log,
                max_read_chars=MAX_READ_CHARS
            )
            continue

        if user_text.startswith("/search"):
            keyword = user_text.replace("/search", "", 1).strip()

            if not keyword:
                reply = "请提供搜索关键词。"
                print(f"AI：{reply}\n")
                write_log("AI", reply)
                continue

            result = search_knowledge(
                keyword=keyword,
                knowledge_dir=KNOWLEDGE_DIR,
                max_results=5,
                max_read_chars=MAX_READ_CHARS
            )

            print(result)

            summarize_search_stream(
                keyword=keyword,
                result=result,
                llm_client=llm_client,
                write_log=write_log
            )
            continue

        if user_text == "/logs":
            log_path = os.path.abspath(LOG_DIR)
            reply = f"日志保存在：{log_path}"
            print(f"AI：{reply}\n")
            write_log("AI", reply)
            continue

        # ======================
        # 自然语言工具调用
        # ======================

        intent = parse_tool_intent(user_text)
        debug_print_tool_router(user_text, intent, debug_tool_router)

        handled, memory, state, history_reply = handle_natural_tool_intent(
            intent=intent,
            user_text=user_text,
            llm_client=llm_client,
            memory=memory,
            state=state
        )

        if handled:
            if history_reply:
                history += f"\n用户：{user_text}\nAI：{history_reply}\n"

                if len(history) > MAX_HISTORY_CHARS:
                    history = history[-MAX_HISTORY_CHARS:]

            continue

        # ======================
        # 普通聊天：自动 RAG + 模型
        # ======================

        memory_text = build_memory_text(memory, manual_notes)
        state_text = build_state_text(state)
        rag_context = build_auto_rag_context(llm_client, user_text)

        prompt = build_chat_prompt(
            system_prompt=system_prompt,
            state_text=state_text,
            memory_text=memory_text,
            rag_context=rag_context,
            history=history,
            user_text=user_text
        )

        debug_print_prompt(prompt)

        try:
            print("AI：", end="", flush=True)
            assistant_text = llm_client.generate_stream(prompt, temperature=TEMPERATURE)
            print("\n")

            write_log("AI", assistant_text)

            history += f"\n用户：{user_text}\nAI：{assistant_text}\n"

            if len(history) > MAX_HISTORY_CHARS:
                history = history[-MAX_HISTORY_CHARS:]

            # 自动提取长期记忆：默认关闭，避免每轮额外调用一次模型
            if ENABLE_AUTO_MEMORY:
                try:
                    candidate = extract_memory_candidate(llm_client, user_text, assistant_text)

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


if __name__ == "__main__":
    main()
