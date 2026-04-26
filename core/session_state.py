# core/session_state.py

import json
import os
import re
from datetime import datetime


STATE_FILE = os.path.join("memory", "scene_state.json")


DEFAULT_HOME_LOCATION = "UG宗"
DEFAULT_HOME_SCENE = "UG宗门内"


DEFAULT_STATE = {
    "mode": "dialogue",
    "user_identity": "昆特上人",
    "user_alias": None,
    "assistant_identity": "灵寰阁器灵",

    # 默认所在地：UG宗
    "scene_active": False,
    "current_scene": DEFAULT_HOME_SCENE,
    "location": DEFAULT_HOME_LOCATION,
    "danger_level": "低",
    "objective": None,
    "scene_summary": "默认位于UG宗门内，环境相对安全。",
    "scene_notes": [],

    "last_topic": None,
    "turn_count": 0,
    "scene_turn_count": 0,
    "updated_at": ""
}


GENERAL_KEYWORDS = [
    "你的主人",
    "主人是谁",
    "介绍一下你的主人",
    "介绍你的主人",
    "昆特",
    "施耐德",
    "你是谁",
    "你叫什么",
    "我是",
    "我叫",
    "知识库",
    "代码",
    "模型",
    "prompt",
    "记忆",
    "rag",
    "RAG",
    "配置",
    "文件",
    "报错",
    "什么意思",
]


SCENE_CONTINUE_KEYWORDS = [
    "继续",
    "往前",
    "前进",
    "后退",
    "躲",
    "探查",
    "查看",
    "看看",
    "附近",
    "这里",
    "此地",
    "风雪",
    "雪原",
    "洞府",
    "密林",
    "敌人",
    "妖兽",
    "禁制",
    "阵法",
    "传送",
    "脚下",
    "远处",
    "气息",
    "危险",
    "过去",
    "离开",
    "返回",
    "回去",
    "回到",
]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_state_dir():
    os.makedirs("memory", exist_ok=True)


def save_state(state):
    ensure_state_dir()
    state["updated_at"] = now_str()

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_state():
    ensure_state_dir()

    if not os.path.exists(STATE_FILE):
        state = DEFAULT_STATE.copy()
        save_state(state)
        return state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = DEFAULT_STATE.copy()

    # 兼容旧版 state：缺字段则补齐
    for key, value in DEFAULT_STATE.items():
        if key not in state:
            if isinstance(value, list):
                state[key] = value.copy()
            else:
                state[key] = value

    # 如果旧版 state 里没有地点，默认补成 UG宗
    if not state.get("location"):
        state["location"] = DEFAULT_HOME_LOCATION

    if not state.get("current_scene"):
        state["current_scene"] = DEFAULT_HOME_SCENE

    if not state.get("danger_level"):
        state["danger_level"] = "低"

    return state


def apply_location_state(state, location, source_text="", objective=None):
    """
    根据地点更新场景状态。
    安全据点：scene_active=False
    险地/外部地点：scene_active=True
    """
    location = location or DEFAULT_HOME_LOCATION
    source_text = source_text.strip()

    if location in ["UG宗", "宗门", "山门", "UG宗门内"]:
        state["mode"] = "dialogue"
        state["scene_active"] = False
        state["current_scene"] = DEFAULT_HOME_SCENE
        state["location"] = DEFAULT_HOME_LOCATION
        state["danger_level"] = "低"
        state["objective"] = objective
        state["scene_summary"] = "当前已回到UG宗门内，环境相对安全。"
        state["scene_turn_count"] = 0
        state["last_topic"] = "location_transition_home"
        return state

    danger_level = infer_danger_level(source_text or location)

    state["mode"] = "roleplay_dialogue"
    state["scene_active"] = True
    state["current_scene"] = location
    state["location"] = location
    state["danger_level"] = danger_level
    state["objective"] = objective or infer_objective(source_text)
    state["last_topic"] = "location_transition"
    state["scene_turn_count"] = 0

    state["scene_summary"] = summarize_scene_from_text(
        source_text or location,
        location=location,
        danger_level=danger_level,
        objective=state.get("objective")
    )

    return state


def reset_scene_state(state):
    """
    清除当前外部场景，回到默认 UG宗门内。
    """
    state["mode"] = "dialogue"
    state["scene_active"] = False
    state["current_scene"] = DEFAULT_HOME_SCENE
    state["location"] = DEFAULT_HOME_LOCATION
    state["danger_level"] = "低"
    state["objective"] = None
    state["scene_summary"] = "默认位于UG宗门内，环境相对安全。"
    state["scene_notes"] = []
    state["scene_turn_count"] = 0
    state["last_topic"] = "clear_scene_home"
    return state


def detect_alias(user_text):
    text = user_text.strip()

    match = re.match(r"^我是\s*(.+)$", text)
    if match:
        return match.group(1).strip()

    match = re.match(r"^我叫\s*(.+)$", text)
    if match:
        return match.group(1).strip()

    return None


def extract_bracket_scene(user_text):
    """
    提取括号中的场景描述：
    （二人突然眼前一黑，再睁眼，竟是一片雪花纷飞的极寒之地）前辈？
    """
    patterns = [
        r"（(.+?)）",
        r"\((.+?)\)",
        r"【(.+?)】",
        r"\[(.+?)\]"
    ]

    for pattern in patterns:
        match = re.search(pattern, user_text)
        if match:
            return match.group(1).strip()

    return None


def infer_location(text):
    if not text:
        return None

    if any(k in text for k in ["UG宗", "宗门", "山门", "宗内", "门内"]):
        return DEFAULT_HOME_LOCATION

    if any(k in text for k in ["雪", "寒", "冰", "极寒", "风雪"]):
        return "极寒雪原"

    if any(k in text for k in ["密林", "森林", "树林", "林中"]):
        return "密林"

    if "洞府" in text:
        return "洞府"

    if "坊市" in text:
        return "坊市"

    if any(k in text for k in ["传送", "眼前一黑", "阵光", "空间"]):
        return "未知传送地点"

    if any(k in text for k in ["遗迹", "古殿", "废墟"]):
        return "古修遗迹"

    if any(k in text for k in ["河", "湖", "海", "水域"]):
        return "水域"

    if any(k in text for k in ["秘境", "禁地"]):
        return "秘境"

    return None


def detect_explicit_location_transition(user_text):
    """
    检测明确的地点迁移。
    这些规则优先级高于普通话题判断。
    """
    text = user_text.strip()

    home_patterns = [
        "回到宗门",
        "回宗门",
        "返回宗门",
        "回到UG宗",
        "回UG宗",
        "返回UG宗",
        "回到山门",
        "回山门",
        "返回山门",
        "回宗",
        "回到门内",
        "返回门内",
        "回到宗内",
        "返回宗内",
        "回来了，宗门",
        "已经回到宗门",
        "已经回到UG宗",
    ]

    if any(p in text for p in home_patterns):
        return {
            "location": DEFAULT_HOME_LOCATION,
            "objective": None,
            "source_text": text
        }

    # 明确去往某地
    movement_patterns = [
        "来到",
        "抵达",
        "到达",
        "进入",
        "前往",
        "去往",
        "去了",
        "传送到",
        "传送至",
        "被传送到",
        "被传送至",
        "落到",
        "落入",
        "回到",
        "返回",
    ]

    if any(p in text for p in movement_patterns):
        location = infer_location(text)

        if location:
            return {
                "location": location,
                "objective": infer_objective(text),
                "source_text": text
            }

    return None


def infer_danger_level(text):
    if not text:
        return "未知"

    if any(k in text for k in ["UG宗", "宗门", "山门", "宗内", "门内", "坊市"]):
        return "低"

    high_words = ["血迹", "尸", "杀气", "元婴", "结丹", "禁制", "大阵", "妖兽群", "追杀", "魔气", "空间裂缝", "秘境", "禁地"]
    medium_words = ["陌生", "传送", "风雪", "寒气", "密林", "洞府", "遗迹", "妖兽", "阵法", "气息", "雪原"]
    low_words = ["坊市", "宗门", "洞府内", "安全", "客栈", "UG宗"]

    if any(k in text for k in high_words):
        return "高"

    if any(k in text for k in medium_words):
        return "中"

    if any(k in text for k in low_words):
        return "低"

    return "未知"


def infer_objective(text):
    if not text:
        return None

    if any(k in text for k in ["回宗门", "回到宗门", "返回宗门", "回UG宗", "回到UG宗"]):
        return None

    if any(k in text for k in ["出去", "离开", "脱身", "回去"]):
        return "寻找退路"

    if any(k in text for k in ["找", "寻找", "打听", "线索"]):
        return "寻找线索"

    if any(k in text for k in ["躲", "隐藏", "避开"]):
        return "隐蔽避险"

    if any(k in text for k in ["探查", "查看", "看看"]):
        return "谨慎探查"

    if any(k in text for k in ["突破", "修炼", "闭关"]):
        return "修炼准备"

    return None


def is_general_topic(user_text):
    return any(k in user_text for k in GENERAL_KEYWORDS)


def seems_scene_related(user_text):
    return any(k in user_text for k in SCENE_CONTINUE_KEYWORDS)


def summarize_scene_from_text(text, location=None, danger_level=None, objective=None):
    text = text.strip()

    if len(text) > 80:
        text = text[:80] + "..."

    parts = []

    if location:
        parts.append(f"地点：{location}")

    if danger_level and danger_level != "未知":
        parts.append(f"危险等级：{danger_level}")

    if objective:
        parts.append(f"当前目标：{objective}")

    if text:
        parts.append(f"场景线索：{text}")

    if not parts:
        return ""

    return "；".join(parts)


def update_scene_from_text(state, user_text):
    bracket_scene = extract_bracket_scene(user_text)
    source_text = bracket_scene or user_text

    location = infer_location(source_text)
    danger_level = infer_danger_level(source_text)
    objective = infer_objective(source_text)

    if not location and not seems_scene_related(user_text):
        return state

    # 如果当前只是“继续/看看/过去”等，但没有新地点，就沿用旧地点
    location = location or state.get("location") or DEFAULT_HOME_LOCATION

    # 如果沿用的是 UG宗，保持安全状态；否则激活外部场景
    if location == DEFAULT_HOME_LOCATION:
        state = apply_location_state(
            state,
            location=DEFAULT_HOME_LOCATION,
            source_text=source_text,
            objective=objective
        )
        return state

    state["mode"] = "roleplay_dialogue"
    state["scene_active"] = True
    state["current_scene"] = location or state.get("current_scene") or "未命名场景"
    state["location"] = location or state.get("location")
    state["danger_level"] = danger_level if danger_level != "未知" else state.get("danger_level", "未知")
    state["objective"] = objective or state.get("objective")
    state["last_topic"] = "scene"
    state["scene_turn_count"] = int(state.get("scene_turn_count", 0)) + 1

    summary = summarize_scene_from_text(
        source_text,
        location=state.get("location"),
        danger_level=state.get("danger_level"),
        objective=state.get("objective")
    )

    if summary:
        state["scene_summary"] = summary

    return state


def update_state_from_user_input(state, user_text):
    """
    根据用户输入更新状态。
    注意：状态只是给模型看的结构化上下文，不直接替模型回答。
    """
    state["turn_count"] = int(state.get("turn_count", 0)) + 1

    # 1. 明确地点迁移优先级最高
    transition = detect_explicit_location_transition(user_text)
    if transition:
        state = apply_location_state(
            state,
            location=transition["location"],
            source_text=transition["source_text"],
            objective=transition.get("objective")
        )
        return state

    # 2. 化名
    alias = detect_alias(user_text)
    if alias:
        state["user_alias"] = alias
        state["mode"] = "dialogue"
        state["scene_active"] = False
        state["last_topic"] = "alias"
        return state

    # 3. 普通话题：中断场景表现，但不清空所在地
    if is_general_topic(user_text):
        state["mode"] = "dialogue"
        state["scene_active"] = False
        state["last_topic"] = "general"
        return state

    # 4. 新场景或场景延续
    state = update_scene_from_text(state, user_text)

    return state


def set_scene_manually(state, scene_text):
    scene_text = scene_text.strip()

    if not scene_text:
        return state

    location = infer_location(scene_text) or scene_text
    objective = infer_objective(scene_text)

    state = apply_location_state(
        state,
        location=location,
        source_text=scene_text,
        objective=objective
    )

    state["last_topic"] = "manual_set_scene"

    return state


def add_scene_note(state, note):
    note = note.strip()

    if not note:
        return state

    notes = state.get("scene_notes", [])

    if not isinstance(notes, list):
        notes = []

    notes.append({
        "time": now_str(),
        "note": note
    })

    # 最多保留最近 20 条场景笔记
    state["scene_notes"] = notes[-20:]

    if state.get("scene_active"):
        state["last_topic"] = "scene_note"

    return state


def build_state_text(state):
    """
    生成给模型看的状态文本。
    注意：这是状态，不是聊天历史。
    """
    lines = []

    lines.append("[当前角色状态]")
    lines.append(f"- 用户核心身份：{state.get('user_identity', '昆特上人')}")
    lines.append(f"- 用户化名：{state.get('user_alias') or '无'}")
    lines.append(f"- AI身份：{state.get('assistant_identity', '灵寰阁器灵')}")
    lines.append(f"- 当前模式：{state.get('mode', 'dialogue')}")
    lines.append(f"- 场景是否激活：{state.get('scene_active', False)}")
    lines.append(f"- 当前场景：{state.get('current_scene') or DEFAULT_HOME_SCENE}")
    lines.append(f"- 地点：{state.get('location') or DEFAULT_HOME_LOCATION}")
    lines.append(f"- 危险等级：{state.get('danger_level') or '未知'}")
    lines.append(f"- 当前目标：{state.get('objective') or '无'}")
    lines.append(f"- 场景摘要：{state.get('scene_summary') or '无'}")
    lines.append(f"- 最近话题：{state.get('last_topic') or '无'}")
    lines.append(f"- 总轮数：{state.get('turn_count', 0)}")
    lines.append(f"- 场景轮数：{state.get('scene_turn_count', 0)}")

    notes = state.get("scene_notes", [])
    if notes:
        lines.append("")
        lines.append("[场景笔记]")
        for item in notes[-5:]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('note', '')}")
            else:
                lines.append(f"- {item}")

    lines.append("")
    lines.append("[状态规则]")
    lines.append("- 用户核心身份始终是昆特上人。")
    lines.append("- 默认所在地是UG宗门内；如果没有其他场景，按UG宗内的安全日常处理。")
    lines.append("- 用户说“我是xxx”或“我叫xxx”时，只代表化名，不改变昆特上人的身份。")
    lines.append("- 如果用户说回到宗门、回UG宗、返回山门，当前地点应视为UG宗门内。")
    lines.append("- 如果场景是否激活为 False，不要延续旧的外部险地场景。")
    lines.append("- 如果当前场景为UG宗门内，不要主动提风雪、寒气、秘境、妖兽、传送。")
    lines.append("- 如果场景激活，可以结合地点、危险等级、当前目标和场景摘要回答。")
    lines.append("- 只回应用户当前这句话，不要客服式寒暄，不要开放式收尾。")

    return "\n".join(lines)


def format_scene_view(state):
    lines = []
    lines.append("====== 当前场景状态 ======")
    lines.append(f"模式：{state.get('mode')}")
    lines.append(f"场景激活：{state.get('scene_active')}")
    lines.append(f"当前场景：{state.get('current_scene') or DEFAULT_HOME_SCENE}")
    lines.append(f"地点：{state.get('location') or DEFAULT_HOME_LOCATION}")
    lines.append(f"危险等级：{state.get('danger_level') or '未知'}")
    lines.append(f"当前目标：{state.get('objective') or '无'}")
    lines.append(f"场景摘要：{state.get('scene_summary') or '无'}")
    lines.append(f"场景轮数：{state.get('scene_turn_count', 0)}")
    lines.append(f"更新时间：{state.get('updated_at') or '未知'}")

    notes = state.get("scene_notes", [])
    if notes:
        lines.append("")
        lines.append("场景笔记：")
        for item in notes[-10:]:
            if isinstance(item, dict):
                lines.append(f"- [{item.get('time', '?')}] {item.get('note', '')}")
            else:
                lines.append(f"- {item}")

    lines.append("=========================")
    return "\n".join(lines)
