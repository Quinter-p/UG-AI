# -*- coding: utf-8 -*-
from storage.json_store import read_json, write_json


DEFAULT_EMOTION = {
    "created_at": "",
    "updated_at": "",
    "mood": "calm",
    "mood_text": "平静",
    "energy": 70,
    "affection": 50,
    "alertness": 30,
    "last_event": "",
    "turn_count": 0
}


MOOD_TEXT = {
    "calm": "平静",
    "happy": "开心",
    "warm": "亲近",
    "alert": "警觉",
    "annoyed": "不爽",
    "hurt": "委屈",
    "focused": "专注",
    "tired": "疲倦",
}


def clamp(v, lo=0, hi=100):
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


class EmotionStore:
    def __init__(self, state_file):
        self.state_file = state_file

    def load(self):
        data = read_json(self.state_file, DEFAULT_EMOTION)
        for k, v in DEFAULT_EMOTION.items():
            data.setdefault(k, v)
        data["energy"] = clamp(data.get("energy", 70))
        data["affection"] = clamp(data.get("affection", 50))
        data["alertness"] = clamp(data.get("alertness", 30))
        data["mood_text"] = MOOD_TEXT.get(data.get("mood", "calm"), "平静")
        return data

    def save(self, data):
        write_json(self.state_file, data)

    def reset(self):
        data = dict(DEFAULT_EMOTION)
        self.save(data)
        return data

    def update_by_message(self, text, is_master=False):
        text = str(text or "")
        data = self.load()

        data["turn_count"] = int(data.get("turn_count", 0)) + 1
        data["energy"] = clamp(data.get("energy", 70) - 1)
        data["last_event"] = text[:80]

        danger_words = ["危险", "敌人", "黑沐", "攻击", "入侵", "杀", "打", "偷袭"]
        happy_words = ["哈哈", "好玩", "笑死", "有趣"]
        warm_words = ["谢谢", "辛苦", "可爱", "喜欢", "乖", "陪我", "不错"]
        annoyed_words = ["笨", "傻", "闭嘴", "烦", "讨厌"]
        focus_words = ["分析", "计划", "代码", "修复", "设计", "推演"]

        if any(w in text for w in danger_words):
            data["mood"] = "alert"
            data["alertness"] = clamp(data.get("alertness", 30) + 20)
        elif any(w in text for w in annoyed_words):
            data["mood"] = "annoyed"
            data["affection"] = clamp(data.get("affection", 50) - 3)
        elif any(w in text for w in warm_words):
            data["mood"] = "warm"
            data["affection"] = clamp(data.get("affection", 50) + 4)
            data["energy"] = clamp(data.get("energy", 70) + 2)
        elif any(w in text for w in happy_words):
            data["mood"] = "happy"
            data["affection"] = clamp(data.get("affection", 50) + 2)
        elif any(w in text for w in focus_words):
            data["mood"] = "focused"
        else:
            data["alertness"] = clamp(data.get("alertness", 30) - 2)
            if data.get("alertness", 30) < 25 and data.get("mood") in ["alert", "focused"]:
                data["mood"] = "calm"

        if is_master:
            data["affection"] = clamp(data.get("affection", 50) + 1)

        if data.get("energy", 70) < 20:
            data["mood"] = "tired"

        data["mood_text"] = MOOD_TEXT.get(data.get("mood", "calm"), "平静")
        self.save(data)
        return data

    def format_status(self):
        data = self.load()
        return (
            "【当前状态】\n"
            f"心情：{data.get('mood_text')} ({data.get('mood')})\n"
            f"精神：{data.get('energy')}/100\n"
            f"亲近度：{data.get('affection')}/100\n"
            f"警觉度：{data.get('alertness')}/100\n"
            f"累计对话：{data.get('turn_count', 0)}\n"
            f"最近事件：{data.get('last_event') or '无'}"
        )
