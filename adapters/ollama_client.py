import requests


class OllamaClient:
    def __init__(self, config):
        cfg = config.get("ollama", {})
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.model = str(cfg.get("model", "qwen3:8b"))
        self.temperature = float(cfg.get("temperature", 0.45))
        self.top_p = float(cfg.get("top_p", 0.85))
        self.num_ctx = int(cfg.get("num_ctx", 4096))
        self.num_predict = int(cfg.get("num_predict", 384))
        self.think = bool(cfg.get("think", False))
        self.keep_alive = cfg.get("keep_alive", "5m")
        self.timeout = int(cfg.get("timeout", 240))

    def chat(self, system_prompt, user_text, timeout=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.chat_messages(messages, timeout=timeout)

    def chat_messages(self, messages, timeout=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        url = f"{self.base_url}/api/chat"
        response = requests.post(url, json=payload, timeout=timeout or self.timeout)
        response.raise_for_status()
        data = response.json()

        message = data.get("message") or {}
        content = message.get("content") or data.get("response") or ""

        if not str(content).strip():
            content = "主人，本喵刚才有些走神了。"

        return str(content).strip(), {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "load_duration": data.get("load_duration"),
            "total_duration": data.get("total_duration"),
        }
