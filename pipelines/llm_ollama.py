# pipelines/llm_ollama.py

import json
import requests


class OllamaClient:
    """
    专门负责和 Ollama 通信。
    version0.03.py 不再直接处理 requests 细节。
    """

    def __init__(
        self,
        model_name="qwen2.5:7b",
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        num_predict=512,
        keep_alive="10m"
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.embeddings_url = f"{self.base_url}/api/embeddings"

        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive

        self.session = requests.Session()
        self.session.trust_env = False

    def generate_once(self, prompt, temperature=0.7, timeout=180):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict
            }
        }

        response = self.session.post(
            self.generate_url,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()
        data = response.json()

        return data.get("response", "").strip()

    def generate_stream(self, prompt, temperature=0.35, timeout=180):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict
            }
        }

        response = self.session.post(
            self.generate_url,
            json=payload,
            stream=True,
            timeout=timeout
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

    def chat_stream(self, messages, temperature=0.35, timeout=180):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict
            }
        }

        response = self.session.post(
            self.chat_url,
            json=payload,
            stream=True,
            timeout=timeout
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

    def embed_text(self, text, embedding_model="nomic-embed-text", timeout=180):
        """
        使用 Ollama embedding 模型生成向量。
        默认需要先运行：
            ollama pull nomic-embed-text
        """
        payload = {
            "model": embedding_model,
            "prompt": text
        }

        response = self.session.post(
            self.embeddings_url,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding")

        if not embedding:
            raise RuntimeError(
                f"Ollama 没有返回 embedding。请确认模型已安装：ollama pull {embedding_model}"
            )

        return embedding
