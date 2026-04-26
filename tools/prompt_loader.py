from pathlib import Path


KNOWLEDGE_DIR = Path("knowledge")

PINNED_PROMPT_FILES = [
    "00_system_prompt.md",
    "00_identity.md",
]


def read_text_file(path: Path) -> str:
    if not path.exists():
        print(f"[WARNING] Prompt file not found: {path}")
        return ""

    return path.read_text(encoding="utf-8")


def load_pinned_system_prompt() -> str:
    """
    固定加载到 system prompt 的内容。
    这些内容不依赖向量检索，每次对话都会生效。
    """
    parts = []

    for filename in PINNED_PROMPT_FILES:
        file_path = KNOWLEDGE_DIR / filename
        content = read_text_file(file_path)

        if content.strip():
            parts.append(f"\n\n# 来自文件：{filename}\n{content}")

    return "\n".join(parts).strip()