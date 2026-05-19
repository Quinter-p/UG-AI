from pathlib import Path
import re
from core.config import project_path


def normalize_text(text):
    text = str(text or "").replace("\r", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def tokenize(text):
    text = str(text or "").lower()
    words = set(re.findall(r"[a-zA-Z0-9_]+", text))
    chars = set(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    return words | chars


def score_doc(filename, content, query):
    q = tokenize(query)
    if not q:
        return 0

    f = tokenize(filename)
    c = tokenize(content)

    return 3 * len(q & f) + len(q & c)


def normalize_filename(name):
    return str(name or "").strip().replace("\\", "/").split("/")[-1]


class WorldLoreStore:
    """
    世界观设定读取器。

    支持两种加载：
    1. force_files：relationship 明确绑定的 lore 文件，必定加载
    2. query retrieval：根据当前消息和 lore_keys 做相关性检索
    """

    def __init__(self, lore_dir="knowledge", max_chars=3500):
        self.lore_dir = lore_dir
        self.max_chars = int(max_chars)

    def ensure_dir(self):
        path = project_path(self.lore_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_files(self):
        path = self.ensure_dir()
        files = []
        for pattern in ["*.md", "*.txt"]:
            files.extend(path.glob(pattern))
        return sorted(files, key=lambda p: p.name.lower())

    def read_file(self, path):
        try:
            content = normalize_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            content = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

        if not content:
            return None

        return {
            "filename": path.name,
            "content": content,
            "source": "file",
        }

    def read_all(self):
        items = []
        for path in self.list_files():
            item = self.read_file(path)
            if item:
                items.append(item)
        return items

    def read_forced_files(self, force_files):
        wanted = [normalize_filename(x) for x in (force_files or []) if normalize_filename(x)]
        if not wanted:
            return []

        by_name = {path.name: path for path in self.list_files()}
        selected = []

        for name in wanted:
            path = by_name.get(name)
            if not path:
                continue
            item = self.read_file(path)
            if item:
                item["source"] = "forced_relationship_file"
                selected.append(item)

        return selected

    def retrieve(self, query, limit=6, force_files=None, force_keys=None):
        force_files = force_files or []
        force_keys = force_keys or []

        forced = self.read_forced_files(force_files)
        forced_names = set(item["filename"] for item in forced)

        # lore_keys 加入 query，增强稳定关联。
        augmented_query = " ".join([str(query or "")] + [str(x) for x in force_keys or []])

        items = [
            item for item in self.read_all()
            if item.get("filename") not in forced_names
        ]

        scored = [
            (score_doc(item["filename"], item["content"], augmented_query), item)
            for item in items
        ]
        scored.sort(key=lambda x: (x[0], x[1]["filename"]), reverse=True)

        remaining_limit = max(0, int(limit) - len(forced))
        retrieved = [item for score, item in scored[:remaining_limit] if score > 0]

        if not retrieved and remaining_limit > 0:
            retrieved = [item for score, item in scored[:min(remaining_limit, 2)]]

        # forced 文件永远排在前面。
        return forced + retrieved

    def format_for_prompt(self, items):
        if not items:
            return "暂无世界观设定。"

        blocks = []
        used = 0

        for item in items:
            filename = item.get("filename", "")
            source = item.get("source", "retrieved")
            content = item.get("content", "")

            block = f"【{filename} | {source}】\n{content}".strip()

            if used + len(block) > self.max_chars:
                remain = self.max_chars - used
                if remain <= 100:
                    break
                block = block[:remain].rstrip() + "\n..."

            blocks.append(block)
            used += len(block)

            if used >= self.max_chars:
                break

        return "\n\n".join(blocks)

    def summary(self):
        files = self.list_files()

        if not files:
            return "knowledge 文件夹里暂无 .md 或 .txt 世界观文件。"

        lines = ["【世界观文件】"]

        for path in files:
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            lines.append(f"- {path.name} ({size} bytes)")

        return "\n".join(lines)
