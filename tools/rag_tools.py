# tools/rag_tools.py

import hashlib
import json
import math
import os
from datetime import datetime

from tools.file_tools import read_text_file


RAG_ALLOWED_EXT = [".txt", ".md", ".json", ".csv", ".py", ".lsf", ".m"]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def file_hash(text):
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_text(text, chunk_size=700, chunk_overlap=120):
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        chunk_size = 700

    if chunk_overlap < 0:
        chunk_overlap = 0

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = end - chunk_overlap

    return chunks


def collect_knowledge_files(knowledge_dir):
    files = []

    if not os.path.exists(knowledge_dir):
        return files

    for root, dirs, filenames in os.walk(knowledge_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()

            if ext not in RAG_ALLOWED_EXT:
                continue

            path = os.path.normpath(os.path.join(root, filename))
            files.append(path)

    files.sort()
    return files


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def scan_chunks_before_embedding(
    knowledge_dir="knowledge",
    chunk_size=700,
    chunk_overlap=120,
    max_read_chars=20000
):
    paths = collect_knowledge_files(knowledge_dir)

    scanned = []
    total_chunks = 0

    for path in paths:
        content, error = read_text_file(path, max_read_chars=max_read_chars)

        if error or not content:
            continue

        chunks = chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            continue

        source_hash = file_hash(content)

        scanned.append({
            "path": path,
            "content": content,
            "chunks": chunks,
            "source_hash": source_hash
        })

        total_chunks += len(chunks)

    return scanned, total_chunks


def empty_index(
    embedding_model="nomic-embed-text",
    knowledge_dir="knowledge",
    chunk_size=700,
    chunk_overlap=120
):
    return {
        "created_at": now_str(),
        "updated_at": now_str(),
        "embedding_model": embedding_model,
        "knowledge_dir": knowledge_dir,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": {},
        "chunks": []
    }


def save_index(index, index_file):
    ensure_parent_dir(index_file)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def load_rag_index(index_file="memory/rag_index.json"):
    if not os.path.exists(index_file):
        return None

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_index_compatible(index, embedding_model, knowledge_dir, chunk_size, chunk_overlap):
    if not index:
        return False

    if index.get("embedding_model") != embedding_model:
        return False

    if index.get("chunk_size") != chunk_size:
        return False

    if index.get("chunk_overlap") != chunk_overlap:
        return False

    # knowledge_dir 不强制比较，因为用户可能用相对路径/绝对路径混用。
    return True


def embed_file_chunks(
    llm_client,
    path,
    chunks,
    source_hash,
    embedding_model,
    file_idx,
    total_files,
    start_done_chunks,
    total_chunks
):
    embedded = []
    done_chunks = start_done_chunks

    print(f"\n[RAG] 文件 {file_idx}/{total_files}：{path}", flush=True)

    for chunk_idx, chunk in enumerate(chunks, 1):
        done_chunks += 1
        percent = done_chunks / total_chunks * 100 if total_chunks else 100.0

        print(
            f"[RAG] 进度 {done_chunks}/{total_chunks} "
            f"({percent:.1f}%) | 当前文件块 {chunk_idx}/{len(chunks)}",
            flush=True
        )

        chunk_id = hashlib.md5(
            f"{path}-{source_hash}-{chunk_idx}-{chunk}".encode("utf-8", errors="ignore")
        ).hexdigest()

        embedding = llm_client.embed_text(
            chunk,
            embedding_model=embedding_model
        )

        embedded.append({
            "id": chunk_id,
            "source": path,
            "source_hash": source_hash,
            "chunk_index": chunk_idx - 1,
            "text": chunk,
            "embedding": embedding
        })

    return embedded, done_chunks


def build_rag_index(
    llm_client,
    knowledge_dir="knowledge",
    index_file="memory/rag_index.json",
    embedding_model="nomic-embed-text",
    chunk_size=700,
    chunk_overlap=120,
    max_read_chars=20000
):
    """
    全量重建索引。
    会重新扫描 knowledge_dir，并重新生成所有 embedding。
    """
    ensure_parent_dir(index_file)

    print("[RAG] 正在扫描 knowledge 文件夹...", flush=True)

    scanned_files, total_chunks = scan_chunks_before_embedding(
        knowledge_dir=knowledge_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_read_chars=max_read_chars
    )

    if not scanned_files:
        index = empty_index(
            embedding_model=embedding_model,
            knowledge_dir=knowledge_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        save_index(index, index_file)

        print("[RAG] 没有找到可索引的知识库文件。", flush=True)
        return {"files": 0, "chunks": 0, "index_file": index_file}

    print(f"[RAG] 扫描完成：{len(scanned_files)} 个文件，{total_chunks} 个知识块。", flush=True)
    print(f"[RAG] 开始全量生成 embedding，模型：{embedding_model}", flush=True)

    all_chunks = []
    files_meta = {}
    done_chunks = 0

    for file_idx, item in enumerate(scanned_files, 1):
        embedded, done_chunks = embed_file_chunks(
            llm_client=llm_client,
            path=item["path"],
            chunks=item["chunks"],
            source_hash=item["source_hash"],
            embedding_model=embedding_model,
            file_idx=file_idx,
            total_files=len(scanned_files),
            start_done_chunks=done_chunks,
            total_chunks=total_chunks
        )

        all_chunks.extend(embedded)

        files_meta[item["path"]] = {
            "hash": item["source_hash"],
            "chunks": len(item["chunks"]),
            "updated_at": now_str()
        }

    index = {
        "created_at": now_str(),
        "updated_at": now_str(),
        "embedding_model": embedding_model,
        "knowledge_dir": knowledge_dir,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": files_meta,
        "chunks": all_chunks
    }

    print(f"\n[RAG] 正在写入索引文件：{index_file}", flush=True)
    save_index(index, index_file)
    print("[RAG] 索引写入完成。", flush=True)

    return {
        "mode": "full_rebuild",
        "files": len(scanned_files),
        "chunks": len(all_chunks),
        "index_file": index_file
    }


def update_rag_index(
    llm_client,
    knowledge_dir="knowledge",
    index_file="memory/rag_index.json",
    embedding_model="nomic-embed-text",
    chunk_size=700,
    chunk_overlap=120,
    max_read_chars=20000
):
    """
    增量更新索引。
    只重新处理新增/修改过的文件；没变的文件复用旧 embedding。
    如果没有旧索引，或参数不兼容，会自动全量构建。
    """
    print("[RAG] 正在检查索引与 knowledge 文件变化...", flush=True)

    old_index = load_rag_index(index_file)

    if not is_index_compatible(
        old_index,
        embedding_model=embedding_model,
        knowledge_dir=knowledge_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ):
        print("[RAG] 没有可复用索引，或 RAG 参数已变化，转为全量构建。", flush=True)
        return build_rag_index(
            llm_client=llm_client,
            knowledge_dir=knowledge_dir,
            index_file=index_file,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_read_chars=max_read_chars
        )

    scanned_files, _ = scan_chunks_before_embedding(
        knowledge_dir=knowledge_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_read_chars=max_read_chars
    )

    current_paths = set(item["path"] for item in scanned_files)
    old_files_meta = old_index.get("files", {})
    old_chunks = old_index.get("chunks", [])

    # 兼容旧版索引：如果旧 chunk 没有 source_hash，就不能安全复用，转全量。
    if old_chunks and any("source_hash" not in chunk for chunk in old_chunks):
        print("[RAG] 检测到旧版索引缺少 source_hash，转为全量构建。", flush=True)
        return build_rag_index(
            llm_client=llm_client,
            knowledge_dir=knowledge_dir,
            index_file=index_file,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_read_chars=max_read_chars
        )

    changed_items = []
    unchanged_paths = set()

    for item in scanned_files:
        path = item["path"]
        source_hash = item["source_hash"]
        old_meta = old_files_meta.get(path)

        if old_meta and old_meta.get("hash") == source_hash:
            unchanged_paths.add(path)
        else:
            changed_items.append(item)

    removed_paths = sorted(set(old_files_meta.keys()) - current_paths)

    kept_chunks = [
        chunk for chunk in old_chunks
        if chunk.get("source") in unchanged_paths
    ]

    total_new_chunks = sum(len(item["chunks"]) for item in changed_items)

    print(
        f"[RAG] 文件变化检查完成：保留 {len(unchanged_paths)} 个文件，"
        f"新增/修改 {len(changed_items)} 个文件，删除 {len(removed_paths)} 个文件。",
        flush=True
    )

    if not changed_items and not removed_paths:
        old_index["updated_at"] = now_str()
        save_index(old_index, index_file)

        print("[RAG] knowledge 没有变化，无需重新生成 embedding。", flush=True)

        return {
            "mode": "incremental_update",
            "changed_files": 0,
            "removed_files": 0,
            "kept_chunks": len(kept_chunks),
            "new_chunks": 0,
            "total_chunks": len(old_chunks),
            "index_file": index_file
        }

    new_chunks = []
    done_chunks = 0

    if changed_items:
        print(f"[RAG] 开始为 {len(changed_items)} 个新增/修改文件生成 embedding。", flush=True)

    for file_idx, item in enumerate(changed_items, 1):
        embedded, done_chunks = embed_file_chunks(
            llm_client=llm_client,
            path=item["path"],
            chunks=item["chunks"],
            source_hash=item["source_hash"],
            embedding_model=embedding_model,
            file_idx=file_idx,
            total_files=len(changed_items),
            start_done_chunks=done_chunks,
            total_chunks=total_new_chunks
        )
        new_chunks.extend(embedded)

    new_files_meta = {}

    # 保留没变文件的 metadata
    for path in unchanged_paths:
        if path in old_files_meta:
            new_files_meta[path] = old_files_meta[path]

    # 写入新增/修改文件的 metadata
    for item in changed_items:
        new_files_meta[item["path"]] = {
            "hash": item["source_hash"],
            "chunks": len(item["chunks"]),
            "updated_at": now_str()
        }

    updated_index = {
        "created_at": old_index.get("created_at", now_str()),
        "updated_at": now_str(),
        "embedding_model": embedding_model,
        "knowledge_dir": knowledge_dir,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": new_files_meta,
        "chunks": kept_chunks + new_chunks
    }

    print(f"\n[RAG] 正在写入更新后的索引文件：{index_file}", flush=True)
    save_index(updated_index, index_file)
    print("[RAG] 增量索引更新完成。", flush=True)

    return {
        "mode": "incremental_update",
        "changed_files": len(changed_items),
        "removed_files": len(removed_paths),
        "kept_chunks": len(kept_chunks),
        "new_chunks": len(new_chunks),
        "total_chunks": len(updated_index["chunks"]),
        "index_file": index_file
    }


def rag_status(index_file="memory/rag_index.json"):
    index = load_rag_index(index_file)

    if not index:
        return f"RAG 索引不存在：{index_file}"

    chunks = index.get("chunks", [])
    files_meta = index.get("files", {})
    sources = sorted(set(item.get("source", "") for item in chunks))

    lines = []
    lines.append("====== RAG 索引状态 ======")
    lines.append(f"索引文件：{index_file}")
    lines.append(f"创建时间：{index.get('created_at', '未知')}")
    lines.append(f"更新时间：{index.get('updated_at', '未知')}")
    lines.append(f"Embedding 模型：{index.get('embedding_model', '未知')}")
    lines.append(f"知识库目录：{index.get('knowledge_dir', '未知')}")
    lines.append(f"文件数：{len(files_meta) if files_meta else len(sources)}")
    lines.append(f"切块数：{len(chunks)}")
    lines.append(f"切块大小：{index.get('chunk_size', '未知')}")
    lines.append(f"切块重叠：{index.get('chunk_overlap', '未知')}")

    if files_meta:
        lines.append("")
        lines.append("来源文件：")
        for source in sorted(files_meta.keys())[:20]:
            meta = files_meta[source]
            lines.append(
                f"- {source} | chunks={meta.get('chunks', '?')} | updated={meta.get('updated_at', '?')}"
            )

        if len(files_meta) > 20:
            lines.append(f"... 还有 {len(files_meta) - 20} 个文件")
    elif sources:
        lines.append("")
        lines.append("来源文件：")
        for source in sources[:20]:
            lines.append(f"- {source}")

        if len(sources) > 20:
            lines.append(f"... 还有 {len(sources) - 20} 个文件")

    lines.append("=========================")

    return "\n".join(lines)


def search_rag(
    query,
    llm_client,
    index_file="memory/rag_index.json",
    embedding_model="nomic-embed-text",
    top_k=4,
    min_score=0.25
):
    index = load_rag_index(index_file)

    if not index:
        return []

    chunks = index.get("chunks", [])

    if not chunks:
        return []

    query_embedding = llm_client.embed_text(
        query,
        embedding_model=embedding_model
    )

    scored = []

    for item in chunks:
        score = cosine_similarity(query_embedding, item.get("embedding", []))

        if score >= min_score:
            scored.append({
                "score": score,
                "source": item.get("source", ""),
                "chunk_index": item.get("chunk_index", 0),
                "text": item.get("text", "")
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]


def format_rag_results(results, max_chars_per_chunk=700):
    if not results:
        return ""

    lines = ["[知识库检索结果]"]

    for i, item in enumerate(results, 1):
        text = item["text"].strip()

        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        lines.append("")
        lines.append(f"片段 {i}")
        lines.append(f"来源：{item['source']}")
        lines.append(f"相似度：{item['score']:.3f}")
        lines.append(text)

    return "\n".join(lines)


def format_rag_context_for_prompt(results, max_chars_per_chunk=700):
    if not results:
        return "[知识库检索结果]\n无相关片段。"

    return format_rag_results(results, max_chars_per_chunk=max_chars_per_chunk)
