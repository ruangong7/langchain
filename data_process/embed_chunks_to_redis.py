from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import redis
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
MAX_EMBED_BATCH_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 JSONL 的 text 字段做向量化并按 corpus_chunk_index 写入 Redis。"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(_PROJECT_ROOT, "content", "gongzhonghao_text_chunks.jsonl"),
        help="输入 JSONL 文件路径。",
    )
    parser.add_argument(
        "--redis-host",
        default=REDIS_HOST,
        help="Redis 主机地址。",
    )
    parser.add_argument(
        "--redis-port",
        default=REDIS_PORT,
        type=int,
        help="Redis 端口。",
    )
    parser.add_argument(
        "--redis-db",
        default=REDIS_DB,
        type=int,
        help="Redis 数据库编号。",
    )
    parser.add_argument(
        "--key-prefix",
        default="qwen3:",
        help="文档 key 前缀，最终 key 形如 prefix + corpus_chunk_index。",
    )
    parser.add_argument(
        "--batch-size",
        default=10,
        type=int,
        help="Embedding 批大小（DashScope 兼容接口最大 10）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖同名文档（默认跳过已存在 key）。",
    )
    parser.add_argument(
        "--index-name",
        default="drug_vectors",
        help="RedisVectorStore 索引名。",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc

            if "corpus_chunk_index" not in row:
                raise ValueError(f"第 {line_no} 行缺少 corpus_chunk_index 字段")
            if "text" not in row:
                raise ValueError(f"第 {line_no} 行缺少 text 字段")
            rows.append(row)
    return rows


def prepare_pending_rows(
    rds: redis.Redis,
    rows: List[Dict[str, Any]],
    key_prefix: str,
    overwrite: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    pending: List[Tuple[str, Dict[str, Any]]] = []
    for row in rows:
        key = f"{key_prefix}{row['corpus_chunk_index']}"
        if not overwrite and rds.exists(key):
            continue
        pending.append((key, row))
    return pending


def _vectorstore_add_with_custom_ids(
    vectorstore: RedisVectorStore,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    custom_ids: List[str],
) -> None:
    """
    兼容不同版本 RedisVectorStore 的参数命名差异（ids/keys）。
    传入的 custom_ids 为完整 Redis key，因此需要去掉 key_prefix，
    避免 vectorstore 再自动拼接一次前缀。
    """
    prefix = f"{vectorstore.key_prefix}:"
    normalized_ids = [
        custom_id[len(prefix) :] if custom_id.startswith(prefix) else custom_id
        for custom_id in custom_ids
    ]
    try:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=normalized_ids)  # type: ignore[arg-type]
        return
    except TypeError:
        pass

    try:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, keys=normalized_ids)  # type: ignore[arg-type]
        return
    except TypeError as exc:
        raise RuntimeError("当前 RedisVectorStore 版本不支持自定义文档 ID（ids/keys）。") from exc


def write_batch(
    vectorstore: RedisVectorStore,
    embeddings: OpenAIEmbeddings,
    key_row_pairs: List[Tuple[str, Dict[str, Any]]],
    overwrite: bool,
) -> None:
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    custom_ids: List[str] = []

    # 保留“只向量化 text”要求：向量输入仅来自 text，其他字段全部进入 metadata。
    for key, row in key_row_pairs:
        texts.append(str(row["text"]))
        metadata: Dict[str, Any] = {"corpus_chunk_index": int(row["corpus_chunk_index"])}
        for field in ("source", "source_file", "chunk_index", "chunk_count", "chunk_id"):
            if field in row:
                metadata[field] = row[field]
        metadata["legacy_key"] = key
        metadatas.append(metadata)
        custom_ids.append(key)

    # 先执行一遍 embedding 做提前失败（例如维度/接口问题），避免半批次写入。
    embeddings.embed_documents(texts)

    if overwrite:
        _vectorstore_add_with_custom_ids(
            vectorstore=vectorstore,
            texts=texts,
            metadatas=metadatas,
            custom_ids=custom_ids,
        )
        return

    # 不覆盖模式：逐条判断是否存在（prepare_pending_rows 已做过一次，这里再兜底一层）
    # 这里不依赖内部 schema，直接再次按 custom_ids 补写。
    _vectorstore_add_with_custom_ids(
        vectorstore=vectorstore,
        texts=texts,
        metadatas=metadatas,
        custom_ids=custom_ids,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.batch_size > MAX_EMBED_BATCH_SIZE:
        logger.warning(
            "检测到 --batch-size=%d 超过 DashScope 上限，已自动降为 %d",
            args.batch_size,
            MAX_EMBED_BATCH_SIZE,
        )
        args.batch_size = MAX_EMBED_BATCH_SIZE

    rows = load_jsonl(args.input)
    logger.info("读取完成: 共 %d 条", len(rows))

    rds = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )
    rds.ping()
    logger.info(
        "Redis 连接成功: host=%s port=%d db=%d",
        args.redis_host,
        args.redis_port,
        args.redis_db,
    )

    redis_url = f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}"

    pending = prepare_pending_rows(
        rds=rds,
        rows=rows,
        key_prefix=args.key_prefix,
        overwrite=args.overwrite,
    )
    if not pending:
        logger.info("没有需要写入的记录（可能都已存在）。")
        return

    logger.info("待向量化并写入: %d 条", len(pending))
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        # DashScope 兼容接口要求 input 是字符串（或字符串列表），
        # 关闭 langchain_openai 的 token 级长度安全分片，避免传入 token id 列表导致 400。
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
    )
    vectorstore = RedisVectorStore(
        redis_url=redis_url,
        index_name=args.index_name,
        embedding=embeddings,
        key_prefix=args.key_prefix.rstrip(":"),
    )

    total = len(pending)
    for i in range(0, total, args.batch_size):
        batch = pending[i : i + args.batch_size]
        write_batch(
            vectorstore=vectorstore,
            embeddings=embeddings,
            key_row_pairs=batch,
            overwrite=args.overwrite,
        )
        logger.info("进度: %d/%d", min(i + args.batch_size, total), total)

    logger.info(
        "完成写入，共 %d 条。index_name=%s, key_prefix=%s",
        total,
        args.index_name,
        args.key_prefix,
    )


if __name__ == "__main__":
    main()
