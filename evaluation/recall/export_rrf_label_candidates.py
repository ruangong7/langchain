from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import jieba
import pandas as pd
import redis
from langchain_community.vectorstores import Redis
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

RRF_K_DEFAULT = 60.0


def find_project_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "config.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("未找到 config.py，无法定位工程根目录。")
        d = parent


PROJECT_ROOT = find_project_root()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import REDIS_DB, REDIS_HOST, REDIS_PORT, VECTOR_INDEX_NAME, VECTOR_KEY_PREFIX  # noqa: E402
from services.embedding_factory import build_embeddings  # noqa: E402
from services.rag_service import RAGService, document_redis_key  # noqa: E402


def _redis_scan_match(key_prefix: str) -> str:
    p = key_prefix.strip()
    if "*" in p:
        return p
    return f"{p}*"


def read_corpus_from_redis(
    host: str,
    port: int,
    db: int,
    key_prefix: str,
) -> Tuple[List[str], List[str]]:
    rds = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    rds.ping()
    texts: List[str] = []
    keys: List[str] = []
    for key in sorted(rds.scan_iter(match=_redis_scan_match(key_prefix))):
        row = rds.hmget(key, "content", "text")
        text = ((row[0] or "").strip() if row and len(row) > 0 else "") or ((row[1] or "").strip() if row and len(row) > 1 else "")
        if not text:
            continue
        texts.append(text)
        keys.append(key)
    if not texts:
        raise RuntimeError(f"Redis 中未找到前缀 {key_prefix!r} 的有效片段")
    logger.info("BM25 语料已加载: %d 条", len(texts))
    return texts, keys


def tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


def rrf_scores(
    ranked_lists: Sequence[Sequence[str]],
    rrf_k: float,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for idx, doc_id in enumerate(lst):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + idx + 1)
    return scores


def build_dense_rag_service(top_k: int, index_name: str, redis_host: str, redis_port: int, redis_db: int) -> RAGService:
    embeddings = build_embeddings()
    vectorstore = Redis(
        redis_url=f"redis://{redis_host}:{redis_port}/{redis_db}",
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return RAGService(retriever, title_index={})


def fetch_doc_meta(rds: redis.Redis, redis_key: str) -> Dict[str, str]:
    row = rds.hmget(
        redis_key,
        "source",
        "source_type",
        "chunk_id",
        "chunk_index",
        "corpus_chunk_index",
        "content",
        "text",
    )
    fields = (
        "source",
        "source_type",
        "chunk_id",
        "chunk_index",
        "corpus_chunk_index",
        "content",
        "text",
    )
    data = {field: value for field, value in zip(fields, row) if value is not None}
    text = (data.get("content") or "").strip() or (data.get("text") or "").strip()
    return {
        "source": str(data.get("source") or ""),
        "source_type": str(data.get("source_type") or ""),
        "chunk_id": str(data.get("chunk_id") or ""),
        "chunk_index": str(data.get("chunk_index") or ""),
        "corpus_chunk_index": str(data.get("corpus_chunk_index") or ""),
        "content": text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出人工标注用的 RRF 候选列表")
    parser.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "evaluation", "recall", "test_data.multigold.template.csv"),
        help="输入 CSV，至少包含 question 列",
    )
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--branch-k", type=int, default=50, help="dense / bm25 各自先召回多少条")
    parser.add_argument("--rrf-k", type=float, default=RRF_K_DEFAULT)
    parser.add_argument("--export-top-n", type=int, default=20, help="导出前多少个 RRF 候选")
    parser.add_argument("--key-prefix", default=VECTOR_KEY_PREFIX)
    parser.add_argument("--index-name", default=VECTOR_INDEX_NAME)
    parser.add_argument("--redis-host", default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    parser.add_argument("--redis-db", type=int, default=REDIS_DB)
    parser.add_argument("--output", default="", help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(args.file)

    df = pd.read_csv(args.file)
    if args.question_col not in df.columns:
        raise ValueError(f"CSV 需含列: {args.question_col}")

    questions = df[args.question_col].astype(str).tolist()

    rds = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)
    rds.ping()

    corpus_texts, corpus_keys = read_corpus_from_redis(
        args.redis_host, args.redis_port, args.redis_db, args.key_prefix
    )
    bm25 = BM25Okapi([tokenize(text) for text in corpus_texts])
    rag = build_dense_rag_service(
        top_k=args.branch_k,
        index_name=args.index_name,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )

    rows: List[Dict[str, object]] = []
    for question_index, question in enumerate(questions, start=1):
        logger.info("导出候选 [%d/%d]", question_index, len(questions))

        dense_docs = rag.retrieve_documents(question)
        dense_ids = [document_redis_key(doc) for doc in dense_docs if document_redis_key(doc)][: args.branch_k]
        dense_rank_map = {doc_id: idx for idx, doc_id in enumerate(dense_ids, start=1)}

        scores = bm25.get_scores(tokenize(question))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.branch_k]
        bm25_ids = [corpus_keys[i] for i in top_indices]
        bm25_rank_map = {doc_id: idx for idx, doc_id in enumerate(bm25_ids, start=1)}

        fused_scores = rrf_scores([dense_ids, bm25_ids], args.rrf_k)
        fused_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[: args.export_top_n]

        for rrf_rank, redis_key in enumerate(fused_ids, start=1):
            meta = fetch_doc_meta(rds, redis_key)
            retrieval_sources = []
            if redis_key in dense_rank_map:
                retrieval_sources.append("dense")
            if redis_key in bm25_rank_map:
                retrieval_sources.append("sparse")
            rows.append(
                {
                    "question_index": question_index,
                    "question": question,
                    "rrf_rank": rrf_rank,
                    "rrf_score": fused_scores.get(redis_key, 0.0),
                    "redis_key": redis_key,
                    "chunk_id": meta["chunk_id"],
                    "chunk_index": meta["chunk_index"],
                    "corpus_chunk_index": meta["corpus_chunk_index"],
                    "source": meta["source"],
                    "source_type": meta["source_type"],
                    "retrieval_sources": "|".join(retrieval_sources),
                    "dense_rank": dense_rank_map.get(redis_key, ""),
                    "bm25_rank": bm25_rank_map.get(redis_key, ""),
                    "content": meta["content"],
                }
            )

    output_path = args.output or os.path.join(
        SCRIPT_DIR, f"rrf_label_candidates_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_index",
                "question",
                "rrf_rank",
                "rrf_score",
                "redis_key",
                "chunk_id",
                "chunk_index",
                "corpus_chunk_index",
                "source",
                "source_type",
                "retrieval_sources",
                "dense_rank",
                "bm25_rank",
                "content",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"已导出 RRF 候选标注文件: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
