from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
        row = rds.hmget(
            key,
            "content",
            "text",
            "source",
            "source_file",
            "corpus_chunk_index",
        )
        fields = ("content", "text", "source", "source_file", "corpus_chunk_index")
        data = {field: value for field, value in zip(fields, row) if value is not None}
        text = (data.get("content") or "").strip() or (data.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        keys.append(key)

    if not texts:
        raise RuntimeError(f"Redis 中未找到前缀 {key_prefix!r} 的有效片段")
    logger.info("从 Redis 加载 %d 条片段（prefix=%s）", len(texts), key_prefix)
    return texts, keys


def tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


def parse_multigold(raw_gold: object) -> List[str]:
    if raw_gold is None or (isinstance(raw_gold, float) and pd.isna(raw_gold)):
        return []
    text = str(raw_gold).strip()
    if not text:
        return []
    values: List[str] = []
    seen = set()
    for item in text.split("|"):
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            values.append(key)
    return values


def rrf_fuse_ids(
    ranked_lists: Sequence[Sequence[str]],
    rrf_k: float,
    final_top: int,
) -> List[str]:
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for i, doc_id in enumerate(lst):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + i + 1)
    ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ordered[:final_top]


def compute_metrics(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> Dict[str, object]:
    gold_set = set(gold_ids)
    matched = [doc_id for doc_id in retrieved_ids if doc_id in gold_set]
    matched_set = set(matched)
    best_rank = 0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            best_rank = idx
            break
    total_gold = len(gold_ids)
    recall = (len(matched_set) / total_gold) if total_gold else 0.0
    return {
        "hit": 1 if matched else 0,
        "best_rank": best_rank,
        "mrr": (1.0 / best_rank) if best_rank else 0.0,
        "recall": recall,
        "matched_ids": list(matched),
        "matched_count": len(matched_set),
        "gold_count": total_gold,
    }


def build_dense_rag_service(top_k: int, index_name: str, redis_host: str, redis_port: int, redis_db: int) -> RAGService:
    embeddings = build_embeddings()
    vectorstore = Redis(
        redis_url=f"redis://{redis_host}:{redis_port}/{redis_db}",
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return RAGService(retriever, title_index={})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 gold 检索评测（dense / bm25 / hybrid）")
    parser.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "evaluation", "recall", "test_data.multigold.template.csv"),
        help="测试集 CSV（需含 question 与 gold_redis_keys）",
    )
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--gold-col", default="gold_redis_keys")
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
        help="评测哪种检索器",
    )
    parser.add_argument("--top-k", type=int, default=5, help="最终评测 Top-K")
    parser.add_argument(
        "--branch-k",
        type=int,
        default=10,
        help="仅 hybrid 使用：dense / bm25 各自先召回多少条",
    )
    parser.add_argument("--rrf-k", type=float, default=RRF_K_DEFAULT, help="hybrid 模式的 RRF 平滑常数")
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
    if args.question_col not in df.columns or args.gold_col not in df.columns:
        raise ValueError(f"CSV 需含列: {args.question_col}, {args.gold_col}")

    questions = df[args.question_col].astype(str).tolist()
    gold_lists = [parse_multigold(value) for value in df[args.gold_col].tolist()]

    t0 = time.perf_counter()
    rag: RAGService | None = None
    bm25: BM25Okapi | None = None
    corpus_keys: List[str] = []

    if args.mode in {"dense", "hybrid"}:
        dense_top_k = args.branch_k if args.mode == "hybrid" else args.top_k
        rag = build_dense_rag_service(
            top_k=dense_top_k,
            index_name=args.index_name,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
        )
        logger.info("向量检索已就绪: top_k=%d index=%s", dense_top_k, args.index_name)

    if args.mode in {"bm25", "hybrid"}:
        corpus_texts, corpus_keys = read_corpus_from_redis(
            args.redis_host,
            args.redis_port,
            args.redis_db,
            args.key_prefix,
        )
        bm25 = BM25Okapi([tokenize(text) for text in corpus_texts])
        logger.info("BM25 索引已就绪: corpus=%d", len(corpus_keys))

    hits: List[int] = []
    best_ranks: List[int] = []
    mrr_values: List[float] = []
    recall_values: List[float] = []
    retrieved_flat: List[str] = []
    matched_flat: List[str] = []
    gold_count_values: List[int] = []
    matched_count_values: List[int] = []
    dense_flat: List[str] = []
    bm25_flat: List[str] = []

    nq = len(questions)
    for idx, (question, gold_ids) in enumerate(zip(questions, gold_lists), start=1):
        if idx == 1 or idx % 20 == 0 or idx == nq:
            logger.info("多 gold 评测 [%d/%d] mode=%s", idx, nq, args.mode)

        dense_ids: List[str] = []
        sparse_ids: List[str] = []

        if args.mode in {"dense", "hybrid"} and rag is not None:
            dense_docs = rag.retrieve_documents(question)
            dense_ids = [document_redis_key(doc) for doc in dense_docs if document_redis_key(doc)]

        if args.mode in {"bm25", "hybrid"} and bm25 is not None:
            scores = bm25.get_scores(tokenize(question))
            limit = args.branch_k if args.mode == "hybrid" else args.top_k
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            sparse_ids = [corpus_keys[i] for i in top_indices]

        if args.mode == "dense":
            retrieved_ids = dense_ids[: args.top_k]
        elif args.mode == "bm25":
            retrieved_ids = sparse_ids[: args.top_k]
        else:
            retrieved_ids = rrf_fuse_ids([dense_ids[: args.branch_k], sparse_ids[: args.branch_k]], args.rrf_k, args.top_k)

        metrics = compute_metrics(retrieved_ids, gold_ids)
        hits.append(int(metrics["hit"]))
        best_ranks.append(int(metrics["best_rank"]))
        mrr_values.append(float(metrics["mrr"]))
        recall_values.append(float(metrics["recall"]))
        gold_count_values.append(int(metrics["gold_count"]))
        matched_count_values.append(int(metrics["matched_count"]))
        retrieved_flat.append("|".join(retrieved_ids))
        matched_flat.append("|".join(metrics["matched_ids"]))  # type: ignore[arg-type]
        dense_flat.append("|".join(dense_ids))
        bm25_flat.append("|".join(sparse_ids))

    n = len(hits)
    hit_rate = sum(hits) / n if n else 0.0
    mean_mrr = sum(mrr_values) / n if n else 0.0
    mean_recall = sum(recall_values) / n if n else 0.0

    out = df.copy()
    out["gold_count"] = gold_count_values
    out["matched_gold_count"] = matched_count_values
    out[f"{args.mode}_multigold_hit@{args.top_k}"] = hits
    out[f"{args.mode}_multigold_best_rank_in_top{args.top_k}"] = best_ranks
    out[f"{args.mode}_multigold_mrr"] = mrr_values
    out[f"{args.mode}_multigold_recall@{args.top_k}"] = recall_values
    out[f"{args.mode}_retrieved_redis_keys"] = retrieved_flat
    out[f"{args.mode}_matched_gold_keys"] = matched_flat
    if args.mode == "hybrid":
        out[f"dense_top{args.branch_k}_redis_keys"] = dense_flat
        out[f"bm25_top{args.branch_k}_redis_keys"] = bm25_flat

    out_path = args.output or os.path.join(
        SCRIPT_DIR, f"{args.mode}_multigold_recall_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 多 Gold 检索评测: {args.mode} ===")
    if args.mode == "hybrid":
        print(
            f"branch_k={args.branch_k}（各路先取） -> RRF -> final top_k={args.top_k}  |  RRF_K={args.rrf_k}"
        )
    else:
        print(f"top_k={args.top_k}")
    print(f"样本数: {n}")
    print(f"Hit@{args.top_k}: {hit_rate:.4f} ({sum(hits)}/{n})")
    print(f"MRR: {mean_mrr:.4f}")
    print(f"Mean Recall@{args.top_k}: {mean_recall:.4f}")
    print(f"总耗时: {time.perf_counter() - t0:.2f} 秒")
    print(f"文件: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
