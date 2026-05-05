"""稠密 + BM25 混合检索评测：两路各取 branch_k 条，RRF 重排后只保留 final_k 条再算 Hit@K。

默认 final_k=5，与单路稠密 / BM25 评测的 Top-5 对齐，避免融合后仍取 10 条造成不公平。
RRF 与 hybrid_retriever 同公式；去重键为 redis_key。语料与 run_bm25_retrieval_recall 同源（Redis 片段）。
"""
from __future__ import annotations

import argparse
import importlib.util
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

from config import EMBEDDING_MODEL, REDIS_DB, REDIS_HOST, REDIS_PORT  # noqa: E402
from services.rag_service import RAGService, document_redis_key  # noqa: E402


def _redis_scan_match(key_prefix: str) -> str:
    p = key_prefix.strip()
    if "*" in p:
        return p
    return f"{p}*"


def read_corpus_from_redis(
    host: str, port: int, db: int, key_prefix: str
) -> Tuple[List[str], List[str]]:
    rds = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    rds.ping()
    texts: List[str] = []
    keys: List[str] = []
    for key in sorted(rds.scan_iter(match=_redis_scan_match(key_prefix))):
        row = rds.hmget(key, "content", "text", "source", "source_file", "corpus_chunk_index")
        fields = ("content", "text", "source", "source_file", "corpus_chunk_index")
        d = {f: v for f, v in zip(fields, row) if v is not None}
        body = (d.get("content") or "").strip() or (d.get("text") or "").strip()
        if not body:
            continue
        texts.append(body)
        keys.append(key)
    if not texts:
        raise RuntimeError(f"Redis 中未找到前缀 {key_prefix!r} 的有效片段")
    logger.info("BM25 语料: %d 条（prefix=%s）", len(texts), key_prefix)
    return texts, keys


def tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


def rrf_fuse_ids(
    ranked_lists: Sequence[Sequence[str]],
    rrf_k: float,
    final_top: int,
) -> List[str]:
    """多路有序 id 列表做 RRF，返回融合后 Top final_top（与 HybridRetriever 同公式）。"""
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for i, doc_id in enumerate(lst):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + i + 1)
    ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ordered[:final_top]


def load_embeddings():
    main_path = os.path.join(PROJECT_ROOT, "main.py")
    spec = importlib.util.spec_from_file_location("main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {main_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DashScopeEmbeddings(model=EMBEDDING_MODEL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="稠密+BM25 RRF 混合检索 Gold Hit@K")
    p.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "evaluation", "recall", "test_data.generated.csv"),
    )
    p.add_argument("--question-col", default="question")
    p.add_argument("--redis-key-col", default="redis_key")
    p.add_argument(
        "--branch-k",
        type=int,
        default=10,
        help="稠密 / BM25 各自先取的条数（默认 10）",
    )
    p.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="RRF 融合后截断为几条再算 Hit（默认 5，与单路 Top5 公平对比）",
    )
    p.add_argument(
        "--rrf-k",
        type=float,
        default=RRF_K_DEFAULT,
        help="RRF 平滑常数（与 hybrid_retriever.HybridRetriever.RRF_K 一致默认 60）",
    )
    p.add_argument("--key-prefix", default="qwen3:")
    p.add_argument("--index-name", default="drug_vectors")
    p.add_argument("--redis-host", default=REDIS_HOST)
    p.add_argument("--redis-port", type=int, default=REDIS_PORT)
    p.add_argument("--redis-db", type=int, default=REDIS_DB)
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(args.file)
    df = pd.read_csv(args.file)
    if args.question_col not in df.columns or args.redis_key_col not in df.columns:
        raise ValueError(f"CSV 需含: {args.question_col}, {args.redis_key_col}")

    t0 = time.perf_counter()
    corpus_texts, corpus_keys = read_corpus_from_redis(
        args.redis_host, args.redis_port, args.redis_db, args.key_prefix
    )
    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 索引完成，耗时 %.2fs", time.perf_counter() - t0)

    emb = load_embeddings()
    vs = Redis(
        redis_url=f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}",
        index_name=args.index_name,
        embedding=emb,
    )
    rag = RAGService(vs.as_retriever(search_kwargs={"k": args.branch_k}), {})
    logger.info(
        "稠密检索 branch_k=%d，BM25 branch_k=%d，RRF_K=%s，融合后 final_k=%d",
        args.branch_k,
        args.branch_k,
        args.rrf_k,
        args.final_k,
    )

    questions = df[args.question_col].astype(str).tolist()
    gold_list = df[args.redis_key_col].tolist()
    bk = args.branch_k
    fk = args.final_k

    hits: List[int] = []
    ranks: List[int] = []
    fused_flat: List[str] = []
    dense_flat: List[str] = []
    bm25_flat: List[str] = []

    nq = len(questions)
    for i, (q, raw_gold) in enumerate(zip(questions, gold_list), start=1):
        if i == 1 or i % 20 == 0 or i == nq:
            logger.info("混合检索 [%d/%d]", i, nq)

        dense_docs = rag.retrieve_documents(q)
        dense_ids = [document_redis_key(d) for d in dense_docs][:bk]

        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:bk]
        bm25_ids = [corpus_keys[j] for j in top_idx]

        fused = rrf_fuse_ids([dense_ids, bm25_ids], args.rrf_k, fk)
        fused_flat.append("|".join(fused))
        dense_flat.append("|".join(dense_ids))
        bm25_flat.append("|".join(bm25_ids))

        gold = str(raw_gold).strip() if pd.notna(raw_gold) else ""
        rank = 0
        if gold:
            for pos, rid in enumerate(fused, start=1):
                if rid == gold:
                    rank = pos
                    break
        ranks.append(rank)
        hits.append(1 if rank else 0)

    n = len(hits)
    hit_rate = sum(hits) / n if n else 0.0
    mrr = (sum(1.0 / r for r in ranks if r) / n) if n else 0.0

    out = df.copy()
    out[f"hybrid_rrf_hit@{fk}"] = hits
    out[f"hybrid_rrf_gold_rank_in_top{fk}"] = ranks
    out["hybrid_rrf_retrieved_redis_keys"] = fused_flat
    out[f"dense_top{bk}_redis_keys"] = dense_flat
    out[f"bm25_top{bk}_redis_keys"] = bm25_flat

    out_path = args.output or os.path.join(
        SCRIPT_DIR, f"hybrid_retrieval_recall_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== 稠密+BM25 RRF 混合 Gold Hit@K ===")
    print(
        f"branch_k={bk}（各路先取）→ RRF 重排 → 只保留 final_k={fk} 条（与单路 Hit@{fk} 对齐）"
        f"  RRF_K={args.rrf_k}"
    )
    print(f"样本: {n}  |  Hit@{fk}: {hit_rate:.4f} ({sum(hits)}/{n})  |  MRR: {mrr:.4f}")
    print(f"总耗时: {time.perf_counter() - t0:.2f} 秒")
    print(f"文件: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
