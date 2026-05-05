from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple

import jieba
import pandas as pd
import redis
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


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

from config import REDIS_DB, REDIS_HOST, REDIS_PORT  # noqa: E402


def _redis_scan_match(key_prefix: str) -> str:
    """与 build_testset_from_redis 一致：默认 qwen3: → 匹配 qwen3:*"""
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
    """返回 (片段文本列表, 与之一一对应的 redis key 列表)。"""
    rds = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    rds.ping()
    texts: List[str] = []
    keys: List[str] = []
    pattern = _redis_scan_match(key_prefix)

    for key in sorted(rds.scan_iter(match=pattern)):
        row = rds.hmget(
            key,
            "content",
            "text",
            "source",
            "source_file",
            "corpus_chunk_index",
        )
        fields = ("content", "text", "source", "source_file", "corpus_chunk_index")
        d = {f: v for f, v in zip(fields, row) if v is not None}
        text = (d.get("content") or "").strip() or (d.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        keys.append(key)
    if not texts:
        raise RuntimeError(f"Redis 中未找到前缀 {key_prefix!r} 的有效片段")
    logger.info("从 Redis 加载 %d 条片段用于 BM25（prefix=%s）", len(texts), key_prefix)
    return texts, keys


def tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM25 检索 Gold Hit@K（与 Redis 语料对齐）")
    p.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "evaluation", "recall", "test_data.generated.csv"),
        help="测试集 CSV（含 question、redis_key）",
    )
    p.add_argument("--question-col", default="question")
    p.add_argument("--redis-key-col", default="redis_key")
    p.add_argument("--top-k", type=int, default=5, help="BM25 返回条数")
    p.add_argument("--key-prefix", default="qwen3:", help="与写入 Redis / 向量索引一致的前缀")
    p.add_argument("--redis-host", default=REDIS_HOST)
    p.add_argument("--redis-port", type=int, default=REDIS_PORT)
    p.add_argument("--redis-db", type=int, default=REDIS_DB)
    p.add_argument("--output", default="", help="默认同目录 bm25_retrieval_recall_<时间>.csv")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(args.file)
    df = pd.read_csv(args.file)
    if args.question_col not in df.columns or args.redis_key_col not in df.columns:
        raise ValueError(f"CSV 需含列: {args.question_col}, {args.redis_key_col}")

    t0 = time.perf_counter()
    corpus_texts, corpus_keys = read_corpus_from_redis(
        args.redis_host, args.redis_port, args.redis_db, args.key_prefix
    )
    logger.info("构建 BM25 索引（jieba 分词）…")
    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 索引就绪，耗时 %.2f 秒", time.perf_counter() - t0)

    questions = df[args.question_col].astype(str).tolist()
    gold_list = df[args.redis_key_col].tolist()
    k = args.top_k
    hits: List[int] = []
    ranks: List[int] = []
    flat: List[str] = []

    nq = len(questions)
    for i, (q, raw_gold) in enumerate(zip(questions, gold_list), start=1):
        if i == 1 or i % 20 == 0 or i == nq:
            logger.info("BM25 检索 [%d/%d]", i, nq)
        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:k]
        top_keys = [corpus_keys[j] for j in top_idx]
        flat.append("|".join(top_keys))

        gold = str(raw_gold).strip() if pd.notna(raw_gold) else ""
        rank = 0
        if gold:
            for pos, rid in enumerate(top_keys, start=1):
                if rid == gold:
                    rank = pos
                    break
        ranks.append(rank)
        hits.append(1 if rank else 0)

    n = len(hits)
    hit_rate = sum(hits) / n if n else 0.0
    mrr = (sum(1.0 / r for r in ranks if r) / n) if n else 0.0

    hit_col = f"bm25_hit@{k}"
    out = df.copy()
    out[hit_col] = hits
    out["bm25_gold_rank_in_topk"] = ranks
    out["bm25_retrieved_redis_keys"] = flat

    out_path = args.output or os.path.join(
        SCRIPT_DIR, f"bm25_retrieval_recall_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== BM25 Gold Hit@K（语料来自 Redis，与向量评测同源）===")
    print(f"语料条数: {len(corpus_texts)}  |  评测条数: {n}  |  Top-K: {k}")
    print(f"Hit@{k}: {hit_rate:.4f} ({sum(hits)}/{n})  |  MRR: {mrr:.4f}")
    print(f"总耗时: {time.perf_counter() - t0:.2f} 秒")
    print(f"文件: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
