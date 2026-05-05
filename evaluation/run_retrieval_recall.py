
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def find_project_root() -> str:
    """向上查找包含 config.py 的目录作为项目根。"""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "config.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "无法定位项目根目录（未找到 config.py）。请在工程根目录下运行本脚本。"
            )
        d = parent


PROJECT_ROOT = find_project_root()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EMBEDDING_MODEL, REDIS_DB, REDIS_HOST, REDIS_PORT  # noqa: E402
from langchain_community.vectorstores import Redis  # noqa: E402
from services.rag_service import RAGService, document_redis_key  # noqa: E402


def load_dashscope_embeddings():
    """与 run_evaluation 一致：从 main.py 加载 DashScopeEmbeddings。"""
    main_path = os.path.join(PROJECT_ROOT, "main.py")
    spec = importlib.util.spec_from_file_location("main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {main_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DashScopeEmbeddings(model=EMBEDDING_MODEL)


def build_rag_service(top_k: int, index_name: str) -> RAGService:
    embeddings = load_dashscope_embeddings()
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    vectorstore = Redis(
        redis_url=redis_url,
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return RAGService(retriever, title_index={})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检索 Gold Hit@K：redis_key 是否出现在 Top-K 向量检索结果中",
    )
    parser.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "evaluation", "test_data.generated.csv"),
        help="测试集 CSV，需含 question 与 redis_key 列",
    )
    parser.add_argument("--question-col", default="question", help="问题列名")
    parser.add_argument(
        "--redis-key-col",
        default="redis_key",
        help="生成测试集时的 gold Redis 文档 key 列名",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索条数 K，需与线上一致",
    )
    parser.add_argument(
        "--index-name",
        default="drug_vectors",
        help="Redis 向量索引名",
    )
    parser.add_argument(
        "--output",
        default="",
        help="结果 CSV 路径；默认写入本脚本所在目录，带时间戳文件名",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"找不到文件: {args.file}")

    df = pd.read_csv(args.file)
    if args.question_col not in df.columns:
        raise ValueError(f"CSV 缺少列: {args.question_col}")
    if args.redis_key_col not in df.columns:
        raise ValueError(
            f"CSV 缺少列: {args.redis_key_col}（需要 gold redis key 才能计算 Hit@K）"
        )

    questions = df[args.question_col].astype(str).tolist()
    gold_keys = df[args.redis_key_col].tolist()

    logger.info("正在连接向量库并加载 Embedding（main.DashScopeEmbeddings）…")
    t0 = time.perf_counter()
    rag = build_rag_service(top_k=args.top_k, index_name=args.index_name)
    logger.info(
        "向量服务就绪 top_k=%s index=%s，将对 %d 条问题逐条检索（每条：query 转向量 + Redis 搜 Top-K）",
        args.top_k,
        args.index_name,
        len(questions),
    )

    k = args.top_k
    hits: list[int] = []
    ranks: list[int] = []
    retrieved_flat: list[str] = []

    nq = len(questions)
    for idx, (question, raw_gold) in enumerate(zip(questions, gold_keys), start=1):
        logger.info("检索 [%d/%d] %s", idx, nq, (question[:120] + "…") if len(question) > 120 else question)
        docs = rag.retrieve_documents(question)
        ids = [document_redis_key(d) for d in docs]
        retrieved_flat.append("|".join(ids))

        gold = str(raw_gold).strip() if pd.notna(raw_gold) else ""
        if gold and gold in ids:
            rank = ids.index(gold) + 1
        else:
            rank = 0
        ranks.append(rank)
        hits.append(1 if rank > 0 else 0)

    elapsed = time.perf_counter() - t0
    logger.info("全部检索完成，用时 %.2f 秒；正在汇总指标并写 CSV", elapsed)

    n = len(hits)
    hit_rate = sum(hits) / n if n else 0.0
    mrr = (sum(1.0 / r for r in ranks if r > 0) / n) if n else 0.0

    out = df.copy()
    out[f"retrieval_hit@{k}"] = hits
    out["gold_rank_in_topk"] = ranks
    out["retrieved_redis_keys"] = retrieved_flat

    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SCRIPT_DIR, f"retrieval_recall_{ts}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== 检索 Gold Hit@K（上述已对每条问题完成向量检索）===")
    print(f"样本数: {n}")
    print(f"Top-K: {k}")
    print(f"检索阶段耗时: {elapsed:.2f} 秒")
    print(f"Hit@{k}: {hit_rate:.4f} ({sum(hits)}/{n})")
    print(f"MRR（未命中按 0 计入平均）: {mrr:.4f}")
    print(f"明细: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
