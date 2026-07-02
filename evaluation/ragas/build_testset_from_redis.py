from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from typing import Dict, List, Sequence, Tuple

import redis
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (  # noqa: E402
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    VECTOR_INDEX_NAME,
    VECTOR_KEY_PREFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Redis 检索片段自动构建 RAGAS 测试集")
    parser.add_argument("--redis-host", default=REDIS_HOST, help="Redis host")
    parser.add_argument("--redis-port", default=REDIS_PORT, type=int, help="Redis port")
    parser.add_argument("--redis-db", default=REDIS_DB, type=int, help="Redis db")
    parser.add_argument(
        "--index-name",
        default=VECTOR_INDEX_NAME,
        help="Redis 向量索引名（用于自动推导默认 key 前缀）",
    )
    parser.add_argument(
        "--key-prefix",
        default=VECTOR_KEY_PREFIX,
        help="片段 key 前缀（默认读取配置中的 VECTOR_KEY_PREFIX）",
    )
    parser.add_argument(
        "--sample-size",
        default=200,
        type=int,
        help="抽样片段数（不超过 Redis 中片段总数；默认 200）",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="随机种子，保证可复现",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_PROJECT_ROOT, "evaluation", "test_data.generated.csv"),
        help="输出 CSV 路径（默认写到工程下 evaluation/test_data.generated.csv）",
    )
    parser.add_argument(
        "--max-text-len",
        default=1200,
        type=int,
        help="每个片段喂给 LLM 的最大字符数",
    )
    return parser.parse_args()


def read_chunk_rows(
    rds: redis.Redis,
    key_prefix: str,
) -> List[Tuple[str, Dict[str, str]]]:
    keys = sorted(k for k in rds.scan_iter(match=f"{key_prefix}*"))
    rows: List[Tuple[str, Dict[str, str]]] = []
    for key in keys:
        values = rds.hmget(
            key,
            "content",
            "text",
            "source",
            "source_file",
            "corpus_chunk_index",
        )
        row: Dict[str, str] = {}
        for field, value in zip(
            ("content", "text", "source", "source_file", "corpus_chunk_index"),
            values,
        ):
            if value is not None:
                row[field] = value

        text = row.get("content", "").strip() or row.get("text", "").strip()
        if not text:
            continue
        rows.append((key, row))
    return rows


def extract_json_object(raw: str) -> Dict[str, str]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"无法解析 JSON: {raw[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("返回不是 JSON object")
    return parsed


def build_single_sample(
    llm: ChatOpenAI,
    chunk_text: str,
    source: str,
) -> Tuple[str, str]:
    prompt = (
        "你是中文医疗健康问答测试集构建助手。"
        "请只基于给定片段生成 1 组用于 RAG 评估的数据，返回严格 JSON："
        '{"question":"...","ground_truth":"..."}。\n'
        "要求：\n"
        "1) question 必须是用户自然提问句；\n"
        "2) ground_truth 必须可被片段直接支持，不能引入片段外事实；\n"
        "3) ground_truth 简洁、完整，控制在 1-3 句；\n"
        "4) 不要输出任何解释或 markdown。\n"
    )
    user_content = (
        f"来源: {source}\n"
        f"片段:\n{chunk_text}\n"
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_content),
    ]
    resp = llm.invoke(messages)
    parsed = extract_json_object(resp.content)
    question = str(parsed.get("question", "")).strip()
    ground_truth = str(parsed.get("ground_truth", "")).strip()
    if not question or not ground_truth:
        raise ValueError(f"LLM 生成字段为空: {resp.content[:200]}")
    return question, ground_truth


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    key_prefix = args.key_prefix or f"doc:{args.index_name}:"

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

    all_rows = read_chunk_rows(rds, key_prefix)
    if not all_rows:
        raise RuntimeError(f"未找到 key 前缀为 {key_prefix} 的有效片段")

    sample_size = min(args.sample_size, len(all_rows))
    selected = random.sample(all_rows, sample_size)
    logger.info("从 %d 条片段中抽样 %d 条（key_prefix=%s）", len(all_rows), sample_size, key_prefix)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base=DASHSCOPE_BASE_URL,
        temperature=0.2,
    )

    out_rows: List[Dict[str, str]] = []
    for idx, (key, row) in enumerate(selected, start=1):
        source = row.get("source", "") or row.get("source_file", "") or "未知来源"
        text = (row.get("content", "") or row.get("text", ""))[: args.max_text_len]
        try:
            question, gt = build_single_sample(llm=llm, chunk_text=text, source=source)
            out_rows.append(
                {
                    "question": question,
                    "ground_truth": gt,
                    "redis_key": key,
                    "source": source,
                    "corpus_chunk_index": row.get("corpus_chunk_index", ""),
                }
            )
            logger.info("生成进度: %d/%d", idx, sample_size)
        except Exception as exc:
            logger.warning("跳过样本 key=%s, 原因=%s", key, exc)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "ground_truth", "redis_key", "source", "corpus_chunk_index"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    logger.info("完成：共输出 %d 条到 %s", len(out_rows), args.output)


if __name__ == "__main__":
    main()
