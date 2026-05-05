
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

# 保证能导入 consultant_py 下的 config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONSULTANT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _CONSULTANT_ROOT not in sys.path:
    sys.path.insert(0, _CONSULTANT_ROOT)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_txt_files(gongzhonghao_dir: str) -> List[tuple[str, str]]:
    """返回 (文件名不含扩展名, 全文) 列表。"""
    pairs: List[tuple[str, str]] = []
    if not os.path.isdir(gongzhonghao_dir):
        raise FileNotFoundError(f"目录不存在: {gongzhonghao_dir}")
    for name in sorted(os.listdir(gongzhonghao_dir)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(gongzhonghao_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            stem = os.path.splitext(name)[0]
            pairs.append((stem, text))
    return pairs


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " ", ""],
    )


def chunk_corpus(
    pairs: List[tuple[str, str]],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    按字符长度递归切分（约 500～800 字量级由 chunk_size 控制），块间重叠由 chunk_overlap 控制。
    每个块 metadata：source、source_file、chunk_index、chunk_count、chunk_id。
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    all_docs: List[Document] = []
    global_offset = 0
    for stem, text in pairs:
        base_meta = {"source": stem, "source_file": f"{stem}.txt"}
        docs = splitter.create_documents([text], metadatas=[dict(base_meta)])
        n = len(docs)
        for i, d in enumerate(docs):
            d.metadata["chunk_index"] = i
            d.metadata["chunk_count"] = n
            d.metadata["chunk_id"] = f"{stem}#{i}"
            d.metadata["corpus_chunk_index"] = global_offset + i
        global_offset += n
        all_docs.extend(docs)
        logger.info("文件 %s.txt -> %d 块 (chunk_size=%d, overlap=%d)", stem, n, chunk_size, chunk_overlap)
    return all_docs


def write_jsonl(documents: List[Document], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in documents:
            row: Dict[str, Any] = {
                "text": d.page_content,
                **d.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("已写入 %d 条到 %s", len(documents), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="公众号 txt 按字符递归分块并导出 JSONL（无向量 API）")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(_CONSULTANT_ROOT, "content", "gongzhonghao"),
        help="txt 目录，默认 content/gongzhonghao/test",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_CONSULTANT_ROOT, "content", "gongzhonghao_text_chunks.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="每块目标最大字符数，建议 500～800",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="相邻块重叠字符数，约为 chunk_size 的 15%%～20%%",
    )
    args = parser.parse_args()

    if args.chunk_size < 100:
        parser.error("--chunk-size 过小，建议 >= 500 用于中文长文")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap 必须小于 --chunk-size")

    pairs = load_txt_files(args.input_dir)
    if not pairs:
        logger.warning("未找到任何非空 .txt，退出")
        return

    docs = chunk_corpus(
        pairs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    write_jsonl(docs, args.output)


if __name__ == "__main__":
    main()
