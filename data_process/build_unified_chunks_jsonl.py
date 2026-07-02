from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_GONGZHONGHAO_PATH = PROJECT_ROOT / "content" / "gongzhonghao_text_chunks.jsonl"
DEFAULT_PDF_CHUNK_DIR = PROJECT_ROOT / "drug_kg" / "PDF" / "chunk_json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "content" / "unified_chunks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一公众号块和 PDF 块为一个最小 JSONL 语料。")
    parser.add_argument(
        "--gongzhonghao-input",
        default=str(DEFAULT_GONGZHONGHAO_PATH),
        help="公众号 chunk JSONL 路径。",
    )
    parser.add_argument(
        "--pdf-chunk-dir",
        default=str(DEFAULT_PDF_CHUNK_DIR),
        help="PDF chunk JSON 根目录。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="统一后的 JSONL 输出路径。",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def load_pdf_rows(chunk_file: Path) -> list[dict[str, Any]]:
    raw = json.loads(chunk_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def normalize_chunk_index(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        tail = value.rsplit(":", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
        if value.isdigit():
            return int(value)
    return fallback


def normalize_source(value: Any, fallback: str) -> str:
    source = str(value or "").strip()
    return source or fallback


def build_gongzhonghao_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        text = normalize_text(row.get("text"))
        if not text:
            continue
        chunk_index = normalize_chunk_index(row.get("chunk_index"), len(rows))
        source = normalize_source(row.get("source"), "公众号")
        chunk_id = str(row.get("chunk_id") or f"{source}#{chunk_index}")
        rows.append(
            {
                "text": text,
                "source": source,
                "source_type": "gongzhonghao",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
            }
        )
    return rows


def build_pdf_rows(chunk_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_file in sorted(chunk_dir.glob("*/*.json")):
        doc_name = chunk_file.parent.name
        for local_offset, row in enumerate(load_pdf_rows(chunk_file)):
            text = normalize_text(row.get("text"))
            if not text:
                continue
            chunk_index = normalize_chunk_index(row.get("chunk_index"), local_offset)
            source = normalize_source(row.get("source"), doc_name)
            chunk_id = f"{doc_name}#{chunk_index}"
            rows.append(
                {
                    "text": text,
                    "source": source,
                    "source_type": "pdf",
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            )
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for corpus_chunk_index, row in enumerate(rows):
            payload = dict(row)
            payload["corpus_chunk_index"] = corpus_chunk_index
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    gongzhonghao_path = Path(args.gongzhonghao_input)
    pdf_chunk_dir = Path(args.pdf_chunk_dir)
    output_path = Path(args.output)

    if not gongzhonghao_path.is_file():
        raise FileNotFoundError(f"公众号 chunk 文件不存在: {gongzhonghao_path}")
    if not pdf_chunk_dir.is_dir():
        raise FileNotFoundError(f"PDF chunk 目录不存在: {pdf_chunk_dir}")

    gongzhonghao_rows = build_gongzhonghao_rows(gongzhonghao_path)
    pdf_rows = build_pdf_rows(pdf_chunk_dir)
    all_rows = gongzhonghao_rows + pdf_rows
    write_jsonl(all_rows, output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "gongzhonghao_rows": len(gongzhonghao_rows),
                "pdf_rows": len(pdf_rows),
                "total_rows": len(all_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
