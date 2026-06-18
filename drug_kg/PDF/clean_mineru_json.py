from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PDF_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PDF_DIR / "outputs_mineru"
DEFAULT_CLEAN_DIR = PDF_DIR / "cleaned_texts"
DEFAULT_CHUNK_DIR = PDF_DIR / "chunk_json"

DROP_TYPES = {"header", "footer", "page_number", "image", "chart"}
KEEP_TYPES = {"text", "list", "table"}
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t\u3000]{2,}")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
BROKEN_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
REFERENCE_RE = re.compile(r"^\s*(参考文献|References?)\s*$", re.I)
META_LINE_RE = re.compile(
    r"(DOI|E-?mail|收稿日期|作者简介|通信作者|关键词|Key words|摘要|Abstract|基金项目|中图分类号|文献标识码|文章编号)",
    re.I,
)
HEADING_RE = re.compile(
    r"^\s*("
    r"[0-9]+(\.[0-9]+)*"
    r"|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+"
    r"|[IVXivx]+"
    r")[\u3001.\uff0e]?\s*[\u4e00-\u9fffA-Za-z].*$"
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = BROKEN_CJK_SPACE_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def sort_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(block: dict[str, Any]) -> tuple[int, float, float, int]:
        page_idx = int(block.get("page_idx", 0) or 0)
        bbox = block.get("bbox") or [0, 0, 0, 0]
        top = float(bbox[1]) if len(bbox) >= 2 else 0.0
        left = float(bbox[0]) if len(bbox) >= 1 else 0.0
        index = int(block.get("index", 0) or 0)
        return (page_idx, top, left, index)

    return sorted(blocks, key=sort_key)


def should_drop_text(text: str) -> bool:
    line = text.strip()
    if not line:
        return True
    if REFERENCE_RE.match(line):
        return True
    if META_LINE_RE.search(line) and len(line) < 120:
        return True
    if len(line) <= 2:
        return True
    return False


def extract_candidate_blocks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type", "")).strip().lower()
        if block_type in DROP_TYPES:
            continue
        if block_type not in KEEP_TYPES and block_type:
            continue

        text = normalize_text(str(item.get("text", "") or ""))
        if should_drop_text(text):
            continue

        block = {
            "type": block_type or "text",
            "text": text,
            "page_idx": item.get("page_idx", 0),
            "bbox": item.get("bbox") or [0, 0, 0, 0],
            "text_level": item.get("text_level"),
            "index": item.get("index", 0),
        }
        blocks.append(block)
    return sort_blocks(blocks)


def merge_blocks_to_paragraphs(blocks: list[dict[str, Any]]) -> list[str]:
    paragraphs: list[str] = []
    buffer = ""

    for block in blocks:
        text = block["text"]
        is_heading = block.get("text_level") in {1, 2, 3} or (HEADING_RE.match(text) and len(text) <= 50)

        if REFERENCE_RE.match(text):
            break

        if is_heading:
            if buffer.strip():
                paragraphs.append(buffer.strip())
                buffer = ""
            paragraphs.append(text)
            continue

        if not buffer:
            buffer = text
            continue

        if buffer.endswith(("。", "！", "？", ";", "；", ":", "：")):
            paragraphs.append(buffer.strip())
            buffer = text
        else:
            if re.search(r"[A-Za-z0-9]$", buffer) and re.match(r"^[A-Za-z0-9]", text):
                buffer += " " + text
            else:
                buffer += text

    if buffer.strip():
        paragraphs.append(buffer.strip())

    cleaned: list[str] = []
    for para in paragraphs:
        para = normalize_text(para)
        if not para:
            continue
        if REFERENCE_RE.match(para):
            break
        if should_drop_text(para):
            continue
        cleaned.append(para)
    return cleaned


def chunk_text(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
            continue
        if len(current) + 1 + len(para) <= chunk_size:
            current = current + "\n" + para
        else:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n" + para
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())
    return chunks


def process_file(path: Path, clean_dir: Path, chunk_dir: Path, chunk_size: int, overlap: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    blocks = extract_candidate_blocks(data)
    paragraphs = merge_blocks_to_paragraphs(blocks)
    cleaned_text = "\n".join(paragraphs).strip()
    chunks = chunk_text(paragraphs, chunk_size=chunk_size, overlap=overlap)

    doc_name = path.parent.name
    clean_path = clean_dir / f"{doc_name}.txt"
    clean_path.write_text(cleaned_text, encoding="utf-8")

    doc_chunk_dir = chunk_dir / doc_name
    doc_chunk_dir.mkdir(parents=True, exist_ok=True)
    for old_file in doc_chunk_dir.glob("chunk_*.json"):
        old_file.unlink()

    for idx, chunk in enumerate(chunks):
        payload = [
            {
                "text": chunk,
                "source": f"{doc_name}.pdf",
                "chunk_index": f"{doc_name}:{idx}",
            }
        ]
        out_path = doc_chunk_dir / f"chunk_{idx}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "file": str(path),
        "doc_name": doc_name,
        "raw_blocks": len(data),
        "kept_blocks": len(blocks),
        "paragraphs": len(paragraphs),
        "clean_chars": len(cleaned_text),
        "chunks": len(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MinerU JSON outputs into text and chunk JSON.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    args.clean_dir.mkdir(parents=True, exist_ok=True)
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.rglob("result.json")):
        summaries.append(
            process_file(
                path=path,
                clean_dir=args.clean_dir,
                chunk_dir=args.chunk_dir,
                chunk_size=max(200, args.chunk_size),
                overlap=max(0, args.overlap),
            )
        )

    summary_path = PDF_DIR / "clean_mineru_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
