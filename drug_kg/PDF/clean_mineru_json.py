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
QA_HEADING_RE = re.compile(r"^\s*\d{1,4}[.\u3001\uff0e]\s*")
SECTION_HEADING_RE = re.compile(
    r"^\s*("
    r"[0-9]+(\.[0-9]+)*"
    r"|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+"
    r"|[IVXivx]+"
    r")[\u3001.\uff0e]?\s*"
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


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    sentence_parts = re.split(r"(?<=[。！？；;])", text)
    parts = [part.strip() for part in sentence_parts if part.strip()]
    if not parts:
        parts = [text]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
            continue
        if len(current) + len(part) <= chunk_size:
            current += part
        else:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + part
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_embedded_qa_units(units: list[str]) -> list[str]:
    refined: list[str] = []
    pattern = re.compile(r"(?m)(?=^\s*\d{1,4}[.\u3001\uff0e]\s*)")

    for unit in units:
        text = unit.strip()
        if not text:
            continue
        if not QA_HEADING_RE.match(text):
            refined.append(text)
            continue

        parts = [part.strip() for part in pattern.split(text) if part.strip()]
        if len(parts) <= 1:
            refined.append(text)
            continue
        refined.extend(parts)

    return refined


def is_qa_document(paragraphs: list[str]) -> bool:
    qa_count = sum(1 for para in paragraphs if QA_HEADING_RE.match(para.strip()))
    if qa_count >= 2:
        return True
    joined = "\n".join(paragraphs[:5])
    if "问题与解答" in joined or "问答" in joined:
        return True
    return False


def build_qa_units(paragraphs: list[str]) -> list[str]:
    buckets: list[tuple[int | None, str]] = []
    current = ""
    current_num: int | None = None

    for para in paragraphs:
        text = para.strip()
        if not text:
            continue

        if text == "问题与解答":
            if current.strip():
                buckets.append((current_num, current.strip()))
            current = text
            current_num = None
            continue

        if QA_HEADING_RE.match(text):
            if current.strip():
                buckets.append((current_num, current.strip()))
            current = text
            match = re.match(r"^\s*(\d{1,4})", text)
            current_num = int(match.group(1)) if match else None
            continue

        if not current:
            current = text
            current_num = None
            continue

        current = current + "\n" + text

    if current.strip():
        buckets.append((current_num, current.strip()))

    refined_pairs: list[tuple[int | None, str]] = []
    for num, unit in buckets:
        if unit.strip() == "问题与解答":
            continue
        if unit.startswith("问题与解答\n"):
            unit = unit.split("\n", 1)[1].strip()
        if unit:
            refined_pairs.append((num, unit))

    numbered = [(num, unit) for num, unit in refined_pairs if num is not None]
    unnumbered = [unit for num, unit in refined_pairs if num is None]
    numbered.sort(key=lambda item: item[0])

    ordered = [unit for _, unit in numbered]
    ordered.extend(unnumbered)
    return ordered


def build_semantic_units(paragraphs: list[str]) -> list[str]:
    if is_qa_document(paragraphs):
        return build_qa_units(paragraphs)

    units: list[str] = []
    current = ""
    current_kind = ""

    for para in paragraphs:
        text = para.strip()
        if not text:
            continue

        is_qa = bool(QA_HEADING_RE.match(text))
        is_heading = bool(SECTION_HEADING_RE.match(text)) and len(text) <= 60

        if is_qa:
            if current.strip():
                units.append(current.strip())
            current = text
            current_kind = "qa"
            continue

        if is_heading:
            if current.strip():
                units.append(current.strip())
            current = text
            current_kind = "heading"
            continue

        if not current:
            current = text
            current_kind = "text"
            continue

        if current_kind == "qa":
            current = current + "\n" + text
            continue

        if current_kind == "heading":
            current = current + "\n" + text
            current_kind = "text"
            continue

        units.append(current.strip())
        current = text
        current_kind = "text"

    if current.strip():
        units.append(current.strip())
    return split_embedded_qa_units(units)


def chunk_text(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    units = build_semantic_units(paragraphs)
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:
            oversized_parts = split_long_text(unit, chunk_size=chunk_size, overlap=overlap)
        else:
            oversized_parts = [unit]

        for part in oversized_parts:
            if not current:
                current = part
                continue
            if len(current) + 1 + len(part) <= chunk_size:
                current = current + "\n" + part
            else:
                chunks.append(current.strip())
                current = part

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
