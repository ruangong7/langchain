"""Clean OCR-derived medical text files and export chunk JSON files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t\u3000]{2,}")
BROKEN_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
DECORATION_RE = re.compile(r"^(.)\1{8,}$")
REFERENCE_START_RE = re.compile(r"^\s*(\u53c2\u8003\u6587\u732e|References?)\s*$", re.I)
DOI_RE = re.compile(r"^\s*DOI[:\uff1a]?\s*", re.I)
FIG_TABLE_RE = re.compile(r"^\s*(\u56fe|\u8868|Figure|Fig\.?|Table|Tab\.?)\s*\d+", re.I)
EMAIL_RE = re.compile(r"E-?mail|\u90ae\u7bb1", re.I)
AUTHOR_LINE_RE = re.compile(
    r"[\uff08(]?\d+[\uff09)]?.{0,30}("
    r"\u533b\u9662|\u5927\u5b66|\u836f\u5b66\u90e8|\u836f\u5242\u79d1|"
    r"\u7814\u7a76\u6240|\u5b66\u9662)"
)
JOURNAL_META_RE = re.compile(
    r"("
    r"\u7b2c.{0,6}\u5377.*\u7b2c.{0,6}\u671f|ISSN|CNKI|"
    r"\u4e2d\u56fe\u5206\u7c7b\u53f7|\u6587\u732e\u6807\u5fd7\u7801|"
    r"\u6587\u7ae0\u7f16\u53f7|\u6536\u7a3f\u65e5\u671f|"
    r"\u4e2d\u56fd\u533b\u9662\u7528\u836f\u8bc4\u4ef7\u4e0e\u5206\u6790|"
    r"\u836f\u5b66\u60c5\u62a5\u901a\u8baf"
    r")",
    re.I,
)
HEADING_RE = re.compile(
    r"^\s*("
    r"[0-9]+(\.[0-9]+)*"
    r"|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+"
    r"|[IVXivx]+"
    r")[\u3001.\uff0e]?\s*[\u4e00-\u9fffA-Za-z].*$"
)
TOO_SHORT_NOISE_RE = re.compile(r"^[\W_]{1,12}$")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = BROKEN_CJK_SPACE_RE.sub("", text)
    return text


def _looks_like_metadata(line: str) -> bool:
    if DOI_RE.search(line):
        return True
    if FIG_TABLE_RE.match(line):
        return True
    if EMAIL_RE.search(line):
        return True
    if JOURNAL_META_RE.search(line):
        return True
    if AUTHOR_LINE_RE.search(line):
        return True
    if "\u5546\u54c1\u540d" not in line and "\u5316\u5b66\u7ed3\u6784\u56fe" in line:
        return True
    if len(line) < 80 and any(token in line for token in ("\u4f5c\u8005", "\u6458\u8981", "\u5173\u952e\u8bcd")):
        return True
    return False


def _garbage_ratio(line: str) -> float:
    if not line:
        return 0.0
    bad = sum(1 for ch in line if ch == "\ufffd")
    return bad / max(len(line), 1)


def _iter_clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if REFERENCE_START_RE.match(line):
            break
        if DECORATION_RE.match(line):
            continue
        if TOO_SHORT_NOISE_RE.match(line):
            continue
        if _garbage_ratio(line) > 0.15:
            continue
        if _looks_like_metadata(line):
            continue
        cleaned.append(line)
    return cleaned


def _join_with_previous(buffer: str, line: str) -> str:
    if not buffer:
        return line
    if buffer.endswith(("\u3002", "\uff01", "\uff1f", ";", "\uff1b")):
        return buffer + "\n" + line
    if re.search(r"[A-Za-z0-9]$", buffer) and re.match(r"^[A-Za-z0-9]", line):
        return buffer + " " + line
    return buffer + line


def _merge_lines_to_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buffer = ""
    for line in lines:
        if HEADING_RE.match(line) and len(line) <= 40:
            if buffer.strip():
                paragraphs.extend(part.strip() for part in buffer.split("\n") if part.strip())
                buffer = ""
            paragraphs.append(line)
            continue

        if not buffer:
            buffer = line
            continue

        buffer = _join_with_previous(buffer, line)

    if buffer.strip():
        paragraphs.extend(part.strip() for part in buffer.split("\n") if part.strip())
    return [p for p in paragraphs if len(p) >= 8]


def clean_ocr_text(text: str) -> str:
    normalized = _normalize_text(text)
    lines = normalized.split("\n")
    cleaned_lines = _iter_clean_lines(lines)
    paragraphs = _merge_lines_to_paragraphs(cleaned_lines)
    final_text = "\n".join(paragraphs)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)
    return final_text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OCR medical text files and export chunk json")
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "docs")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--clean-dir", type=Path, default=Path(__file__).resolve().parent / "cleaned_texts")
    parser.add_argument("--chunk-dir", type=Path, default=Path(__file__).resolve().parent / "pdf_chunks")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    args.clean_dir.mkdir(parents=True, exist_ok=True)
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    txt_paths = sorted(args.input_dir.glob(args.pattern))
    chunk_global_idx = 0
    summary: list[dict[str, object]] = []

    for path in txt_paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_ocr_text(raw)
        clean_path = args.clean_dir / path.name
        clean_path.write_text(cleaned, encoding="utf-8")

        chunks = chunk_text(cleaned, chunk_size=max(200, args.chunk_size), overlap=max(0, args.overlap))
        for local_idx, chunk in enumerate(chunks):
            payload = [
                {
                    "text": chunk,
                    "source": path.stem,
                    "chunk_index": f"{path.stem}:{local_idx}",
                }
            ]
            out_path = args.chunk_dir / f"chunk_{chunk_global_idx}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            chunk_global_idx += 1

        summary.append(
            {
                "file": path.name,
                "raw_chars": len(raw),
                "clean_chars": len(cleaned),
                "chunks": len(chunks),
            }
        )

    (args.chunk_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
