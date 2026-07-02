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

DROP_TYPES = {"header", "footer", "page_number", "page_footnote", "image", "chart"}
KEEP_TYPES = {"text", "list", "table"}
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t\u3000]{2,}")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
BROKEN_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
REFERENCE_RE = re.compile(
    r"^\s*(参考文献|References?)(\s*[\(\[（].*?[\)\]）])?(\s*[:：])?\s*$",
    re.I,
)
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
COMPARE_TEXT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
AUTHOR_LINE_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9·•,，\s\\\*\-]+$")
AFFILIATION_RE = re.compile(
    r"(医院|大学|学院|研究所|研究院|中心|实验室|药剂科|School|University|College|Department)",
    re.I,
)
EN_ABSTRACT_RE = re.compile(r"^\s*(Abstract|Keywords?|Key words)\s*[:：]", re.I)
LEADING_META_RE = re.compile(r"^\s*(收稿日期|修回日期|Tel\s*[:：]?|基金项目|作者简介|通信作者)\b", re.I)
FLOW_NOTE_RE = re.compile(r"[（(]\s*(下转第\s*\d+\s*页|上接第\s*\d+\s*页|本文编辑\s*[^）)]*)\s*[）)]")
LATEX_COMMAND_RE = re.compile(r"\\(?:mathrm|mathbf|text|operatorname)\s*\{\s*([^{}]+?)\s*\}")
LATEX_STYLE_RE = re.compile(r"\\mathfrak\s*\{\s*([^{}]+?)\s*\}")
SIMPLE_BRACE_RE = re.compile(r"\{\s*([^{}]{1,80}?)\s*\}")
DECIMAL_SPACE_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
RANGE_SPACE_RE = re.compile(r"(?<=\d)\s*～\s*(?=\d)")
BROKEN_UNIT_PATTERNS = (
    (re.compile(r"\bm\s+g\b", re.I), "mg"),
    (re.compile(r"\bn\s+g\b", re.I), "ng"),
    (re.compile(r"\bp\s+g\b", re.I), "pg"),
    (re.compile(r"\bm\s+L\b"), "mL"),
    (re.compile(r"\bL\s*-\s*1\b", re.I), "L-1"),
    (re.compile(r"\bm\s+m\s+o\s+l\b", re.I), "mmol"),
    (re.compile(r"\bq\s+d\b", re.I), "qd"),
    (re.compile(r"\bb\s+i\s+d\b", re.I), "bid"),
    (re.compile(r"\bt\s+i\s+d\b", re.I), "tid"),
    (re.compile(r"\bq\s+i\s+d\b", re.I), "qid"),
    (re.compile(r"\bp\s+o\b", re.I), "po"),
    (re.compile(r"\bi\s+v\b", re.I), "iv"),
    (re.compile(r"\bi\s+m\b", re.I), "im"),
)
HIGH_VALUE_SECTION_TERMS = (
    "摘要",
    "abstract",
    "引言",
    "前言",
    "背景",
    "结果",
    "分析",
    "讨论",
    "结论",
    "病例",
    "相互作用",
    "联用",
    "合用",
    "不良反应",
    "禁忌",
    "慎用",
    "风险",
    "监测",
    "临床意义",
    "用药建议",
    "注意事项",
)
LOW_VALUE_SECTION_TERMS = (
    "材料与方法",
    "对象与方法",
    "资料与方法",
    "实验方法",
    "方法",
    "研究方法",
    "数据来源",
    "统计学方法",
    "统计学处理",
    "仪器",
    "试剂",
    "蛋白质序列",
    "序列分析",
    "小分子的准备",
)
DROP_SECTION_TERMS = (
    "参考文献",
    "references",
    "作者简介",
    "基金项目",
    "收稿日期",
    "通信作者",
    "致谢",
)
MEDICAL_SIGNAL_TERMS = (
    "相互作用",
    "联用",
    "合用",
    "禁忌",
    "慎用",
    "不良反应",
    "风险",
    "监测",
    "剂量",
    "血药浓度",
    "疗效",
    "毒性",
    "cyp",
    "p-gp",
    "inr",
    "肝损伤",
    "肾损伤",
    "出血",
    "低血糖",
    "高血压",
    "药物代谢",
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = BROKEN_CJK_SPACE_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def strip_latex_artifacts(text: str) -> str:
    original = None
    while original != text:
        original = text
        text = LATEX_STYLE_RE.sub("", text)
        text = LATEX_COMMAND_RE.sub(r"\1", text)
        text = SIMPLE_BRACE_RE.sub(r"\1", text)

    replacements = {
        "\\sim": "～",
        "\\cdot": "·",
        "\\bullet": "·",
        "\\times": "x",
        "\\pm": "±",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\\[,:;! ]", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = text.replace("$", "")
    text = text.replace("^", "")
    text = text.replace("_", "")
    return text


def clean_inline_noise(text: str) -> str:
    text = FLOW_NOTE_RE.sub("", text)
    text = strip_latex_artifacts(text)
    text = re.sub(r"[（(]\s*本文编辑\s*[^）)]*[）)]", "", text)
    text = re.sub(r"\s*参考文献\s*[:：]\s*$", "", text, flags=re.I)
    text = DECIMAL_SPACE_RE.sub(".", text)
    text = RANGE_SPACE_RE.sub("～", text)
    text = re.sub(r"\s*~\s*", " ", text)
    for pattern, replacement in BROKEN_UNIT_PATTERNS:
        text = pattern.sub(replacement, text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = BROKEN_CJK_SPACE_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def normalize_compare_text(text: str) -> str:
    return COMPARE_TEXT_RE.sub("", text).lower()


def extract_expected_title(doc_name: str) -> str:
    if "_" in doc_name:
        return doc_name.rsplit("_", 1)[0].strip()
    return doc_name.strip()


def is_probable_author_line(line: str) -> bool:
    if len(line) > 50 or "。" in line or "：" in line or ":" in line:
        return False
    if not AUTHOR_LINE_RE.match(line):
        return False
    if AFFILIATION_RE.search(line):
        return False
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", line)
    return 2 <= len(chinese_chars) <= 15


def is_affiliation_line(line: str) -> bool:
    if len(line) > 120:
        return False
    return bool(AFFILIATION_RE.search(line)) and ("(" in line or "（" in line or "," in line or "，" in line)


def is_mostly_ascii_line(line: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", line):
        return False
    ascii_chars = re.findall(r"[A-Za-z0-9 ,.;:()\\/*_\-]", line)
    alpha_chars = re.findall(r"[A-Za-z]", line)
    if len(alpha_chars) < 10:
        return False
    return len(ascii_chars) / max(len(line), 1) >= 0.7


def looks_like_main_title(line: str) -> bool:
    text = line.strip()
    if not text or len(text) < 6 or len(text) > 40:
        return False
    if SECTION_HEADING_RE.match(text) or REFERENCE_RE.match(text):
        return False
    if any(token in text for token in DROP_SECTION_TERMS):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def strip_heading_prefix(text: str) -> str:
    return SECTION_HEADING_RE.sub("", text).strip("[]（）() \t")


def has_medical_signal(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in MEDICAL_SIGNAL_TERMS)


def looks_like_qa_heading(text: str) -> bool:
    stripped = text.strip()
    if not QA_HEADING_RE.match(stripped):
        return False
    return any(token in stripped for token in ("？", "?", "问", "什么", "如何", "是否", "能否", "为什么"))


def classify_section_type(text: str) -> tuple[str, str]:
    section_name = strip_heading_prefix(text) or text.strip()
    lowered = section_name.lower()

    if REFERENCE_RE.match(section_name) or any(term in lowered for term in DROP_SECTION_TERMS):
        return section_name, "drop"
    if any(term in lowered for term in LOW_VALUE_SECTION_TERMS):
        return section_name, "low_value"
    if any(term in lowered for term in HIGH_VALUE_SECTION_TERMS):
        return section_name, "high_value"
    return section_name, "neutral"


def trim_blocks_to_document(blocks: list[dict[str, Any]], doc_name: str) -> list[dict[str, Any]]:
    if not blocks:
        return blocks

    expected_title = extract_expected_title(doc_name)
    expected_key = normalize_compare_text(expected_title)
    start_idx = 0

    if expected_key:
        for idx, block in enumerate(blocks):
            text = block["text"].strip()
            text_key = normalize_compare_text(text)
            if not text_key:
                continue
            if expected_key in text_key or text_key in expected_key:
                start_idx = idx
                break

    trimmed: list[dict[str, Any]] = []
    started = False
    for idx, block in enumerate(blocks[start_idx:], start=start_idx):
        text = block["text"].strip()
        if not text:
            continue

        if not started:
            started = True

        if idx > start_idx and REFERENCE_RE.match(text):
            break

        if (
            idx > start_idx + 10
            and block.get("text_level") == 1
            and looks_like_main_title(text)
            and expected_key
            and expected_key not in normalize_compare_text(text)
        ):
            break

        trimmed.append(block)

    return trimmed or blocks


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
    if EN_ABSTRACT_RE.match(line):
        return True
    if LEADING_META_RE.match(line):
        return True
    if FLOW_NOTE_RE.fullmatch(line):
        return True
    if META_LINE_RE.search(line) and len(line) < 120:
        return True
    if is_probable_author_line(line):
        return True
    if is_affiliation_line(line):
        return True
    if is_mostly_ascii_line(line) and len(line) < 1500:
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
        text = clean_inline_noise(text)
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
        para = clean_inline_noise(para)
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


def split_embedded_qa_units(units: list[dict[str, str]]) -> list[dict[str, str]]:
    refined: list[dict[str, str]] = []
    pattern = re.compile(r"(?m)(?=^\s*\d{1,4}[.\u3001\uff0e]\s*)")

    for unit in units:
        text = unit["text"].strip()
        if not text:
            continue
        if not QA_HEADING_RE.match(text):
            refined.append(unit)
            continue

        parts = [part.strip() for part in pattern.split(text) if part.strip()]
        if len(parts) <= 1:
            refined.append(unit)
            continue
        for part in parts:
            refined.append(
                {
                    "text": part,
                    "section": unit.get("section", "问答"),
                    "section_type": unit.get("section_type", "qa"),
                }
            )

    return refined


def is_qa_document(paragraphs: list[str]) -> bool:
    qa_count = sum(1 for para in paragraphs if looks_like_qa_heading(para.strip()))
    if qa_count >= 2:
        return True
    joined = "\n".join(paragraphs[:5])
    if "问题与解答" in joined or "问答" in joined:
        return True
    return False


def build_qa_units(paragraphs: list[str]) -> list[dict[str, str]]:
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
    return [
        {
            "text": unit,
            "section": "问答",
            "section_type": "qa",
        }
        for unit in ordered
    ]


def build_semantic_units(paragraphs: list[str]) -> list[dict[str, str]]:
    if is_qa_document(paragraphs):
        return build_qa_units(paragraphs)

    units: list[dict[str, str]] = []
    current = ""
    current_kind = ""
    current_section = "正文"
    current_section_type = "neutral"

    for para in paragraphs:
        text = para.strip()
        if not text:
            continue

        is_qa = looks_like_qa_heading(text)
        is_heading = bool(SECTION_HEADING_RE.match(text)) and len(text) <= 60

        if is_qa:
            if current.strip():
                units.append(
                    {
                        "text": current.strip(),
                        "section": current_section,
                        "section_type": current_section_type,
                    }
                )
            current = text
            current_kind = "qa"
            current_section = "问答"
            current_section_type = "qa"
            continue

        if is_heading:
            if current.strip():
                units.append(
                    {
                        "text": current.strip(),
                        "section": current_section,
                        "section_type": current_section_type,
                    }
                )
            current_section, current_section_type = classify_section_type(text)
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

        units.append(
            {
                "text": current.strip(),
                "section": current_section,
                "section_type": current_section_type,
            }
        )
        current = text
        current_kind = "text"

    if current.strip():
        units.append(
            {
                "text": current.strip(),
                "section": current_section,
                "section_type": current_section_type,
            }
        )
    return split_embedded_qa_units(units)


def filter_semantic_units(units: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for unit in units:
        text = unit["text"].strip()
        section_type = unit.get("section_type", "neutral")
        if not text:
            continue
        if section_type == "drop":
            continue
        if should_drop_text(text) and section_type not in {"high_value", "qa"}:
            continue
        if section_type == "low_value" and not has_medical_signal(text):
            continue
        filtered.append(unit)
    return filtered


def chunk_text(paragraphs: list[str], chunk_size: int, overlap: int) -> list[dict[str, str]]:
    units = build_semantic_units(paragraphs)
    units = filter_semantic_units(units)
    chunks: list[dict[str, str]] = []
    current_parts: list[dict[str, str]] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        text = "\n".join(part["text"] for part in current_parts).strip()
        if not text:
            current_parts = []
            current_len = 0
            return
        sections = {part.get("section", "正文") for part in current_parts}
        section_types = {part.get("section_type", "neutral") for part in current_parts}
        chunks.append(
            {
                "text": text,
                "section": next(iter(sections)) if len(sections) == 1 else "mixed",
                "section_type": next(iter(section_types)) if len(section_types) == 1 else "mixed",
            }
        )
        current_parts = []
        current_len = 0

    for unit in units:
        text = unit["text"]
        if len(text) > chunk_size:
            oversized_parts = split_long_text(text, chunk_size=chunk_size, overlap=overlap)
        else:
            oversized_parts = [text]

        for part in oversized_parts:
            candidate = {
                "text": part,
                "section": unit.get("section", "正文"),
                "section_type": unit.get("section_type", "neutral"),
            }
            if not current_parts:
                current_parts.append(candidate)
                current_len = len(part)
                continue
            same_section = current_parts[-1].get("section") == candidate["section"]
            if (
                len(part) + current_len + 1 <= chunk_size
                and (same_section or current_len < int(chunk_size * 0.6))
            ):
                current_parts.append(candidate)
                current_len += len(part) + 1
                continue
            flush_current()
            current_parts.append(candidate)
            current_len = len(part)

    flush_current()
    return chunks


def process_file(path: Path, clean_dir: Path, chunk_dir: Path, chunk_size: int, overlap: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    blocks = extract_candidate_blocks(data)
    doc_name = path.parent.name
    trimmed_blocks = trim_blocks_to_document(blocks, doc_name=doc_name)
    paragraphs = merge_blocks_to_paragraphs(trimmed_blocks)
    cleaned_text = "\n".join(paragraphs).strip()
    chunks = chunk_text(paragraphs, chunk_size=chunk_size, overlap=overlap)

    clean_path = clean_dir / f"{doc_name}.txt"
    clean_path.write_text(cleaned_text, encoding="utf-8")

    doc_chunk_dir = chunk_dir / doc_name
    doc_chunk_dir.mkdir(parents=True, exist_ok=True)
    for old_file in doc_chunk_dir.glob("chunk_*.json"):
        old_file.unlink()

    for idx, chunk in enumerate(chunks):
        payload = [
            {
                "text": chunk["text"],
                "source": f"{doc_name}.pdf",
                "chunk_index": f"{doc_name}:{idx}",
                "section": chunk.get("section", "正文"),
                "section_type": chunk.get("section_type", "neutral"),
            }
        ]
        out_path = doc_chunk_dir / f"chunk_{idx}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "file": str(path),
        "doc_name": doc_name,
        "raw_blocks": len(data),
        "kept_blocks": len(blocks),
        "trimmed_blocks": len(trimmed_blocks),
        "paragraphs": len(paragraphs),
        "clean_chars": len(cleaned_text),
        "chunks": len(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MinerU JSON outputs into text and chunk JSON.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
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
