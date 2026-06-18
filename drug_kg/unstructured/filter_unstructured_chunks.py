"""Filter unstructured text chunks before triple extraction.

This script applies:
1. Rule-based hard filtering
2. Optional DeepSeek LLM judging for borderline chunks

Input:
  data_process/txt/chunk_*.json

Output:
  - kept chunk files under output dir
  - decisions jsonl
  - summary json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2a70ab5f703d4c929ec8860ffab46b9a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


MEDICAL_SIGNAL_PATTERNS = [
    r"药",
    r"用药",
    r"相互作用",
    r"禁忌",
    r"慎用",
    r"适应症",
    r"不良反应",
    r"副作用",
    r"监测",
    r"剂量",
    r"给药",
    r"合用",
    r"治疗",
    r"症状",
    r"疾病",
    r"血糖",
    r"INR",
    r"肝功能",
    r"肾功能",
]

TRIPLE_SIGNAL_PATTERNS = [
    r"与.{0,12}(合用|联用|同用|同服)",
    r"(增加|降低|减弱|增强).{0,16}(风险|浓度|药效|暴露|作用)",
    r"(禁用|慎用|不宜用于|避免用于)",
    r"(适用于|用于治疗|可治疗|治疗)",
    r"(可引起|可能导致|不良反应|副作用)",
    r"(需要监测|监测.{0,12}(血糖|INR|肝功能|肾功能|血药浓度))",
    r"(影响|升高|降低).{0,12}(血糖|INR|肝功能|肾功能|血药浓度)",
]

REFERENCE_PATTERNS = [
    r"\[\d+\]",
    r"[Jj]\]\.?",
    r"et al\.",
    r"doi[:：]",
    r"参考文献",
    r"学报",
    r"杂志",
    r"vol\.",
]

NEWS_NOISE_PATTERNS = [
    r"检察院",
    r"警方",
    r"法院",
    r"起诉",
    r"犯罪嫌疑人",
    r"微博",
    r"微信",
    r"热搜",
    r"网红",
    r"记者",
    r"通报",
]

METHOD_PATTERNS = [
    r"样本量",
    r"随机交叉试验",
    r"平行试验",
    r"试验设计",
    r"药代动力学",
    r"半衰期",
    r"PBPK",
    r"受试者",
    r"研究方法",
]

PROMOTION_PATTERNS = [
    r"直播预告",
    r"报名",
    r"公开课",
    r"讲座",
    r"课程",
    r"扫码",
    r"点击",
    r"关注",
    r"免费活动",
    r"报名截止",
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _post_chat_completions(*, headers: dict[str, str], payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    resp = requests.post(
        _chat_completions_url(DEEPSEEK_BASE_URL),
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_json(text: str) -> dict[str, Any]:
    text = _norm(text)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    return json.loads(text)


def rule_filter(text: str, min_chars: int) -> tuple[bool, str]:
    text = _norm(text)
    if len(text) < min_chars:
        return False, "too_short"

    ref_hits = sum(1 for pattern in REFERENCE_PATTERNS if re.search(pattern, text, re.I))
    med_hits = sum(1 for pattern in MEDICAL_SIGNAL_PATTERNS if re.search(pattern, text, re.I))
    triple_hits = sum(1 for pattern in TRIPLE_SIGNAL_PATTERNS if re.search(pattern, text, re.I))
    news_hits = sum(1 for pattern in NEWS_NOISE_PATTERNS if re.search(pattern, text, re.I))
    method_hits = sum(1 for pattern in METHOD_PATTERNS if re.search(pattern, text, re.I))
    promo_hits = sum(1 for pattern in PROMOTION_PATTERNS if re.search(pattern, text, re.I))

    if ref_hits >= 3 and med_hits <= 2:
        return False, "reference_like"
    if news_hits >= 2 and med_hits <= 2:
        return False, "news_noise"
    if promo_hits >= 2 and triple_hits == 0:
        return False, "promotion_noise"
    if method_hits >= 2 and triple_hits == 0:
        return False, "methodology_noise"
    if med_hits >= 1 and triple_hits == 0 and method_hits >= 1:
        return False, "weak_medical_background"
    if med_hits >= 2 and triple_hits == 0 and len(text) > 300:
        return False, "no_explicit_triple_signal"

    return True, "rule_pass"


def llm_filter(*, text: str, source: str, chunk_index: Any, model: str, timeout_s: int) -> tuple[bool, str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable")

    system_prompt = """你是一个用于药物知识图谱预过滤的审核器。

你的任务是判断一段非结构化文本，是否值得进入“药物知识三元组抽取”流程。

保留（keep=true）的标准：
1. 文本中包含明确的医疗/用药事实。
2. 这些事实必须能够比较直接地抽成药物知识图谱三元组。
3. 至少应较明确地包含以下之一：
   - 药物与药物/食物相互作用
   - 药物适应症
   - 药物禁忌/慎用人群
   - 药物不良反应
   - 药物影响或需要监测的指标
4. 即使是科普或新闻，也只有在其中存在上述“可直接成三元组”的事实时才保留。

丢弃（keep=false）的标准：
1. 主要是新闻人物、案件、机构、传播、营销、舆情内容。
2. 主要是参考文献、尾注、文献列表。
3. 主要是研究方法学、试验设计、统计描述，而不是具体药物事实。
4. 主要是背景介绍、概念综述、机制综述，但缺少可直接落三元组的明确事实。
5. 基本无法抽出可入药物知识图谱的实体关系。

只输出一个 JSON：
{
  "keep": true,
  "reason": "一句简短原因",
  "category": "medical_fact|reference|news_noise|methodology|other"
}
"""
    user_prompt = json.dumps(
        {
            "source": source,
            "chunk_index": chunk_index,
            "text": text[:1800],
        },
        ensure_ascii=False,
        indent=2,
    )
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    obj = _post_chat_completions(headers=headers, payload=payload, timeout_s=timeout_s)
    content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json(content)
    keep = bool(parsed.get("keep"))
    category = _norm(parsed.get("category") or "other")
    reason = _norm(parsed.get("reason") or category or "llm_decision")
    return keep, f"{category}:{reason}"


def process_file(
    *,
    input_path: Path,
    output_dir: Path,
    decisions_path: Path,
    failed_path: Path,
    use_llm: bool,
    model: str,
    timeout_s: int,
    min_chars: int,
    force: bool,
    max_workers: int,
) -> tuple[int, int]:
    output_path = output_dir / input_path.name
    if output_path.exists() and not force:
        logger.info("Skip existing kept file: %s", output_path)
        return 0, 0

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    kept_records: list[dict[str, Any]] = []
    existing_done_keys: set[str] = set()
    total = 0

    if decisions_path.exists() and not force:
        with decisions_path.open("r", encoding="utf-8") as df:
            for line in df:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _norm(item.get("file")) != input_path.name:
                    continue
                existing_done_keys.add(f"{input_path.name}::{_norm(item.get('record_index'))}::{_norm(item.get('chunk_index'))}")

    pending_records: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        total += 1
        chunk_index = record.get("chunk_index", idx)
        done_key = f"{input_path.name}::{idx}::{_norm(chunk_index)}"
        if done_key in existing_done_keys:
            continue
        pending_records.append((idx, record))

    def _flush_decision(result: dict[str, Any]) -> None:
        with decisions_path.open("a", encoding="utf-8") as df:
            df.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _flush_failure(result: dict[str, Any]) -> None:
        with failed_path.open("a", encoding="utf-8") as ff:
            ff.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _run_one(idx: int, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        text = _norm(record.get("text"))
        source = _norm(record.get("source"))
        chunk_index = record.get("chunk_index", idx)
        keep, reason = rule_filter(text, min_chars)
        decision_stage = "rule"
        error_payload = None

        if keep and use_llm:
            try:
                keep, reason = llm_filter(
                    text=text,
                    source=source,
                    chunk_index=chunk_index,
                    model=model,
                    timeout_s=timeout_s,
                )
                decision_stage = "llm"
            except Exception as exc:
                keep = False
                reason = f"llm_error:{exc}"
                decision_stage = "llm"
                error_payload = {
                    "file": input_path.name,
                    "record_index": idx,
                    "chunk_index": chunk_index,
                    "source": source,
                    "error": str(exc),
                    "text_preview": text[:200],
                }

        decision = {
            "file": input_path.name,
            "record_index": idx,
            "chunk_index": chunk_index,
            "source": source,
            "keep": keep,
            "stage": decision_stage,
            "reason": reason,
            "text_preview": text[:200],
        }
        return decision, error_payload

    if max_workers <= 1:
        for idx, record in pending_records:
            decision, error_payload = _run_one(idx, record)
            if decision.get("keep"):
                kept_records.append(record)
            _flush_decision(decision)
            if error_payload is not None:
                _flush_failure(error_payload)
    else:
        results_by_idx: dict[int, tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_one, idx, record): (idx, record)
                for idx, record in pending_records
            }
            for future in concurrent.futures.as_completed(future_map):
                idx, record = future_map[future]
                decision, error_payload = future.result()
                results_by_idx[idx] = (decision, error_payload, record)

        for idx in sorted(results_by_idx):
            decision, error_payload, record = results_by_idx[idx]
            if decision.get("keep"):
                kept_records.append(record)
            _flush_decision(decision)
            if error_payload is not None:
                _flush_failure(error_payload)

    if kept_records:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(kept_records, ensure_ascii=False, indent=2), encoding="utf-8")
        return total, len(kept_records)

    return total, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter unstructured chunks before KG extraction")
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "data_process" / "txt")
    parser.add_argument("--pattern", default="chunk_*.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "filtered_chunks")
    parser.add_argument("--decisions-file", type=Path, default=Path(__file__).resolve().parent / "filtered_chunks.decisions.jsonl")
    parser.add_argument("--summary-file", type=Path, default=Path(__file__).resolve().parent / "filtered_chunks.summary.json")
    parser.add_argument("--model", default=DEEPSEEK_MODEL_NAME or MODEL_NAME)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count for LLM filtering")
    parser.add_argument("--use-llm", action="store_true", help="Call DeepSeek after rule filter")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    input_paths = sorted(args.input_dir.glob(args.pattern))
    if args.limit > 0:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        logger.warning("No input files found")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.force and args.decisions_file.exists():
        args.decisions_file.unlink()
    failed_path = args.decisions_file.with_suffix(".failed.jsonl")
    if args.force and failed_path.exists():
        failed_path.unlink()

    total_records = 0
    kept_records = 0
    kept_files = 0
    if max(1, args.max_workers) <= 1:
        for input_path in input_paths:
            total, kept = process_file(
                input_path=input_path,
                output_dir=args.output_dir,
                decisions_path=args.decisions_file,
                failed_path=failed_path,
                use_llm=args.use_llm,
                model=args.model,
                timeout_s=args.timeout_s,
                min_chars=args.min_chars,
                force=args.force,
                max_workers=1,
            )
            total_records += total
            kept_records += kept
            if kept > 0:
                kept_files += 1
            logger.info("Processed %s total=%s kept=%s", input_path.name, total, kept)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {
                executor.submit(
                    process_file,
                    input_path=input_path,
                    output_dir=args.output_dir,
                    decisions_path=args.decisions_file,
                    failed_path=failed_path,
                    use_llm=args.use_llm,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    min_chars=args.min_chars,
                    force=args.force,
                    max_workers=1,
                ): input_path
                for input_path in input_paths
            }
            for future in concurrent.futures.as_completed(future_map):
                input_path = future_map[future]
                total, kept = future.result()
                total_records += total
                kept_records += kept
                if kept > 0:
                    kept_files += 1
                logger.info("Processed %s total=%s kept=%s", input_path.name, total, kept)

    summary = {
        "input_files": len(input_paths),
        "kept_files": kept_files,
        "total_records": total_records,
        "kept_records": kept_records,
        "keep_ratio": (kept_records / total_records) if total_records else 0.0,
        "use_llm": args.use_llm,
        "max_workers": max(1, args.max_workers),
        "output_dir": str(args.output_dir),
        "decisions_file": str(args.decisions_file),
        "failed_file": str(failed_path),
    }
    args.summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
