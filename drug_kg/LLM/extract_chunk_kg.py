"""Use an LLM to extract entities and relations from cut/txt chunk files.

Outputs one JSON file per input chunk under `LLM/output/` with the same name.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


# project root: .../langchain (this file is .../langchain/drug_kg/LLM/extract_chunk_kg.py)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-5d54452178d54322b9b1bbce96e2a135")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


OUTPUT_SCHEMA = {
    "entities": [
        {
            "id": "ent_1",
            "type": "Drug",
            "name": "",
            "properties": {},
        }
    ],
    "relations": [
        {
            "id": "rel_1",
            "type": "INTERACTS_WITH",
            "source_id": "ent_1",
            "target_id": "int_1",
            "name": "喝了酒不能吃头孢",
        }
    ]
}


def load_modeling_rules(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_system_prompt(modeling_rules: str) -> str:
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""你是一个医学知识图谱抽取器。

你必须严格遵守下面的知识图谱建模规则，从输入文本中抽取实体与关系（边）。

【知识图谱建模规则】
{modeling_rules}

【抽取要求】
1. 只抽取文本中明确表达或可直接严格推出的信息，不要臆造。
2. 优先使用建模规则中的实体类型与关系类型（但本任务**不创建 Interaction 节点**，相互作用直接表示为参与者之间的一条边）。
   - 实体类型建议：Drug, DrugClass, Food, Enzyme, Mechanism, Disease, SideEffect, Symptom, Indicator, Population
   - 关系类型建议：IN_CLASS, INDICATED_FOR, HAS_ADVERSE_REACTION, HAS_MECHANISM, INCREASES_RISK_OF, AFFECTS_INDICATOR, HAS_SYMPTOM, CONTRAINDICATED_FOR, APPLIES_TO
3. 当文本表达“药物相互作用/食物相互作用”等事实时：
   - 用一条边表达：(参与者A)-[:INTERACTS_WITH]->(参与者B)
   - 把机制、后果、建议、严重程度、置信度、证据等信息放到该边的 `properties` 里（参考 schema）。
   - `properties.evidence` 必须包含 chunk_index/source/text（text 为原文片段，优先用原句）。
5. 若某个字段无法确定，使用空字符串、空数组或空对象，不要编造。
6. `id` 必须只在当前记录内唯一，推荐使用 `ent_1`、`int_1`、`rel_1`、`ev_1` 这类格式。
9. 关系中的 `source_id` 和 `target_id` 必须引用本次输出里的已有节点 id。
10. 如果文本没有明确的相互作用，可只抽取实体，`relations` 为空数组。

【输出格式】
在输出前先做“反思自检”（只在脑中完成，不要写出来）：
1. 你最终返回的内容必须是“严格合法 JSON”，可以被 `json.loads` 解析。
2. 只能输出 JSON 对象本身，不能包含任何解释、Markdown、代码块、前后缀文字。
3. 必须至少包含顶层键 `entities` 和 `relations`（缺失时补空数组）。

只输出 JSON，不要输出解释、Markdown、代码块或额外文字。
输出必须是一个对象，结构如下：
{schema_text}
"""


def build_user_prompt(record: dict[str, Any], record_index: int) -> str:
    payload = {
        "record_index": record_index,
        "chunk_index": record.get("chunk_index"),
        "source": record.get("source", ""),
        "text": record.get("text", ""),
    }
    return (
        "请根据给定文本抽取知识图谱实体、相互作用事件和关系，并严格返回 JSON。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def extract_json(text: str) -> dict[str, Any]:
    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if not s.startswith("```"):
            return s
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.S)
        return match.group(1).strip() if match else s

    def _extract_first_json_object(s: str) -> str | None:
        # Find the first balanced {...} object, respecting quoted strings.
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    def _cleanup_common_json_issues(s: str) -> str:
        # Remove trailing commas: { "a": 1, }  or  [1,2,]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # LLM 常见：对象里两个属性之间少了逗号 —— `}\n    "next_key"` 应为 `},\n    "next_key"`
        s = re.sub(r"}\s*\n(\s*)\"", r"},\n\1\"", s)
        # 数组里相邻对象/元素之间：`}\n    {` 应为 `},\n    {`
        s = re.sub(r"}\s*\n(\s*)\{", r"},\n\1{", s)
        # 数组里 `]\n    [` 或 `]\n    {`
        s = re.sub(r"]\s*\n(\s*)([\[{])", r"],\n\1\2", s)
        # Remove BOM if present
        return s.lstrip("\ufeff").strip()

    def _try_json_repair(s: str) -> dict[str, Any] | None:
        """LLM 常产出缺逗号、尾逗号等；json-repair 可修大部分（可选依赖）。"""
        try:
            import json_repair  # type: ignore[import-untyped]
        except ImportError:
            return None
        try:
            return json_repair.loads(s)
        except Exception:
            return None

    text = _strip_code_fence(text)
    if not text.strip():
        raise json.JSONDecodeError("Empty response content", text, 0)

    candidate = _extract_first_json_object(text) or text.strip()
    candidate = _cleanup_common_json_issues(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _try_json_repair(candidate)
        if repaired is not None:
            return repaired
        # Fallback: greedy regex (last resort)
        match = re.search(r"(\{[\s\S]*\})", text, re.S)
        if not match:
            raise
        blob = _cleanup_common_json_issues(match.group(1))
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            repaired2 = _try_json_repair(blob)
            if repaired2 is not None:
                return repaired2
            raise


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat/completions"


def _post_chat_completions(*, headers: dict[str, str], payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    resp = requests.post(
        _chat_completions_url(DEEPSEEK_BASE_URL),
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _get_choice_content(obj: dict[str, Any]) -> str:
    return (
        obj.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def process_record(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout_s: int = 120,
) -> dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    def _call_llm(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": model,
            "temperature": 0,
            "messages": messages,
        }
        obj = _post_chat_completions(headers=headers, payload=payload, timeout_s=timeout_s)
        return _get_choice_content(obj)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_llm(base_messages)
    try:
        data = extract_json(content)
    except json.JSONDecodeError:
        # Second pass: ask the model to reformat into strict JSON only.
        repair_system = (
            "你是一个 JSON 修复器。你将收到一段可能不完整/不合法的 JSON 或混入了额外文本的内容。\n"
            "任务：只输出一个**严格合法**的 JSON 对象，键必须包含 entities 和 relations（若缺失则给空数组）。\n"
            "在输出前先做“反思自检”（只在脑中完成，不要写出来）：\n"
            "1) 结果必须能被严格 `json.loads` 解析；\n"
            "2) 只能输出 JSON 对象本身；\n"
            "3) 顶层必须有 entities 和 relations。\n"
            "不要输出解释、Markdown、代码块或任何额外文字。"
        )
        repair_user = (
            "把下面内容修复为严格合法 JSON（只输出 JSON 对象本身）：\n\n"
            + content
        )
        repaired = _call_llm(
            [
                {"role": "system", "content": repair_system},
                {"role": "user", "content": repair_user},
            ]
        )
        data = extract_json(repaired)

    data.setdefault("entities", [])
    data.setdefault("relations", [])
    return data


def process_file(
    *,
    system_prompt: str,
    model: str,
    input_path: Path,
    output_path: Path,
) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    results = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or not record.get("text"):
            continue
        logger.info("Processing %s record %s", input_path.name, idx)
        user_prompt = build_user_prompt(record, idx)
        try:
            data = process_record(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
            )
        except Exception as e:
            logger.exception("Record failed: file=%s record=%s err=%s", input_path.name, idx, e)
            # keep going; do not abort the whole file
            continue
        # attach minimal trace fields (so downstream can join back)
        data.setdefault("chunk_index", record.get("chunk_index"))
        data.setdefault("source", record.get("source", ""))
        data.setdefault("text", record.get("text", ""))
        results.append(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM entity/relation extraction for cut/txt JSON files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "data_process" / "txt",
        help="Input directory containing chunk JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory for extracted chunk JSON files",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=ROOT_DIR / "knowledge_graph_model.txt",
        help="Modeling rules file used to build the prompt",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for input files",
    )
    parser.add_argument(
        "--model",
        default=DEEPSEEK_MODEL_NAME or MODEL_NAME,
        help="LLM model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N files when > 0",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if output file already exists (overwrite)",
    )
    # Backward-compatible flag (previous behavior required opt-in skipping).
    # Now skipping existing outputs is the default; use --force to override.
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="(deprecated) Kept for compatibility; skipping is now default",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.rules_file.is_file():
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")

    modeling_rules = load_modeling_rules(args.rules_file)
    system_prompt = build_system_prompt(modeling_rules)

    input_paths = sorted(args.input_dir.glob(args.pattern))
    if args.limit > 0:
        input_paths = input_paths[: args.limit]

    if not input_paths:
        logger.warning("No input files matched %s in %s", args.pattern, args.input_dir)
        return

    logger.info("Found %s files to process", len(input_paths))
    for input_path in input_paths:
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.force:
            logger.info("Skip existing %s (use --force to overwrite)", output_path)
            continue
        try:
            process_file(
                system_prompt=system_prompt,
                model=args.model,
                input_path=input_path,
                output_path=output_path,
            )
            # 每处理完一个 chunk 文件就立刻写出（process_file 内部已写），这里再打日志确认
            logger.info("Wrote %s", output_path)
        except Exception as e:
            logger.exception("File failed: %s err=%s", input_path, e)
            # continue next file
            continue


if __name__ == "__main__":
    main()
