"""Extract medication-related triples from unstructured text chunks with LLM."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field, ValidationError, model_validator


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2a70ab5f703d4c929ec8860ffab46b9a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


OUTPUT_SCHEMA = {
    "entities": [
        {
            "id": "ent_1",
            "type": "Drug",
            "name": "阿司匹林",
            "canonical_name": "阿司匹林",
            "properties": {},
        }
    ],
    "relations": [
        {
            "id": "rel_1",
            "type": "INTERACTS_WITH",
            "source_id": "ent_1",
            "target_id": "ent_2",
            "properties": {
                "source": "某公众号",
                "chunk_index": "12",
                "raw_text": "与华法林合用可能增加出血风险",
            },
        }
    ],
}

GENERIC_ENTITY_NAMES = {
    "患者",
    "医生",
    "医院",
    "研究",
    "结果",
    "风险",
    "作用",
    "说明",
    "资料",
    "情况",
    "建议",
    "禁忌",
    "慎用",
    "不宜",
    "尚不明确",
}
LONG_ENTITY_PUNCT = ("，", "。", "；", ";", "：", ":", "、", "（", "）", "(", ")")

EntityType = Literal[
    "Drug",
    "DrugClass",
    "Food",
    "Disease",
    "SideEffect",
    "Symptom",
    "Indicator",
    "Population",
]

RelationType = Literal[
    "IN_CLASS",
    "INDICATED_FOR",
    "HAS_ADVERSE_REACTION",
    "AFFECTS_INDICATOR",
    "HAS_SYMPTOM",
    "CONTRAINDICATED_FOR",
    "APPLIES_TO",
    "INTERACTS_WITH",
]


class EntityModel(BaseModel):
    id: str
    type: EntityType
    name: str
    canonical_name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationModel(BaseModel):
    id: str
    type: RelationType
    source_id: str
    target_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractionResultModel(BaseModel):
    entities: list[EntityModel] = Field(default_factory=list)
    relations: list[RelationModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relation_refs(self) -> "ExtractionResultModel":
        entity_ids = {entity.id for entity in self.entities}
        self.relations = [
            relation
            for relation in self.relations
            if relation.source_id in entity_ids and relation.target_id in entity_ids
        ]
        return self


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_modeling_rules(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_system_prompt(modeling_rules: str) -> str:
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""你是一个严格按照药物知识图谱模式进行抽取的结构化信息抽取器。

【知识图谱建模规则】
{modeling_rules}

【任务】
从输入的非结构化文本 chunk 中抽取“医疗/药物相关”的实体与关系，最终生成可直接落库的三元组。

【抽取边界】
1. 只抽与药物、疾病、症状、不良反应、监测指标、人群、食物/补充剂、药物类别直接相关的事实。
2. 对新闻报道、公众号、科普文章中的人物、机构、时间、地点、案件经过，不要建实体。
3. 如果文本主体不是具体药物知识，而是社会新闻，只保留其中明确的药物事实部分。
4. 只保留可以支持用药问答和医疗知识图谱的事实，不要抽舆情、传播、营销、司法过程。

【允许的实体类型】
只允许：
`Drug`, `DrugClass`, `Food`, `Disease`, `SideEffect`, `Symptom`, `Indicator`, `Population`

【允许的关系类型】
只允许：
`IN_CLASS`, `INDICATED_FOR`, `HAS_ADVERSE_REACTION`, `AFFECTS_INDICATOR`, `HAS_SYMPTOM`, `CONTRAINDICATED_FOR`, `APPLIES_TO`, `INTERACTS_WITH`

【抽取要求】
1. 只抽文本中明确表达、或可直接严格推出的信息，不脑补。
2. 不创建事件节点，不创建“研究”“新闻”“案件”“医生建议”等抽象节点。
3. 不要把整句整段说明当实体。
4. 一条关系必须能落成一个清晰三元组。
5. 如果无法确定关系，可以保留核心实体，但不要强造关系。
6. 优先抽取这些高价值事实：
   - 药物与药物/食物相互作用
   - 药物适应症
   - 药物禁忌/慎用人群
   - 药物不良反应
   - 药物影响/需要监测的指标
7. 每条关系的 `properties` 中尽量保留：
   - `source`
   - `chunk_id`
   - `chunk_index`
   - `raw_text`

【输出要求】
1. 只输出一个合法 JSON 对象。
2. 顶层必须包含 `entities` 和 `relations`。
3. 只保留参与三元组或强相关的核心实体。
4. 不要输出解释、Markdown、额外说明。

输出结构如下：
{schema_text}
"""


def build_user_prompt(record: dict[str, Any], record_index: int) -> str:
    payload = {
        "record_index": record_index,
        "chunk_index": record.get("chunk_index", record_index),
        "source": record.get("source", ""),
        "text": record.get("text", ""),
    }
    return (
        "请基于下面这段非结构化文本抽取药物知识图谱三元组。\n"
        "要求：\n"
        "1. 只抽医疗/用药事实。\n"
        "2. 过滤人物、机构、时间、地点、案件过程、营销宣传。\n"
        "3. 只输出 JSON。\n\n"
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
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(s[start:], start=start):
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
                    return s[start : i + 1]
        return None

    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        first_obj = _extract_first_json_object(candidate)
        if first_obj is None:
            raise
        return json.loads(first_obj)


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


def _get_choice_content(obj: dict[str, Any]) -> str:
    return obj.get("choices", [{}])[0].get("message", {}).get("content", "")


def process_record(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout_s: int = 120,
    max_parse_retries: int = 1,
) -> dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_exc: Exception | None = None
    last_content = ""
    for attempt in range(max_parse_retries + 1):
        obj = _post_chat_completions(headers=headers, payload=payload, timeout_s=timeout_s)
        content = _get_choice_content(obj)
        last_content = content
        try:
            data = extract_json(content)
            validated = ExtractionResultModel.model_validate(data)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
            logger.warning("Structured parse failed on attempt %s/%s: %s", attempt + 1, max_parse_retries + 1, exc)
            if attempt >= max_parse_retries:
                break
    snippet = _norm(last_content)[:800]
    raise RuntimeError(
        f"Invalid structured output from model after {max_parse_retries + 1} attempts: {last_exc}; "
        f"response_snippet={snippet}"
    ) from last_exc


def _looks_like_bad_entity(entity_name: str) -> bool:
    text = _norm(entity_name)
    if not text:
        return True
    if text in GENERIC_ENTITY_NAMES:
        return True
    if len(text) > 30:
        return True
    if any(punct in text for punct in LONG_ENTITY_PUNCT):
        return True
    if re.fullmatch(r"[0-9一二三四五六七八九十年月日.%\-]+", text):
        return True
    if any(token in text for token in ("检察院", "警方", "记者", "微博", "微信", "热搜", "网红")):
        return True
    return False


def _relation_properties_from_record(record: dict[str, Any], raw_text: str) -> dict[str, Any]:
    return {
        "chunk_id": _norm(record.get("chunk_index", "")),
        "data": _norm(raw_text or record.get("text", ""))[:1000],
    }


def _post_process_extraction(data: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relations = data.get("relations") if isinstance(data.get("relations"), list) else []

    dedup_entities: dict[tuple[str, str], dict[str, Any]] = {}
    entity_id_remap: dict[str, str] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        old_id = _norm(entity.get("id"))
        entity_type = _norm(entity.get("type") or "Entity")
        name = _norm(entity.get("name"))
        canonical_name = _norm(entity.get("canonical_name") or name)
        if not old_id or not name or _looks_like_bad_entity(name):
            continue
        key = (entity_type, canonical_name or name)
        if key not in dedup_entities:
            new_id = f"ent_{len(dedup_entities) + 1}"
            dedup_entities[key] = {
                "id": new_id,
                "type": entity_type,
                "name": name,
                "canonical_name": canonical_name or name,
                "properties": entity.get("properties") or {},
            }
        entity_id_remap[old_id] = dedup_entities[key]["id"]

    valid_entity_ids = {_norm(entity["id"]) for entity in dedup_entities.values()}
    cleaned_relations: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        rel_type = _norm(relation.get("type"))
        source_id = entity_id_remap.get(_norm(relation.get("source_id")), "")
        target_id = entity_id_remap.get(_norm(relation.get("target_id")), "")
        if not rel_type or source_id not in valid_entity_ids or target_id not in valid_entity_ids or source_id == target_id:
            continue
        rel_key = (source_id, rel_type, target_id)
        if rel_key in seen_relations:
            continue
        seen_relations.add(rel_key)
        props = dict(relation.get("properties") or {})
        props.setdefault("chunk_id", _norm(record.get("chunk_index", "")))
        props.setdefault("data", _norm(record.get("text", ""))[:1000])
        cleaned_relations.append(
            {
                "id": _norm(relation.get("id") or f"rel_{len(cleaned_relations) + 1}"),
                "type": rel_type,
                "source_id": source_id,
                "target_id": target_id,
                "properties": props,
            }
        )

    referenced_ids: set[str] = set()
    for relation in cleaned_relations:
        referenced_ids.add(_norm(relation.get("source_id")))
        referenced_ids.add(_norm(relation.get("target_id")))
    final_entities = [
        entity
        for entity in dedup_entities.values()
        if _norm(entity.get("id")) in referenced_ids
    ]
    data["entities"] = final_entities
    data["relations"] = cleaned_relations
    return data


def process_file(
    *,
    system_prompt: str,
    model: str,
    input_path: Path,
    output_path: Path,
    limit: int = 0,
    max_workers: int = 1,
) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    if limit > 0:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = output_path.with_suffix(".failed.jsonl")
    success_jsonl_path = output_path.with_suffix(".success.jsonl")
    existing_results: list[dict[str, Any]] = []
    existing_chunk_ids: set[str] = set()

    if success_jsonl_path.exists():
        with success_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    existing_results.append(item)
    elif output_path.exists():
        try:
            existing_raw = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing_raw, list):
                existing_results = [item for item in existing_raw if isinstance(item, dict)]
        except json.JSONDecodeError:
            logger.warning("Existing output is invalid JSON, starting fresh append set: %s", output_path)

    existing_chunk_ids = {
        _norm(item.get("chunk_index"))
        for item in existing_results
        if _norm(item.get("chunk_index"))
    }

    pending_records: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        chunk_id = _norm(record.get("chunk_index", idx))
        if chunk_id and chunk_id in existing_chunk_ids:
            continue
        pending_records.append((idx, record))

    logger.info("Prepared %s pending records, skipped %s existing records", len(pending_records), len(existing_chunk_ids))

    def _flush_success(result: dict[str, Any]) -> None:
        existing_results.append(result)
        chunk_id = _norm(result.get("chunk_index"))
        if chunk_id:
            existing_chunk_ids.add(chunk_id)
        with success_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flush_failure(failure: dict[str, Any]) -> None:
        with fail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def _run_one(idx: int, record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(record, dict) or not _norm(record.get("text")):
            return None, None
        logger.info("Processing %s record %s", input_path.name, idx)
        try:
            data = process_record(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(record, idx),
                model=model,
            )
            data = _post_process_extraction(data, record)
            data["record_index"] = idx
            data["chunk_index"] = record.get("chunk_index", idx)
            data["source"] = record.get("source", "")
            data["text"] = record.get("text", "")
            return data, None
        except Exception as exc:
            logger.exception("Record failed: file=%s record=%s err=%s", input_path.name, idx, exc)
            return None, {
                "record_index": idx,
                "chunk_index": record.get("chunk_index", idx),
                "source": record.get("source", ""),
                "text": record.get("text", ""),
                "error": str(exc),
            }

    if max_workers <= 1:
        for idx, record in pending_records:
            data, failure = _run_one(idx, record)
            if failure is not None:
                _flush_failure(failure)
            if data is not None:
                _flush_success(data)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_one, idx, record): idx
                for idx, record in pending_records
            }
            for future in concurrent.futures.as_completed(future_map):
                data, failure = future.result()
                if failure is not None:
                    _flush_failure(failure)
                if data is not None:
                    _flush_success(data)

    output_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2), encoding="utf-8")


def write_aggregate_result(*, output_dir: Path, result_path: Path) -> None:
    aggregated: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("chunk_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skip invalid JSON output file: %s", path)
            continue
        if isinstance(payload, list):
            aggregated.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            aggregated.append(payload)
    result_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote aggregate result file: %s records=%s", result_path, len(aggregated))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM triple extraction for unstructured medication text chunks")
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "data_process" / "txt", help="Directory containing chunk JSON files")
    parser.add_argument("--input-file", type=Path, default=None, help="Single input JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "unstructured_output",
        help="Output directory",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=ROOT_DIR / "knowledge_graph_model.txt",
        help="KG modeling rules file",
    )
    parser.add_argument("--pattern", default="chunk_*.json", help="Glob pattern for input files")
    parser.add_argument("--model", default=DEEPSEEK_MODEL_NAME or MODEL_NAME, help="LLM model name")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N files/records when > 0")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count at file level")
    parser.add_argument("--force", action="store_true", help="Re-process even if output file exists")
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path(__file__).resolve().parent / "result.json",
        help="Aggregate result JSON path, aligned with other result.json outputs",
    )
    args = parser.parse_args()

    if args.input_file is not None and args.input_dir is not None:
        input_paths = [args.input_file]
    else:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        input_paths = sorted(args.input_dir.glob(args.pattern))

    if args.limit > 0:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        logger.warning("No input files found")
        return
    if not args.rules_file.is_file():
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")

    modeling_rules = load_modeling_rules(args.rules_file)
    system_prompt = build_system_prompt(modeling_rules)

    logger.info("Found %s files to process", len(input_paths))

    def _run_file(input_path: Path) -> tuple[Path, str]:
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.force:
            return input_path, "skipped"
        process_file(
            system_prompt=system_prompt,
            model=args.model,
            input_path=input_path,
            output_path=output_path,
            limit=0,
            max_workers=1,
        )
        return input_path, "written"

    if max(1, args.max_workers) <= 1:
        for input_path in input_paths:
            try:
                path, status = _run_file(input_path)
                logger.info("%s %s", status.capitalize(), path.name)
            except Exception as exc:
                logger.exception("File failed: %s err=%s", input_path, exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {
                executor.submit(_run_file, input_path): input_path
                for input_path in input_paths
            }
            for future in concurrent.futures.as_completed(future_map):
                input_path = future_map[future]
                try:
                    path, status = future.result()
                    logger.info("%s %s", status.capitalize(), path.name)
                except Exception as exc:
                    logger.exception("File failed: %s err=%s", input_path, exc)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_aggregate_result(output_dir=args.output_dir, result_path=args.result_path)


if __name__ == "__main__":
    main()
