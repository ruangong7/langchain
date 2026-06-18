"""Use an LLM to extract entities and relations from structured drug JSON records."""
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
import orjson


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2a70ab5f703d4c929ec8860ffab46b9a")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


OUTPUT_SCHEMA = {
    "entities": [
        {
            "id": "ent_1",
            "type": "Drug",
            "name": "",
            "canonical_name": "",
            "properties": {},
        }
    ],
    "relations": [
        {
            "id": "rel_1",
            "type": "INTERACTS_WITH",
            "source_id": "ent_1",
            "target_id": "ent_2",
            "properties": {},
        }
    ],
}

ALLOWED_RELATIONS_BY_FIELD = {
    "interaction": {"INTERACTS_WITH", "AFFECTS_INDICATOR"},
    "precaution": {"CONTRAINDICATED_FOR", "AFFECTS_INDICATOR", "HAS_ADVERSE_REACTION", "INTERACTS_WITH"},
    "indication": {"INDICATED_FOR", "APPLIES_TO", "HAS_SYMPTOM"},
}
PREFERRED_ENTITY_TYPES_BY_FIELD = {
    "interaction": {"Drug", "Food", "Indicator", "DrugClass"},
    "precaution": {"Population", "Disease", "Indicator", "Food", "SideEffect", "Symptom", "Drug"},
    "indication": {"Disease", "Population", "Symptom", "Drug"},
}
MAX_ENTITIES_BY_FIELD = {
    "interaction": 12,
    "precaution": 12,
    "indication": 10,
}
MAX_RELATIONS_BY_FIELD = {
    "interaction": 10,
    "precaution": 10,
    "indication": 6,
}
LONG_ENTITY_PUNCT = ("，", "。", "；", ";", "：", ":", "、")
GENERIC_ENTITY_NAMES = {
    "慎用",
    "禁用",
    "不宜",
    "避免",
    "尚不明确",
    "详见说明书",
}

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


def _relation_properties_from_record(*, record: dict[str, Any], inferred_from_field: bool = False) -> dict[str, Any]:
    props: dict[str, Any] = {
        "chunk_id": _norm(record.get("chunk_index", "")),
    }
    if inferred_from_field:
        props["inferred_from_field"] = True
    return props


def load_modeling_rules(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_system_prompt(modeling_rules: str) -> str:
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""你是一个严格按照药物知识图谱模式进行抽取的结构化信息抽取器。
你的任务不是做自由摘要，而是把输入记录中的 `text` 字段抽取成最终可落库的知识图谱实体和关系。

【知识图谱建模规则】
{modeling_rules}

【任务范围】
1. 当前输入只针对三个字段抽取：`interaction`、`precaution`、`indication`。
2. 每条记录都带有一个主实体 `main_entity`，它就是当前药物，默认是本条记录最核心的主体。
3. `main_entity.properties` 里的 `approval_number`、`instruction`、`ingredient`、`picture` 只是辅助上下文，只能帮助你理解文本，不能作为本轮主要抽取对象。
4. 本轮目标是围绕三个字段文本构建三元组，其他列不是独立节点来源。

【抽取目标】
你需要基于当前字段完成：
1. 实体识别
2. 关系抽取
3. 三元组落地
但最终只输出可落库的 `entities` 和 `relations`。

【允许的实体类型】
只允许使用以下类型：
`Drug`, `DrugClass`, `Food`, `Disease`, `SideEffect`, `Symptom`, `Indicator`, `Population`

【允许的关系类型】
只允许使用以下类型：
`IN_CLASS`, `INDICATED_FOR`, `HAS_ADVERSE_REACTION`, `AFFECTS_INDICATOR`, `HAS_SYMPTOM`, `CONTRAINDICATED_FOR`, `APPLIES_TO`, `INTERACTS_WITH`

【字段约束】
1. `interaction` 重点抽取：
`INTERACTS_WITH`, `AFFECTS_INDICATOR`
2. `precaution` 重点抽取：
`CONTRAINDICATED_FOR`, `AFFECTS_INDICATOR`, `HAS_ADVERSE_REACTION`, `INTERACTS_WITH`
3. `indication` 重点抽取：
`INDICATED_FOR`, `APPLIES_TO`, `HAS_SYMPTOM`

【主药约束】
1. `main_entity` 是当前记录的主药，不要重复创造一个语义完全相同的新主药节点。
2. 如果文本中出现主药简称、别名、剂型变体，优先将其视为 `main_entity` 的同一语义，除非文本明确在讨论另一种独立药物。
3. 大多数关系应从 `main_entity` 出发，不要输出大量与主药无关的边。

【严格抽取规则】
1. 只抽取文本中明确表达、或可直接严格推出的信息，不要脑补。
2. 不要把整句、整段说明文字当成实体。
3. 不创建事件节点，不创建“注意事项说明”“相互作用说明”之类抽象节点。
4. 如果主药在文本中被省略，默认文本讨论对象就是 `main_entity`。
5. 如果无法确定关系，可以保留实体，但不要强造关系。
6. 每条关系都应能够直接落成三元组。
7. `source_id` 和 `target_id` 必须引用当前输出中的已有实体 id。
8. 如果某个信息只来自 `approval_number`、`instruction`、`ingredient`、`picture` 这些辅助属性，而不是当前字段 `text`，不要单独抽取它。
9. 不要输出 schema 之外的实体类型，不要输出 schema 之外的关系类型。

【输出要求】
1. 最终只输出一个合法 JSON 对象。
2. 顶层必须包含 `entities` 和 `relations`。
3. `entities` 中只保留参与三元组或强相关的核心实体，避免冗余。
4. `relations` 中每一条都应服务于最终图谱落库。
5. 不要输出解释，不要输出 Markdown，不要输出额外文字。

输出结构如下：
{schema_text}
"""


def build_user_prompt(record: dict[str, Any], record_index: int) -> str:
    payload = {
        "record_index": record_index,
        "record_id": record.get("record_id"),
        "chunk_index": record.get("chunk_index"),
        "source": record.get("source", ""),
        "sheet": record.get("sheet", ""),
        "row_index": record.get("row_index", ""),
        "field": record.get("field", ""),
        "main_entity": record.get("main_entity", {}),
        "text": record.get("text", ""),
    }
    return (
        "请基于下面这条结构化药品字段记录进行知识图谱抽取。\n"
        "要求如下：\n"
        "1. 只围绕 `main_entity` 这个主药和当前 `field` 进行抽取。\n"
        "2. 只对 `text` 内容做实体识别、关系抽取和三元组落地。\n"
        "3. `approval_number`、`instruction`、`ingredient`、`picture` 只是辅助理解，不要把它们当作本轮主要抽取结果。\n"
        "4. 如果文本中出现主药简称、别名、剂型表达，优先合并到 `main_entity` 语义下。\n"
        "5. 只保留和最终三元组有关的核心实体与关系。\n"
        "6. 不要输出解释，只输出 JSON。\n"
        "7. `interaction` 优先抽相互作用对象、监测指标，不要创造机制类实体或风险类关系。\n"
        "8. `precaution` 优先抽禁忌对象、监测指标、不良反应相关约束，不要创造 schema 外关系。\n"
        "9. `indication` 优先抽适应症疾病、适用人群、相关症状。\n\n"
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
        s = re.sub(r",\s*([}\]])", r"\1", s)
        s = re.sub(r"}\s*\n(\s*)\"", r"},\n\1\"", s)
        s = re.sub(r"}\s*\n(\s*)\{", r"},\n\1{", s)
        s = re.sub(r"]\s*\n(\s*)([\[{])", r"],\n\1\2", s)
        s = re.sub(r'"\s*\n\s*"', r'",\n"', s)
        return s.lstrip("\ufeff").strip()

    def _trim_to_balanced_json(s: str) -> str | None:
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        last_balanced_end = -1
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
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_balanced_end = i + 1
        if last_balanced_end > start:
            return s[start:last_balanced_end]
        return None

    def _parse_candidate(s: str) -> dict[str, Any]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return orjson.loads(s)

    text = _strip_code_fence(text)
    if not text.strip():
        raise json.JSONDecodeError("Empty response content", text, 0)

    candidate = _extract_first_json_object(text) or text.strip()
    candidate = _cleanup_common_json_issues(candidate)
    try:
        return _parse_candidate(candidate)
    except Exception:
        match = re.search(r"(\{[\s\S]*\})", text, re.S)
        if not match:
            raise
        blob = _cleanup_common_json_issues(match.group(1))
        try:
            return _parse_candidate(blob)
        except Exception:
            trimmed = _trim_to_balanced_json(blob)
            if trimmed:
                return _parse_candidate(_cleanup_common_json_issues(trimmed))
            raise


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _entity_name(entity: dict[str, Any]) -> str:
    return _norm(entity.get("canonical_name") or entity.get("canonical") or entity.get("name"))


def _canonicalize_drug_name(name: str) -> str:
    value = _norm(name)
    if not value:
        return value
    suffixes = (
        "分散片",
        "缓释片",
        "控释片",
        "肠溶片",
        "肠溶胶囊",
        "泡腾片",
        "咀嚼片",
        "注射液",
        "注射剂",
        "口服液",
        "混悬液",
        "乳膏",
        "胶囊",
        "颗粒",
        "胶丸",
        "糖浆",
        "滴丸",
        "凝胶",
        "片",
        "散",
        "栓",
        "丸",
        "膏",
    )
    changed = True
    while changed and value:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                value = value[: -len(suffix)].strip()
                changed = True
                break
    return value


def _is_main_drug_alias(entity_name: str, main_name: str) -> bool:
    if not entity_name or not main_name:
        return False
    if entity_name == main_name:
        return True
    entity_base = _canonicalize_drug_name(entity_name)
    main_base = _canonicalize_drug_name(main_name)
    if not entity_base or not main_base:
        return False
    return entity_base == main_base or entity_base in main_name or main_base in entity_name


def _looks_like_bad_entity(name: str) -> bool:
    if not name:
        return True
    if name in GENERIC_ENTITY_NAMES:
        return True
    if len(name) > 20:
        return True
    if sum(1 for ch in LONG_ENTITY_PUNCT if ch in name) >= 2:
        return True
    if any(token in name for token in ("说明书", "同上", "本品", "患者在同时", "可引起", "可能导致")):
        return True
    return False


def _split_evidence_sentences(text: str) -> list[str]:
    cleaned = _norm(text)
    if not cleaned:
        return []
    cleaned = re.sub(r"\s*(?=(?:\d+[\.、]))", "\n", cleaned)
    parts = re.split(r"(?<=[。；;！？!?])|\n+", cleaned)
    sentences = [_norm(part) for part in parts if _norm(part)]
    if sentences:
        return sentences
    return [cleaned]


def _relation_keywords(rel_type: str) -> tuple[str, ...]:
    mapping = {
        "INTERACTS_WITH": ("相互作用", "合用", "同用", "同时使用", "联用", "并用", "避免", "禁止"),
        "AFFECTS_INDICATOR": ("监测", "指标", "血药浓度", "清除率", "肝功能", "肾功能", "肌酐", "血压", "血糖"),
        "CONTRAINDICATED_FOR": ("禁用", "禁忌", "禁止", "不宜", "避免"),
        "HAS_ADVERSE_REACTION": ("不良反应", "反应", "头晕", "头痛", "恶心", "呕吐", "腹泻", "嗜睡"),
        "HAS_SYMPTOM": ("症状", "表现", "伴有"),
        "INDICATED_FOR": ("适应症", "用于", "治疗", "可用于"),
        "APPLIES_TO": ("适用于", "用于", "人群", "患者"),
    }
    return mapping.get(rel_type, ())


def _score_relation_sentence(*, sentence: str, source_name: str, target_name: str, rel_type: str) -> int:
    score = 0
    if source_name and source_name in sentence:
        score += 4
    if target_name and target_name in sentence:
        score += 6
    for keyword in _relation_keywords(rel_type):
        if keyword and keyword in sentence:
            score += 2
    if source_name and target_name and source_name in sentence and target_name in sentence:
        score += 4
    return score


def _pick_relation_evidence(
    *,
    relation: dict[str, Any],
    record: dict[str, Any],
    source_name: str,
    target_name: str,
    rel_type: str,
) -> str:
    relation_props = relation.get("properties") if isinstance(relation.get("properties"), dict) else {}
    candidate_texts: list[str] = []
    if isinstance(relation_props, dict):
        candidate_texts.extend(
            [
                _norm(relation_props.get("data")),
                _norm(relation_props.get("raw_data")),
                _norm(relation_props.get("raw_text")),
            ]
        )
        evidence = relation_props.get("evidence")
        if isinstance(evidence, dict):
            candidate_texts.append(_norm(evidence.get("text")))

    record_text = _norm(record.get("text"))
    for candidate in candidate_texts:
        if not candidate:
            continue
        if candidate != record_text and len(candidate) <= max(160, len(record_text) // 2):
            return candidate

    scored_sentences: list[tuple[int, str]] = []
    for sentence in _split_evidence_sentences(record_text):
        score = _score_relation_sentence(
            sentence=sentence,
            source_name=source_name,
            target_name=target_name,
            rel_type=rel_type,
        )
        if score > 0:
            scored_sentences.append((score, sentence))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: (-item[0], len(item[1])))
        return scored_sentences[0][1]

    if target_name:
        for sentence in _split_evidence_sentences(record_text):
            if target_name in sentence:
                return sentence

    return record_text


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
            logger.warning(
                "Structured parse failed on attempt %s/%s: %s",
                attempt + 1,
                max_parse_retries + 1,
                exc,
            )
            if attempt >= max_parse_retries:
                break
    snippet = _norm(last_content)[:800]
    raise RuntimeError(
        f"Invalid structured output from model after {max_parse_retries + 1} attempts: {last_exc}; "
        f"response_snippet={snippet}"
    ) from last_exc


def _post_process_extraction(data: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    field = _norm(record.get("field"))
    allowed_relations = ALLOWED_RELATIONS_BY_FIELD.get(field, set())
    preferred_types = PREFERRED_ENTITY_TYPES_BY_FIELD.get(field, set())
    main_entity = record.get("main_entity") or {}
    main_name = _norm(main_entity.get("name"))

    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relations = data.get("relations") if isinstance(data.get("relations"), list) else []

    dedup_entities: dict[tuple[str, str], dict[str, Any]] = {}
    entity_id_remap: dict[str, str] = {}
    main_drug_entity_id = ""

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        old_id = _norm(entity.get("id"))
        entity_type = _norm(entity.get("type") or "Entity")
        entity_name = _entity_name(entity)
        if not old_id or not entity_name:
            continue
        if _looks_like_bad_entity(entity_name) and not (entity_type == "Drug" and _is_main_drug_alias(entity_name, main_name)):
            continue

        key = (entity_type, entity_name)
        if entity_type == "Drug" and _is_main_drug_alias(entity_name, main_name):
            if not main_drug_entity_id:
                main_drug_entity_id = "ent_main"
            entity_id_remap[old_id] = main_drug_entity_id
            dedup_entities[("Drug", main_name)] = {
                "id": main_drug_entity_id,
                "type": "Drug",
                "name": main_name,
                "canonical_name": main_name,
                "properties": main_entity.get("properties") or {},
            }
            continue

        if key not in dedup_entities:
            dedup_entities[key] = {
                "id": old_id,
                "type": entity_type,
                "name": _norm(entity.get("name") or entity_name),
                "canonical_name": entity_name,
                "properties": entity.get("properties") or {},
            }
        entity_id_remap[old_id] = dedup_entities[key]["id"]

    if main_name and not main_drug_entity_id:
        main_drug_entity_id = "ent_main"
        dedup_entities[("Drug", main_name)] = {
            "id": main_drug_entity_id,
            "type": "Drug",
            "name": main_name,
            "canonical_name": main_name,
            "properties": main_entity.get("properties") or {},
        }

    prioritized: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for entity in dedup_entities.values():
        if entity["type"] == "Drug" and _is_main_drug_alias(entity["canonical_name"], main_name):
            prioritized.append(entity)
        elif entity["type"] in preferred_types:
            prioritized.append(entity)
        else:
            deferred.append(entity)

    cleaned_entities: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    max_entities = MAX_ENTITIES_BY_FIELD.get(field, 9999)
    for entity in prioritized + deferred:
        key = (entity["type"], entity["canonical_name"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned_entities.append(entity)
        if len(cleaned_entities) >= max_entities:
            break

    valid_entity_ids = {_norm(entity["id"]) for entity in cleaned_entities}
    entity_by_id = {_norm(entity["id"]): entity for entity in cleaned_entities}
    relation_priority = {
        "CONTRAINDICATED_FOR": 0,
        "INDICATED_FOR": 0,
        "INTERACTS_WITH": 1,
        "AFFECTS_INDICATOR": 1,
        "HAS_ADVERSE_REACTION": 2,
        "HAS_SYMPTOM": 2,
        "APPLIES_TO": 3,
    }

    cleaned_relations: list[dict[str, Any]] = []
    existing_relation_keys: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        rel_type = _norm(relation.get("type"))
        source_id = entity_id_remap.get(_norm(relation.get("source_id")), _norm(relation.get("source_id")))
        target_id = entity_id_remap.get(_norm(relation.get("target_id")), _norm(relation.get("target_id")))
        if allowed_relations and rel_type not in allowed_relations:
            continue
        if not source_id or not target_id:
            continue
        if source_id not in valid_entity_ids or target_id not in valid_entity_ids:
            continue
        rel_key = (source_id, rel_type, target_id)
        if rel_key in existing_relation_keys:
            continue
        props = _relation_properties_from_record(record=record)
        source_entity = entity_by_id.get(source_id, {})
        target_entity = entity_by_id.get(target_id, {})
        props["data"] = _pick_relation_evidence(
            relation=relation,
            record=record,
            source_name=_entity_name(source_entity) if isinstance(source_entity, dict) else "",
            target_name=_entity_name(target_entity) if isinstance(target_entity, dict) else "",
            rel_type=rel_type,
        )
        cleaned_relations.append(
            {
                "id": _norm(relation.get("id") or f"rel_{len(cleaned_relations) + 1}"),
                "type": rel_type,
                "source_id": source_id,
                "target_id": target_id,
                "properties": props,
            }
        )
        existing_relation_keys.add(rel_key)

    cleaned_relations.sort(key=lambda item: relation_priority.get(_norm(item.get("type")), 9))
    cleaned_relations = cleaned_relations[: MAX_RELATIONS_BY_FIELD.get(field, 9999)]

    if field == "indication":
        allowed_entity_types = {"Drug", "Disease", "Population", "Symptom"}
        cleaned_entities = [entity for entity in cleaned_entities if entity.get("type") in allowed_entity_types]
        valid_entity_ids = {_norm(entity["id"]) for entity in cleaned_entities}
        cleaned_relations = [
            relation
            for relation in cleaned_relations
            if _norm(relation.get("source_id")) in valid_entity_ids
            and _norm(relation.get("target_id")) in valid_entity_ids
        ]
        existing_relation_keys = {
            (_norm(relation.get("source_id")), _norm(relation.get("type")), _norm(relation.get("target_id")))
            for relation in cleaned_relations
        }
        if main_drug_entity_id:
            for entity in cleaned_entities:
                if entity["type"] != "Disease" or entity["id"] == main_drug_entity_id:
                    continue
                rel_key = (main_drug_entity_id, "INDICATED_FOR", entity["id"])
                if rel_key in existing_relation_keys:
                    continue
                cleaned_relations.append(
                    {
                        "id": f"rel_auto_{len(cleaned_relations) + 1}",
                        "type": "INDICATED_FOR",
                        "source_id": main_drug_entity_id,
                        "target_id": entity["id"],
                        "properties": {
                            **_relation_properties_from_record(
                                record=record,
                                inferred_from_field=True,
                            ),
                            "data": _pick_relation_evidence(
                                relation={},
                                record=record,
                                source_name=main_name,
                                target_name=_entity_name(entity),
                                rel_type="INDICATED_FOR",
                            ),
                        },
                    }
                )
                existing_relation_keys.add(rel_key)

    if field == "precaution":
        caution_markers = ("慎用", "监测", "密切监测", "调整剂量", "减量", "观察")
        hard_contra_markers = ("禁用", "禁忌", "禁止", "不宜", "忌")
        text = _norm(record.get("text"))
        has_hard_contra = any(marker in text for marker in hard_contra_markers)
        has_soft_caution = any(marker in text for marker in caution_markers)
        if not has_hard_contra or has_soft_caution:
            cleaned_relations = [
                relation
                for relation in cleaned_relations
                if relation.get("type") != "CONTRAINDICATED_FOR"
            ]

    referenced_ids: set[str] = set()
    for relation in cleaned_relations:
        referenced_ids.add(_norm(relation.get("source_id")))
        referenced_ids.add(_norm(relation.get("target_id")))

    final_entities: list[dict[str, Any]] = []
    for entity in cleaned_entities:
        entity_id = _norm(entity.get("id"))
        if entity_id == main_drug_entity_id or entity_id in referenced_ids:
            final_entities.append(entity)

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
    existing_record_ids: set[str] = set()

    if success_jsonl_path.exists():
        with success_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skip invalid success jsonl line in %s", success_jsonl_path)
                    continue
                if isinstance(item, dict):
                    existing_results.append(item)
    elif output_path.exists():
        try:
            existing_raw = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing_raw, list):
                existing_results = [item for item in existing_raw if isinstance(item, dict)]
        except json.JSONDecodeError:
            logger.warning("Existing output is not valid JSON, starting fresh append set: %s", output_path)

    existing_record_ids = {
        _norm(item.get("record_id"))
        for item in existing_results
        if _norm(item.get("record_id"))
    }

    pending_records: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_id = _norm(record.get("record_id"))
        if record_id and record_id in existing_record_ids:
            continue
        pending_records.append((idx, record))

    logger.info(
        "Prepared %s pending records, skipped %s existing records",
        len(pending_records),
        len(existing_record_ids),
    )

    results_by_idx: dict[int, dict[str, Any]] = {}

    def _run_one(idx: int, record: dict[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(record, dict) or not record.get("text"):
            return idx, None, None
        logger.info("Processing %s record %s", input_path.name, idx)
        try:
            data = process_record(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(record, idx),
                model=model,
            )
        except Exception as e:
            logger.exception("Record failed: file=%s record=%s err=%s", input_path.name, idx, e)
            return idx, None, {
                "record_id": record.get("record_id", ""),
                "record_index": record.get("record_index", idx),
                "chunk_index": record.get("chunk_index", ""),
                "source": record.get("source", ""),
                "sheet": record.get("sheet", ""),
                "row_index": record.get("row_index", ""),
                "field": record.get("field", ""),
                "text": record.get("text", ""),
                "main_entity": record.get("main_entity", {}),
                "error": str(e),
            }

        data = _post_process_extraction(data, record)
        data["record_id"] = record.get("record_id", "")
        data["record_index"] = record.get("record_index", idx)
        data["chunk_index"] = record.get("chunk_index", "")
        data["source"] = record.get("source", "")
        data["sheet"] = record.get("sheet", "")
        data["row_index"] = record.get("row_index", "")
        data["field"] = record.get("field", "")
        data["text"] = record.get("text", "")
        data["main_entity"] = record.get("main_entity", {})
        return idx, data, None

    def _flush_success(result: dict[str, Any]) -> None:
        existing_results.append(result)
        record_id = _norm(result.get("record_id"))
        if record_id:
            existing_record_ids.add(record_id)
        with success_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flush_failure(failure: dict[str, Any]) -> None:
        with fail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    if max_workers <= 1:
        for idx, record in pending_records:
            out_idx, data, failure = _run_one(idx, record)
            if failure is not None:
                _flush_failure(failure)
            if data is not None:
                results_by_idx[out_idx] = data
                _flush_success(data)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_one, idx, record): idx
                for idx, record in pending_records
            }
            for future in concurrent.futures.as_completed(future_map):
                out_idx, data, failure = future.result()
                if failure is not None:
                    _flush_failure(failure)
                if data is not None:
                    results_by_idx[out_idx] = data
                    _flush_success(data)

    output_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM entity/relation extraction for JSON records")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory containing JSON files")
    parser.add_argument("--input-file", type=Path, default=None, help="Single input JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory for extracted JSON files",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=ROOT_DIR / "knowledge_graph_model.txt",
        help="Modeling rules file used to build the prompt",
    )
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for input files")
    parser.add_argument("--model", default=DEEPSEEK_MODEL_NAME or MODEL_NAME, help="LLM model name")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N records/files when > 0")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count for record-level extraction")
    parser.add_argument("--force", action="store_true", help="Re-process even if output file already exists")
    parser.add_argument("--skip-existing", action="store_true", help="Deprecated compatibility flag")
    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        args.input_dir = ROOT_DIR / "data_process" / "txt"
    if args.input_dir is not None and args.input_file is not None:
        raise ValueError("Use either --input-dir or --input-file, not both")
    if args.input_dir is not None and not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.input_file is not None and not args.input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not args.rules_file.is_file():
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")

    modeling_rules = load_modeling_rules(args.rules_file)
    system_prompt = build_system_prompt(modeling_rules)

    if args.input_file is not None:
        input_paths = [args.input_file]
    else:
        input_paths = sorted(args.input_dir.glob(args.pattern))
        if args.limit > 0:
            input_paths = input_paths[: args.limit]

    if not input_paths:
        logger.warning("No input files found")
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
                limit=args.limit if args.input_file is not None else 0,
                max_workers=max(1, args.max_workers),
            )
            logger.info("Wrote %s", output_path)
        except Exception as e:
            logger.exception("File failed: %s err=%s", input_path, e)


if __name__ == "__main__":
    main()
