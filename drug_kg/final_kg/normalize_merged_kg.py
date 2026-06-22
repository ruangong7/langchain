from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = FINAL_KG_DIR / "merged_result.json"
DEFAULT_DRUG_ALIAS_MAP = FINAL_KG_DIR / "drug_dict" / "drug_alias_map.json"

SPACE_RE = re.compile(r"\s+")
BRACKET_SPACE_RE = re.compile(r"[\(\)（）\[\]【】]")
PUNCT_RE = re.compile(r"[,:;：；、/\\\-]+")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value: str) -> str:
    value = _norm(value)
    value = SPACE_RE.sub("", value)
    value = BRACKET_SPACE_RE.sub("", value)
    value = PUNCT_RE.sub("", value)
    return value.lower()


def _canonical_entity_name(entity: dict[str, Any], drug_alias_map: dict[str, str] | None = None) -> str:
    entity_type = _norm(entity.get("type"))
    canonical_name = _norm(entity.get("canonical_name"))
    name = _norm(entity.get("name"))

    if entity_type == "Drug" and drug_alias_map:
        for candidate in (canonical_name, name):
            normalized_candidate = _normalize_text(candidate)
            if normalized_candidate and normalized_candidate in drug_alias_map:
                return drug_alias_map[normalized_candidate]
    if canonical_name:
        return canonical_name
    return name


def _entity_key(entity: dict[str, Any], drug_alias_map: dict[str, str] | None = None) -> tuple[str, str]:
    entity_type = _norm(entity.get("type"))
    canonical_name = _canonical_entity_name(entity, drug_alias_map=drug_alias_map)
    normalized_name = _normalize_text(canonical_name)
    return entity_type, normalized_name or _normalize_text(_norm(entity.get("name")))


def _relation_key(
    source_type: str,
    source_name: str,
    relation_type: str,
    target_type: str,
    target_name: str,
) -> tuple[str, str, str, str, str]:
    return (
        source_type,
        _normalize_text(source_name),
        relation_type,
        target_type,
        _normalize_text(target_name),
    )


def normalize_graph(
    records: list[dict[str, Any]],
    drug_alias_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entity_map: dict[tuple[str, str], dict[str, Any]] = {}
    relation_map: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    alias_map: dict[str, set[str]] = defaultdict(set)

    for record in records:
        entity_id_to_key: dict[str, tuple[str, str]] = {}

        for entity in record.get("entities") or []:
            if not isinstance(entity, dict):
                continue

            entity_type, normalized_name = _entity_key(entity, drug_alias_map=drug_alias_map)
            if not entity_type or not normalized_name:
                continue

            canonical_name = _canonical_entity_name(entity, drug_alias_map=drug_alias_map)
            key = (entity_type, normalized_name)
            entity_id_to_key[_norm(entity.get("id"))] = key

            item = entity_map.setdefault(
                key,
                {
                    "entity_id": f"{entity_type}::{normalized_name}",
                    "entity_type": entity_type,
                    "canonical_name": canonical_name,
                    "normalized_name": normalized_name,
                    "aliases": set(),
                    "properties": {},
                    "source_types": set(),
                    "chunk_ids": set(),
                    "source_records": 0,
                },
            )

            item["source_records"] += 1
            item["source_types"].add(_norm(record.get("source_type")))
            chunk_id = _norm(record.get("chunk_index"))
            if chunk_id:
                item["chunk_ids"].add(chunk_id)

            raw_name = _norm(entity.get("name"))
            if raw_name:
                item["aliases"].add(raw_name)
                alias_map[item["canonical_name"]].add(raw_name)
            raw_canonical = _norm(entity.get("canonical_name"))
            if raw_canonical:
                item["aliases"].add(raw_canonical)
                alias_map[item["canonical_name"]].add(raw_canonical)

            props = entity.get("properties") or {}
            if isinstance(props, dict):
                for prop_key, prop_value in props.items():
                    if prop_key not in item["properties"] or not item["properties"][prop_key]:
                        item["properties"][prop_key] = prop_value

        for relation in record.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            source_key = entity_id_to_key.get(_norm(relation.get("source_id")))
            target_key = entity_id_to_key.get(_norm(relation.get("target_id")))
            if not source_key or not target_key:
                continue

            source_entity = entity_map[source_key]
            target_entity = entity_map[target_key]
            relation_type = _norm(relation.get("type"))
            key = _relation_key(
                source_type=source_entity["entity_type"],
                source_name=source_entity["canonical_name"],
                relation_type=relation_type,
                target_type=target_entity["entity_type"],
                target_name=target_entity["canonical_name"],
            )

            item = relation_map.setdefault(
                key,
                {
                    "relation_id": "::".join(key),
                    "source_entity_id": source_entity["entity_id"],
                    "source_type": source_entity["entity_type"],
                    "source_name": source_entity["canonical_name"],
                    "relation_type": relation_type,
                    "target_entity_id": target_entity["entity_id"],
                    "target_type": target_entity["entity_type"],
                    "target_name": target_entity["canonical_name"],
                    "chunk_ids": set(),
                    "source_types": set(),
                    "evidence": [],
                    "evidence_count": 0,
                },
            )

            props = relation.get("properties") or {}
            chunk_id = _norm(props.get("chunk_id"))
            if chunk_id:
                item["chunk_ids"].add(chunk_id)
            source_type = _norm(props.get("source_type") or record.get("source_type"))
            if source_type:
                item["source_types"].add(source_type)
            data_text = _norm(props.get("data"))
            if data_text:
                item["evidence_count"] += 1
                if len(item["evidence"]) < 10 and data_text not in item["evidence"]:
                    item["evidence"].append(data_text)

    entities = []
    for item in entity_map.values():
        entities.append(
            {
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "canonical_name": item["canonical_name"],
                "normalized_name": item["normalized_name"],
                "aliases": sorted(item["aliases"]),
                "properties": item["properties"],
                "source_types": sorted(item["source_types"]),
                "chunk_ids": sorted(item["chunk_ids"]),
                "source_records": item["source_records"],
            }
        )

    relations = []
    for item in relation_map.values():
        relations.append(
            {
                "relation_id": item["relation_id"],
                "source_entity_id": item["source_entity_id"],
                "source_type": item["source_type"],
                "source_name": item["source_name"],
                "relation_type": item["relation_type"],
                "target_entity_id": item["target_entity_id"],
                "target_type": item["target_type"],
                "target_name": item["target_name"],
                "chunk_ids": sorted(item["chunk_ids"]),
                "source_types": sorted(item["source_types"]),
                "evidence": item["evidence"],
                "evidence_count": item["evidence_count"],
            }
        )

    entities.sort(key=lambda x: (x["entity_type"], x["canonical_name"]))
    relations.sort(key=lambda x: (x["relation_type"], x["source_name"], x["target_name"]))

    summary = {
        "input_records": len(records),
        "normalized_entities": len(entities),
        "normalized_relations": len(relations),
        "entity_types": defaultdict(int),
        "relation_types": defaultdict(int),
    }
    for entity in entities:
        summary["entity_types"][entity["entity_type"]] += 1
    for relation in relations:
        summary["relation_types"][relation["relation_type"]] += 1

    summary["entity_types"] = dict(sorted(summary["entity_types"].items()))
    summary["relation_types"] = dict(sorted(summary["relation_types"].items()))
    return entities, relations, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize merged KG results into aggregated entity/relation tables.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--drug-alias-map", type=Path, default=DEFAULT_DRUG_ALIAS_MAP)
    parser.add_argument("--output-dir", type=Path, default=FINAL_KG_DIR)
    args = parser.parse_args()

    records = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {args.input_path}")

    drug_alias_map: dict[str, str] | None = None
    if args.drug_alias_map.exists():
        raw_map = json.loads(args.drug_alias_map.read_text(encoding="utf-8"))
        if isinstance(raw_map, dict):
            drug_alias_map = {_normalize_text(k): _norm(v) for k, v in raw_map.items() if _norm(k) and _norm(v)}

    entities, relations, summary = normalize_graph(records, drug_alias_map=drug_alias_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "normalized_entities.json").write_text(
        json.dumps(entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "normalized_relations.json").write_text(
        json.dumps(relations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "normalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
