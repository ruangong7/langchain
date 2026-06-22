from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUTS = [
    ("structured", ROOT_DIR / "drug_kg" / "structured" / "result.json"),
    ("unstructured", ROOT_DIR / "drug_kg" / "unstructured" / "result.json"),
    ("pdf", ROOT_DIR / "drug_kg" / "PDF" / "result.json"),
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _normalize_entity(entity: dict[str, Any]) -> dict[str, Any]:
    props = entity.get("properties")
    if not isinstance(props, dict):
        props = {}
    return {
        "id": _norm(entity.get("id")),
        "name": _norm(entity.get("name")),
        "canonical_name": _norm(entity.get("canonical_name") or entity.get("name")),
        "type": _norm(entity.get("type")),
        "properties": props,
    }


def _normalize_relation(
    relation: dict[str, Any],
    record: dict[str, Any],
    source_type: str,
) -> dict[str, Any]:
    raw_props = relation.get("properties")
    if not isinstance(raw_props, dict):
        raw_props = {}

    chunk_id = _norm(raw_props.get("chunk_id") or record.get("chunk_index"))
    data_text = _norm(raw_props.get("data") or raw_props.get("raw_text") or record.get("text"))
    source_name = _norm(raw_props.get("source") or record.get("source"))

    props = dict(raw_props)
    props["chunk_id"] = chunk_id
    props["data"] = data_text
    props["source"] = source_name
    props["source_type"] = source_type

    props.pop("chunk_index", None)
    props.pop("raw_text", None)

    return {
        "id": _norm(relation.get("id")),
        "source_id": _norm(relation.get("source_id")),
        "target_id": _norm(relation.get("target_id")),
        "type": _norm(relation.get("type")),
        "properties": props,
    }


def _normalize_record(record: dict[str, Any], source_type: str) -> dict[str, Any]:
    normalized = {
        "chunk_index": _norm(record.get("chunk_index")),
        "record_index": record.get("record_index"),
        "source": _norm(record.get("source")),
        "text": _norm(record.get("text")),
        "entities": [],
        "relations": [],
        "source_type": source_type,
        "metadata": {},
    }

    for key in ("record_id", "row_index", "sheet", "field", "main_entity"):
        if key in record:
            normalized["metadata"][key] = record.get(key)

    entities = record.get("entities") or []
    relations = record.get("relations") or []

    if isinstance(entities, list):
        normalized["entities"] = [
            _normalize_entity(entity)
            for entity in entities
            if isinstance(entity, dict)
        ]
    if isinstance(relations, list):
        normalized["relations"] = [
            _normalize_relation(relation, record, source_type=source_type)
            for relation in relations
            if isinstance(relation, dict)
        ]

    return normalized


def merge_results(input_specs: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "sources": {},
        "total_records": 0,
        "total_entities": 0,
        "total_relations": 0,
    }

    for source_type, path in input_specs:
        records = _load_records(path)
        normalized_records = [_normalize_record(record, source_type=source_type) for record in records]
        merged.extend(normalized_records)

        source_entities = sum(len(record["entities"]) for record in normalized_records)
        source_relations = sum(len(record["relations"]) for record in normalized_records)
        summary["sources"][source_type] = {
            "path": str(path),
            "records": len(normalized_records),
            "entities": source_entities,
            "relations": source_relations,
        }
        summary["total_records"] += len(normalized_records)
        summary["total_entities"] += source_entities
        summary["total_relations"] += source_relations

    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the 3 KG extraction result.json files into one aligned schema.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write merged_result.json and merge_summary.json",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged, summary = merge_results(DEFAULT_INPUTS)

    merged_path = args.output_dir / "merged_result.json"
    summary_path = args.output_dir / "merge_summary.json"

    merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
