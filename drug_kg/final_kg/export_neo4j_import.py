from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_ENTITY_PATH = FINAL_KG_DIR / "normalized_entities.json"
DEFAULT_RELATION_PATH = FINAL_KG_DIR / "normalized_relations.json"
DEFAULT_OUTPUT_DIR = FINAL_KG_DIR / "neo4j_import"

LONG_TEXT_RE = re.compile(r"[。；;：:，,]{3,}")
LISTING_RE = re.compile(r"\b\d+\.")
BAD_DRUG_PATTERNS = [
    re.compile(r"其它成份|其他成份|辅料|注射用水"),
    re.compile(r"用法用量|适应症|注意事项|药理作用"),
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_bad_entity(entity: dict[str, Any]) -> tuple[bool, str]:
    name = _norm(entity.get("canonical_name"))
    entity_type = _norm(entity.get("entity_type"))

    if not name:
        return True, "empty_name"
    if len(name) > 120:
        return True, "name_too_long"
    if entity_type == "Drug":
        if any(pattern.search(name) for pattern in BAD_DRUG_PATTERNS):
            return True, "drug_description_like"
        if LONG_TEXT_RE.search(name) and LISTING_RE.search(name):
            return True, "drug_listing_like"
    return False, ""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter normalized KG and export Neo4j import CSV files.")
    parser.add_argument("--entity-path", type=Path, default=DEFAULT_ENTITY_PATH)
    parser.add_argument("--relation-path", type=Path, default=DEFAULT_RELATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    entities = json.loads(args.entity_path.read_text(encoding="utf-8"))
    relations = json.loads(args.relation_path.read_text(encoding="utf-8"))

    kept_entities: list[dict[str, Any]] = []
    dropped_entities: list[dict[str, Any]] = []
    kept_entity_ids: set[str] = set()

    for entity in entities:
        is_bad, reason = _is_bad_entity(entity)
        if is_bad:
            dropped = dict(entity)
            dropped["drop_reason"] = reason
            dropped_entities.append(dropped)
            continue
        kept_entities.append(entity)
        kept_entity_ids.add(_norm(entity.get("entity_id")))

    kept_relations: list[dict[str, Any]] = []
    dropped_relations: list[dict[str, Any]] = []
    for relation in relations:
        source_id = _norm(relation.get("source_entity_id"))
        target_id = _norm(relation.get("target_entity_id"))
        if source_id not in kept_entity_ids or target_id not in kept_entity_ids:
            dropped = dict(relation)
            dropped["drop_reason"] = "missing_endpoint_after_entity_filter"
            dropped_relations.append(dropped)
            continue
        kept_relations.append(relation)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "entities.filtered.json").write_text(
        json.dumps(kept_entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "relations.filtered.json").write_text(
        json.dumps(kept_relations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "entities.dropped.json").write_text(
        json.dumps(dropped_entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "relations.dropped.json").write_text(
        json.dumps(dropped_relations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    node_rows = []
    for entity in kept_entities:
        node_rows.append(
            {
                ":ID": entity["entity_id"],
                ":LABEL": entity["entity_type"],
                "entity_type": entity["entity_type"],
                "canonical_name": entity["canonical_name"],
                "normalized_name": entity["normalized_name"],
                "aliases": "|".join(entity.get("aliases") or []),
                "source_types": "|".join(entity.get("source_types") or []),
                "source_records:int": entity.get("source_records", 0),
                "properties_json": json.dumps(entity.get("properties") or {}, ensure_ascii=False),
            }
        )

    rel_rows = []
    for relation in kept_relations:
        rel_rows.append(
            {
                ":START_ID": relation["source_entity_id"],
                ":END_ID": relation["target_entity_id"],
                ":TYPE": relation["relation_type"],
                "relation_id": relation["relation_id"],
                "source_type": relation["source_type"],
                "source_name": relation["source_name"],
                "target_type": relation["target_type"],
                "target_name": relation["target_name"],
                "chunk_ids": "|".join(relation.get("chunk_ids") or []),
                "source_types": "|".join(relation.get("source_types") or []),
                "evidence_count:int": relation.get("evidence_count", 0),
                "evidence_json": json.dumps(relation.get("evidence") or [], ensure_ascii=False),
            }
        )

    _write_csv(
        args.output_dir / "nodes.csv",
        [
            ":ID",
            ":LABEL",
            "entity_type",
            "canonical_name",
            "normalized_name",
            "aliases",
            "source_types",
            "source_records:int",
            "properties_json",
        ],
        node_rows,
    )
    _write_csv(
        args.output_dir / "relations.csv",
        [
            ":START_ID",
            ":END_ID",
            ":TYPE",
            "relation_id",
            "source_type",
            "source_name",
            "target_type",
            "target_name",
            "chunk_ids",
            "source_types",
            "evidence_count:int",
            "evidence_json",
        ],
        rel_rows,
    )

    summary = {
        "kept_entities": len(kept_entities),
        "dropped_entities": len(dropped_entities),
        "kept_relations": len(kept_relations),
        "dropped_relations": len(dropped_relations),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
