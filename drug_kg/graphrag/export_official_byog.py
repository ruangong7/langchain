from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from graphrag.data_model.schemas import (
    DOCUMENTS_FINAL_COLUMNS,
    ENTITIES_FINAL_COLUMNS,
    RELATIONSHIPS_FINAL_COLUMNS,
    TEXT_UNITS_FINAL_COLUMNS,
)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _split_pipe(value: Any) -> list[str]:
    text = _norm(value)
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _safe_json_loads(value: Any) -> Any:
    text = _norm(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _approx_tokens(text: str) -> int:
    text = _norm(text)
    if not text:
        return 0
    # A light approximation is sufficient for bootstrapping the official tables.
    return max(1, len(text) // 4)


def _entity_description(row: dict[str, Any]) -> str:
    aliases = _split_pipe(row.get("aliases"))
    source_types = _split_pipe(row.get("source_types"))
    parts = [
        f"实体类型：{_norm(row.get('entity_type'))}",
        f"规范名称：{_norm(row.get('canonical_name'))}",
    ]
    if aliases:
        parts.append(f"别名：{', '.join(aliases[:12])}")
    if source_types:
        parts.append(f"来源类型：{', '.join(source_types)}")
    source_records = _norm(row.get("source_records:int"))
    if source_records:
        parts.append(f"来源记录数：{source_records}")
    return "；".join([part for part in parts if part])


def _relation_description(
    *,
    relation_type: str,
    source_title: str,
    target_title: str,
    evidence_list: list[str],
    source_types: list[str],
) -> str:
    parts = [f"关系类型：{relation_type}", f"源实体：{source_title}", f"目标实体：{target_title}"]
    if source_types:
        parts.append(f"来源类型：{', '.join(source_types)}")
    if evidence_list:
        parts.append(f"证据：{' | '.join(evidence_list[:3])}")
    return "；".join([part for part in parts if part])


def build_tables(
    *,
    nodes: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    node_title_by_id = {_norm(row.get(":ID")): _norm(row.get("canonical_name")) for row in nodes}

    entity_text_unit_ids: dict[str, set[str]] = defaultdict(set)
    entity_degree: dict[str, int] = defaultdict(int)
    relationship_rows: list[dict[str, Any]] = []

    text_unit_entity_ids: dict[str, set[str]] = defaultdict(set)
    text_unit_relationship_ids: dict[str, set[str]] = defaultdict(set)
    text_unit_snippets: dict[str, list[str]] = defaultdict(list)
    text_unit_document_id: dict[str, str] = {}

    for idx, row in enumerate(relations, start=1):
        source_id = _norm(row.get(":START_ID"))
        target_id = _norm(row.get(":END_ID"))
        relation_id = _norm(row.get("relation_id")) or f"relationship::{idx}"
        relation_type = _norm(row.get(":TYPE"))
        source_title = node_title_by_id.get(source_id) or _norm(row.get("source_name")) or source_id
        target_title = node_title_by_id.get(target_id) or _norm(row.get("target_name")) or target_id
        chunk_ids = _split_pipe(row.get("chunk_ids"))
        source_types = _split_pipe(row.get("source_types"))
        evidence_json = _safe_json_loads(row.get("evidence_json"))
        evidence_list = []
        if isinstance(evidence_json, list):
            evidence_list = [_norm(item) for item in evidence_json if _norm(item)]
        elif evidence_json is not None:
            evidence_list = [_norm(evidence_json)]

        entity_degree[source_id] += 1
        entity_degree[target_id] += 1
        entity_text_unit_ids[source_id].update(chunk_ids)
        entity_text_unit_ids[target_id].update(chunk_ids)

        relation_description = _relation_description(
            relation_type=relation_type,
            source_title=source_title,
            target_title=target_title,
            evidence_list=evidence_list,
            source_types=source_types,
        )

        relationship_rows.append(
            {
                "id": relation_id,
                "human_readable_id": idx,
                "source": source_title,
                "target": target_title,
                "description": relation_description,
                "weight": float(_norm(row.get("evidence_count:int")) or len(chunk_ids) or 1),
                "combined_degree": 0,
                "text_unit_ids": chunk_ids,
            }
        )

        text_unit_ids = chunk_ids or [f"virtual::{relation_id}"]
        for text_unit_id in text_unit_ids:
            text_unit_entity_ids[text_unit_id].update([source_id, target_id])
            text_unit_relationship_ids[text_unit_id].add(relation_id)
            text_unit_document_id[text_unit_id] = f"document::{text_unit_id}"
            snippets = evidence_list[:3] if evidence_list else [relation_description]
            for snippet in snippets:
                if snippet and snippet not in text_unit_snippets[text_unit_id]:
                    text_unit_snippets[text_unit_id].append(snippet)

    relationships_df = pd.DataFrame(relationship_rows)
    if not relationships_df.empty:
        title_to_degree = {
            node_title_by_id.get(entity_id, entity_id): degree
            for entity_id, degree in entity_degree.items()
        }
        relationships_df["combined_degree"] = relationships_df["source"].map(title_to_degree).fillna(0).astype(int)
        relationships_df["combined_degree"] += relationships_df["target"].map(title_to_degree).fillna(0).astype(int)
        relationships_df = relationships_df.loc[:, RELATIONSHIPS_FINAL_COLUMNS]
    else:
        relationships_df = pd.DataFrame(columns=RELATIONSHIPS_FINAL_COLUMNS)

    entity_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(nodes, start=1):
        entity_id = _norm(row.get(":ID"))
        entity_rows.append(
            {
                "id": entity_id,
                "human_readable_id": idx,
                "title": _norm(row.get("canonical_name")) or entity_id,
                "type": _norm(row.get("entity_type")) or _norm(row.get(":LABEL")),
                "description": _entity_description(row),
                "text_unit_ids": sorted(entity_text_unit_ids.get(entity_id, set())),
                "frequency": int(_norm(row.get("source_records:int")) or 0),
                "degree": int(entity_degree.get(entity_id, 0)),
            }
        )
    entities_df = pd.DataFrame(entity_rows)
    if not entities_df.empty:
        entities_df = entities_df.loc[:, ENTITIES_FINAL_COLUMNS]
    else:
        entities_df = pd.DataFrame(columns=ENTITIES_FINAL_COLUMNS)

    text_unit_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for idx, text_unit_id in enumerate(sorted(text_unit_entity_ids), start=1):
        snippets = text_unit_snippets.get(text_unit_id, [])
        text = "\n".join(snippets[:5]).strip()
        if not text:
            text = f"Graph evidence for {text_unit_id}"
        document_id = text_unit_document_id.get(text_unit_id, f"document::{text_unit_id}")
        text_unit_rows.append(
            {
                "id": text_unit_id,
                "human_readable_id": idx,
                "text": text,
                "n_tokens": _approx_tokens(text),
                "document_id": document_id,
                "entity_ids": sorted(text_unit_entity_ids.get(text_unit_id, set())),
                "relationship_ids": sorted(text_unit_relationship_ids.get(text_unit_id, set())),
                "covariate_ids": None,
            }
        )
        document_rows.append(
            {
                "id": document_id,
                "human_readable_id": idx,
                "title": document_id,
                "text": text,
                "text_unit_ids": [text_unit_id],
                "creation_date": created_at,
                "raw_data": json.dumps({"source": "kg_relation_evidence", "text_unit_id": text_unit_id}, ensure_ascii=False),
            }
        )

    text_units_df = pd.DataFrame(text_unit_rows)
    if not text_units_df.empty:
        text_units_df = text_units_df.loc[:, TEXT_UNITS_FINAL_COLUMNS]
    else:
        text_units_df = pd.DataFrame(columns=TEXT_UNITS_FINAL_COLUMNS)

    documents_df = pd.DataFrame(document_rows)
    if not documents_df.empty:
        documents_df = documents_df.loc[:, DOCUMENTS_FINAL_COLUMNS]
    else:
        documents_df = pd.DataFrame(columns=DOCUMENTS_FINAL_COLUMNS)

    summary = {
        "entities": len(entities_df),
        "relationships": len(relationships_df),
        "text_units": len(text_units_df),
        "documents": len(documents_df),
    }
    return entities_df, relationships_df, text_units_df, documents_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export current KG CSVs to official GraphRAG BYOG parquet tables.")
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("drug_kg/final_kg/neo4j_import_v1/nodes.csv"),
        help="Path to the cleaned node CSV.",
    )
    parser.add_argument(
        "--relations-csv",
        type=Path,
        default=Path("drug_kg/final_kg/neo4j_import_v1/relations.csv"),
        help="Path to the cleaned relation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/official_byog/output"),
        help="Directory for official GraphRAG parquet outputs.",
    )
    args = parser.parse_args()

    nodes = _load_csv(args.nodes_csv)
    relations = _load_csv(args.relations_csv)
    entities_df, relationships_df, text_units_df, documents_df, summary = build_tables(
        nodes=nodes,
        relations=relations,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entities_df.to_parquet(args.output_dir / "entities.parquet", index=False)
    relationships_df.to_parquet(args.output_dir / "relationships.parquet", index=False)
    text_units_df.to_parquet(args.output_dir / "text_units.parquet", index=False)
    documents_df.to_parquet(args.output_dir / "documents.parquet", index=False)

    summary["output_dir"] = str(args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
