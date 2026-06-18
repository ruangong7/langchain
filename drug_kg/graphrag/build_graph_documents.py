from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _entity_name(entity: dict[str, Any]) -> str:
    return _norm(entity.get("canonical_name") or entity.get("name"))


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return (_norm(entity.get("type")), _entity_name(entity))


def _edge_key(
    source_entity: dict[str, Any],
    rel_type: str,
    target_entity: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        _norm(source_entity.get("type")),
        _entity_name(source_entity),
        rel_type,
        _norm(target_entity.get("type")),
        _entity_name(target_entity),
    )


def _join_nonempty(parts: list[str], sep: str = "；") -> str:
    return sep.join([part for part in parts if part])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_graph_documents(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    node_map: dict[tuple[str, str], dict[str, Any]] = {}
    edge_map: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    adjacency: dict[tuple[str, str], list[tuple[str, str, str, str, str]]] = defaultdict(list)

    for record in records:
        if not isinstance(record, dict):
            continue

        entities = {
            _norm(entity.get("id")): entity
            for entity in (record.get("entities") or [])
            if isinstance(entity, dict) and _norm(entity.get("id"))
        }

        for entity in entities.values():
            key = _entity_key(entity)
            node = node_map.setdefault(
                key,
                {
                    "entity_id": f"{key[0]}::{key[1]}",
                    "entity_type": key[0],
                    "entity_name": key[1],
                    "aliases": set(),
                    "properties": entity.get("properties") or {},
                    "evidence_chunks": set(),
                    "related_edges": set(),
                },
            )
            raw_name = _norm(entity.get("name"))
            if raw_name and raw_name != key[1]:
                node["aliases"].add(raw_name)
            chunk_id = _norm(record.get("chunk_index"))
            if chunk_id:
                node["evidence_chunks"].add(chunk_id)

        for relation in (record.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            source_entity = entities.get(_norm(relation.get("source_id")))
            target_entity = entities.get(_norm(relation.get("target_id")))
            if not isinstance(source_entity, dict) or not isinstance(target_entity, dict):
                continue

            rel_type = _norm(relation.get("type"))
            edge_key = _edge_key(source_entity, rel_type, target_entity)
            props = relation.get("properties") or {}
            data_text = _norm(props.get("data"))
            edge = edge_map.setdefault(
                edge_key,
                {
                    "edge_id": "::".join(edge_key),
                    "source_type": edge_key[0],
                    "source_name": edge_key[1],
                    "relation_type": edge_key[2],
                    "target_type": edge_key[3],
                    "target_name": edge_key[4],
                    "evidence": [],
                    "chunk_ids": set(),
                },
            )
            if data_text:
                edge["evidence"].append(data_text)
            chunk_id = _norm(props.get("chunk_id") or record.get("chunk_index"))
            if chunk_id:
                edge["chunk_ids"].add(chunk_id)

            adjacency[_entity_key(source_entity)].append(edge_key)
            adjacency[_entity_key(target_entity)].append(edge_key)
            node_map[_entity_key(source_entity)]["related_edges"].add(edge["edge_id"])
            node_map[_entity_key(target_entity)]["related_edges"].add(edge["edge_id"])

    edge_docs: list[dict[str, Any]] = []
    for edge in edge_map.values():
        unique_evidence = []
        seen_evidence = set()
        for item in edge["evidence"]:
            if item and item not in seen_evidence:
                unique_evidence.append(item)
                seen_evidence.add(item)
        evidence_preview = " | ".join(unique_evidence[:3])
        text = _join_nonempty(
            [
                f"{edge['source_name']}（{edge['source_type']}）",
                f"关系：{edge['relation_type']}",
                f"{edge['target_name']}（{edge['target_type']}）",
                f"证据：{evidence_preview}" if evidence_preview else "",
            ],
            sep="；",
        )
        edge_docs.append(
            {
                "doc_type": "edge",
                "doc_id": edge["edge_id"],
                "source_type": edge["source_type"],
                "source_name": edge["source_name"],
                "relation_type": edge["relation_type"],
                "target_type": edge["target_type"],
                "target_name": edge["target_name"],
                "chunk_ids": sorted(edge["chunk_ids"]),
                "evidence": unique_evidence,
                "text": text,
            }
        )

    node_docs: list[dict[str, Any]] = []
    for key, node in node_map.items():
        related_edge_docs = []
        for edge_key in adjacency.get(key, [])[:12]:
            edge = edge_map[edge_key]
            related_edge_docs.append(f"{edge['relation_type']} -> {edge['target_name']}" if edge["source_name"] == key[1] else f"{edge['source_name']} -> {edge['relation_type']}")
        text = _join_nonempty(
            [
                f"实体：{node['entity_name']}",
                f"类型：{node['entity_type']}",
                f"别名：{', '.join(sorted(node['aliases']))}" if node["aliases"] else "",
                f"相关图关系：{'；'.join(related_edge_docs)}" if related_edge_docs else "",
            ],
            sep="；",
        )
        node_docs.append(
            {
                "doc_type": "node",
                "doc_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "entity_name": node["entity_name"],
                "aliases": sorted(node["aliases"]),
                "properties": node["properties"],
                "chunk_ids": sorted(node["evidence_chunks"]),
                "related_edge_ids": sorted(node["related_edges"]),
                "text": text,
            }
        )

    subgraph_docs: list[dict[str, Any]] = []
    for key, node in node_map.items():
        if node["entity_type"] != "Drug":
            continue
        local_edges = []
        local_evidence = []
        for edge_key in adjacency.get(key, [])[:20]:
            edge = edge_map[edge_key]
            local_edges.append(f"{edge['source_name']} -[{edge['relation_type']}]-> {edge['target_name']}")
            if edge["evidence"]:
                local_evidence.append(edge["evidence"][0])
        text = _join_nonempty(
            [
                f"主实体：{node['entity_name']}（Drug）",
                f"局部子图关系：{'；'.join(local_edges)}" if local_edges else "",
                f"局部证据：{' | '.join(local_evidence[:5])}" if local_evidence else "",
            ],
            sep="；",
        )
        subgraph_docs.append(
            {
                "doc_type": "subgraph",
                "doc_id": f"subgraph::{node['entity_name']}",
                "center_type": node["entity_type"],
                "center_name": node["entity_name"],
                "edge_ids": sorted(node["related_edges"]),
                "text": text,
            }
        )

    node_docs.sort(key=lambda item: (item["entity_type"], item["entity_name"]))
    edge_docs.sort(key=lambda item: item["doc_id"])
    subgraph_docs.sort(key=lambda item: item["center_name"])
    return node_docs, edge_docs, subgraph_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphRAG-ready graph documents from KG extraction output")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("drug_kg/structured/output/real_drug_structured_extract_input.json"),
        help="KG extraction JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/output"),
        help="Directory for GraphRAG document JSONL files",
    )
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Input must be a JSON array: {args.input}")

    node_docs, edge_docs, subgraph_docs = build_graph_documents(records)
    _write_jsonl(args.output_dir / "node_docs.jsonl", node_docs)
    _write_jsonl(args.output_dir / "edge_docs.jsonl", edge_docs)
    _write_jsonl(args.output_dir / "subgraph_docs.jsonl", subgraph_docs)

    summary = {
        "node_docs": len(node_docs),
        "edge_docs": len(edge_docs),
        "subgraph_docs": len(subgraph_docs),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
