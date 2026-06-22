from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _join_nonempty(parts: list[str], sep: str = "；") -> str:
    return sep.join([part for part in parts if part])


def _load_evidence(evidence_json: str) -> list[str]:
    try:
        evidence = json.loads(evidence_json) if evidence_json else []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
    except Exception:
        evidence = [evidence_json] if evidence_json else []
    return [_norm(item) for item in evidence if _norm(item)]


def build_documents(
    *,
    nodes: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    top_edges_per_node: int = 12,
    top_edges_per_subgraph: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rel_type_counter_by_node: dict[str, Counter[str]] = defaultdict(Counter)

    edge_docs: list[dict[str, Any]] = []
    for row in relations:
        start_id = _norm(row.get(":START_ID"))
        end_id = _norm(row.get(":END_ID"))
        rel_type = _norm(row.get(":TYPE"))
        source_name = _norm(row.get("source_name"))
        target_name = _norm(row.get("target_name"))
        source_type = _norm(row.get("source_type"))
        target_type = _norm(row.get("target_type"))
        evidence = _load_evidence(_norm(row.get("evidence_json")))
        evidence_preview = " | ".join(evidence[:3])

        edge_doc = {
            "doc_type": "edge",
            "doc_id": _norm(row.get("relation_id")),
            "source_id": start_id,
            "source_type": source_type,
            "source_name": source_name,
            "relation_type": rel_type,
            "target_id": end_id,
            "target_type": target_type,
            "target_name": target_name,
            "chunk_ids": _norm(row.get("chunk_ids")).split("|") if _norm(row.get("chunk_ids")) else [],
            "evidence": evidence,
            "text": _join_nonempty(
                [
                    f"源实体：{source_name}（{source_type}）" if source_name else "",
                    f"关系：{rel_type}" if rel_type else "",
                    f"目标实体：{target_name}（{target_type}）" if target_name else "",
                    f"证据：{evidence_preview}" if evidence_preview else "",
                ]
            ),
        }
        edge_docs.append(edge_doc)
        adjacency[start_id].append(edge_doc)
        adjacency[end_id].append(edge_doc)
        rel_type_counter_by_node[start_id][rel_type] += 1
        rel_type_counter_by_node[end_id][rel_type] += 1

    node_docs: list[dict[str, Any]] = []
    for row in nodes:
        entity_id = _norm(row.get(":ID"))
        entity_name = _norm(row.get("canonical_name"))
        entity_type = _norm(row.get("entity_type"))
        aliases = [_norm(x) for x in _norm(row.get("aliases")).split("|") if _norm(x)]
        related_edges = adjacency.get(entity_id, [])
        related_edge_ids = [edge["doc_id"] for edge in related_edges]
        related_edge_briefs: list[str] = []
        for edge in related_edges[:top_edges_per_node]:
            if edge["source_id"] == entity_id:
                related_edge_briefs.append(f"{edge['relation_type']} -> {edge['target_name']}")
            else:
                related_edge_briefs.append(f"{edge['source_name']} -> {edge['relation_type']}")
        rel_stats = rel_type_counter_by_node.get(entity_id, Counter())
        rel_summary = "；".join(f"{k}:{v}" for k, v in rel_stats.most_common(5))

        node_docs.append(
            {
                "doc_type": "node",
                "doc_id": entity_id,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "aliases": aliases,
                "source_types": _norm(row.get("source_types")).split("|") if _norm(row.get("source_types")) else [],
                "related_edge_ids": related_edge_ids,
                "text": _join_nonempty(
                    [
                        f"实体：{entity_name}" if entity_name else "",
                        f"类型：{entity_type}" if entity_type else "",
                        f"别名：{', '.join(aliases)}" if aliases else "",
                        f"关系分布：{rel_summary}" if rel_summary else "",
                        f"相关图关系：{'；'.join(related_edge_briefs)}" if related_edge_briefs else "",
                    ]
                ),
            }
        )

    subgraph_docs: list[dict[str, Any]] = []
    for row in nodes:
        entity_id = _norm(row.get(":ID"))
        entity_name = _norm(row.get("canonical_name"))
        entity_type = _norm(row.get("entity_type"))
        if entity_type != "Drug":
            continue
        local_edges = adjacency.get(entity_id, [])
        local_briefs: list[str] = []
        local_evidence: list[str] = []
        edge_ids: list[str] = []
        for edge in local_edges[:top_edges_per_subgraph]:
            edge_ids.append(edge["doc_id"])
            local_briefs.append(f"{edge['source_name']} -[{edge['relation_type']}]-> {edge['target_name']}")
            if edge["evidence"]:
                local_evidence.append(edge["evidence"][0])
        subgraph_docs.append(
            {
                "doc_type": "subgraph",
                "doc_id": f"subgraph::{entity_name}",
                "center_id": entity_id,
                "center_type": entity_type,
                "center_name": entity_name,
                "edge_ids": edge_ids,
                "text": _join_nonempty(
                    [
                        f"中心实体：{entity_name}（Drug）",
                        f"局部子图关系：{'；'.join(local_briefs)}" if local_briefs else "",
                        f"局部证据：{' | '.join(local_evidence[:5])}" if local_evidence else "",
                    ]
                ),
            }
        )

    community_docs = build_community_docs(nodes=nodes, edge_docs=edge_docs)
    node_docs.sort(key=lambda item: (item["entity_type"], item["entity_name"]))
    edge_docs.sort(key=lambda item: item["doc_id"])
    subgraph_docs.sort(key=lambda item: item["center_name"])
    community_docs.sort(key=lambda item: item["community_id"])
    return node_docs, edge_docs, subgraph_docs, community_docs


def build_community_docs(*, nodes: list[dict[str, Any]], edge_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    graph = nx.Graph()
    node_meta = {_norm(row.get(":ID")): row for row in nodes}
    for node_id, row in node_meta.items():
        graph.add_node(
            node_id,
            name=_norm(row.get("canonical_name")),
            entity_type=_norm(row.get("entity_type")),
        )

    for edge in edge_docs:
        start_id = _norm(edge.get("source_id"))
        end_id = _norm(edge.get("target_id"))
        if not start_id or not end_id or start_id == end_id:
            continue
        graph.add_edge(
            start_id,
            end_id,
            relation_type=_norm(edge.get("relation_type")),
            weight=max(1, len(edge.get("chunk_ids") or [])),
            edge_doc=edge,
        )

    communities: list[set[str]] = []
    for component_nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(component_nodes).copy()
        if subgraph.number_of_nodes() <= 2:
            communities.append(set(subgraph.nodes()))
            continue
        detected = nx.algorithms.community.greedy_modularity_communities(subgraph, weight="weight")
        communities.extend(set(group) for group in detected)

    community_docs: list[dict[str, Any]] = []
    for idx, group in enumerate(communities, start=1):
        community_id = f"community_{idx:04d}"
        entity_counter = Counter()
        type_counter = Counter()
        relation_counter = Counter()
        evidence_snippets: list[str] = []
        member_names: list[str] = []
        edge_ids: list[str] = []
        seen_edge_ids = set()

        for node_id in group:
            row = node_meta.get(node_id)
            if not row:
                continue
            name = _norm(row.get("canonical_name"))
            entity_type = _norm(row.get("entity_type"))
            if name:
                member_names.append(name)
                entity_counter[name] += 1
            if entity_type:
                type_counter[entity_type] += 1

        for _, _, data in graph.subgraph(group).edges(data=True):
            relation_type = _norm(data.get("relation_type"))
            if relation_type:
                relation_counter[relation_type] += 1
            edge_doc = data.get("edge_doc") or {}
            edge_id = _norm(edge_doc.get("doc_id"))
            if edge_id and edge_id not in seen_edge_ids:
                seen_edge_ids.add(edge_id)
                edge_ids.append(edge_id)
            for snippet in edge_doc.get("evidence") or []:
                snippet = _norm(snippet)
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)
                if len(evidence_snippets) >= 5:
                    break

        top_entities = [name for name, _ in entity_counter.most_common(10)]
        top_relations = [f"{name}:{count}" for name, count in relation_counter.most_common(8)]
        type_summary = [f"{name}:{count}" for name, count in type_counter.most_common()]
        summary_text = _join_nonempty(
            [
                f"社区ID：{community_id}",
                f"成员数：{len(group)}",
                f"实体类型分布：{'；'.join(type_summary)}" if type_summary else "",
                f"核心实体：{'；'.join(top_entities[:8])}" if top_entities else "",
                f"高频关系：{'；'.join(top_relations)}" if top_relations else "",
                f"代表证据：{' | '.join(evidence_snippets[:3])}" if evidence_snippets else "",
            ]
        )
        community_docs.append(
            {
                "doc_type": "community",
                "doc_id": community_id,
                "community_id": community_id,
                "member_entity_ids": sorted(group),
                "member_entity_names": sorted(member_names),
                "edge_ids": edge_ids,
                "top_entities": top_entities,
                "top_relation_types": [name for name, _ in relation_counter.most_common(8)],
                "text": summary_text,
            }
        )

    return community_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standard GraphRAG docs including community reports from v1 KG CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("drug_kg/final_kg/neo4j_import_v1"),
        help="Directory containing nodes.csv and relations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/output_standard"),
        help="Directory for node_docs / edge_docs / subgraph_docs / community_docs",
    )
    args = parser.parse_args()

    nodes = _load_csv(args.input_dir / "nodes.csv")
    relations = _load_csv(args.input_dir / "relations.csv")
    node_docs, edge_docs, subgraph_docs, community_docs = build_documents(nodes=nodes, relations=relations)

    _write_jsonl(args.output_dir / "node_docs.jsonl", node_docs)
    _write_jsonl(args.output_dir / "edge_docs.jsonl", edge_docs)
    _write_jsonl(args.output_dir / "subgraph_docs.jsonl", subgraph_docs)
    _write_jsonl(args.output_dir / "community_docs.jsonl", community_docs)

    summary = {
        "node_docs": len(node_docs),
        "edge_docs": len(edge_docs),
        "subgraph_docs": len(subgraph_docs),
        "community_docs": len(community_docs),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
