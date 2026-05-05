"""Build Neo4j import files from LLM chunk extraction outputs.

Input:
  drug_kg/LLM/output/*.json

Each file is a JSON array of records. Each record contains:
  entities, relations

Output (default under drug_kg/LLM/neo4j_import/):
  - nodes.csv
  - edges.csv
  - import.cypher

Then in Neo4j Browser (or cypher-shell), run import.cypher after copying CSVs
to Neo4j's import directory, or adjust the file URLs accordingly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _json(obj: Any) -> str:
    # store as compact JSON string in CSV
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _safe_label(label: str) -> str:
    # Neo4j label/type must be alnum and underscore
    label = re.sub(r"[^0-9A-Za-z_]", "_", label.strip())
    return label or "Unknown"


def _entity_key(e: dict[str, Any]) -> str:
    etype = _safe_label(_norm(e.get("type") or "Unknown"))
    canonical = _norm(e.get("canonical_name") or e.get("canonical") or e.get("name"))
    canonical = canonical or _norm(e.get("name"))
    return f"{etype}::{canonical}"


def _global_entity_id(e: dict[str, Any]) -> str:
    # stable across runs if canonical/type stable
    key = _entity_key(e)
    etype = _safe_label(_norm(e.get("type") or "Unknown"))
    return f"E_{etype}_{_sha1(key)[:12]}"


@dataclass(frozen=True)
class NodeRow:
    node_id: str
    labels: str
    type: str
    name: str
    canonical_name: str
    aliases: str
    properties: str
    evidence_text: str
    source_file: str
    record_index: int
    chunk_index: str
    source: str
    raw_text: str


@dataclass(frozen=True)
class EdgeRow:
    start_id: str
    end_id: str
    rel_type: str
    properties: str
    evidence_text: str
    source_file: str
    record_index: int


def iter_output_files(output_dir: Path) -> Iterable[Path]:
    yield from sorted(output_dir.glob("*.json"))


def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def build_rows(output_dir: Path) -> tuple[list[NodeRow], list[EdgeRow]]:
    nodes: dict[str, NodeRow] = {}
    edges: list[EdgeRow] = []

    for path in iter_output_files(output_dir):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue

            record_index = int(rec.get("record_index") or 0)
            chunk_index = _norm(rec.get("chunk_index"))
            source = _norm(rec.get("source"))
            text = _norm(rec.get("text"))

            # local-id -> global-id mapping within this record
            local_to_global: dict[str, str] = {}

            for e in _ensure_list(rec.get("entities")):
                if not isinstance(e, dict):
                    continue
                gid = _global_entity_id(e)
                local_id = _norm(e.get("id"))
                if local_id:
                    local_to_global[local_id] = gid

                n = NodeRow(
                    node_id=gid,
                    labels=_safe_label(_norm(e.get("type") or "Entity")),
                    type=_safe_label(_norm(e.get("type") or "Unknown")),
                    name=_norm(e.get("name")),
                    canonical_name=_norm(e.get("canonical_name") or e.get("canonical") or e.get("name")),
                    aliases=_json(_ensure_list(e.get("aliases"))),
                    properties=_json(e.get("properties") or {}),
                    evidence_text=_norm(e.get("evidence_text") or ""),
                    source_file=path.name,
                    record_index=record_index,
                    chunk_index=chunk_index,
                    source=source,
                    raw_text="",
                )
                # de-dup entities globally by gid
                if gid not in nodes:
                    nodes[gid] = n
                else:
                    # merge aliases/properties lightly
                    old = nodes[gid]
                    try:
                        old_alias = set(json.loads(old.aliases))
                        new_alias = set(_ensure_list(e.get("aliases")))
                        aliases = sorted({*(old_alias), *(new_alias)})
                    except Exception:
                        aliases = _ensure_list(e.get("aliases"))
                    # prefer non-empty evidence_text/name if missing
                    nodes[gid] = NodeRow(
                        node_id=old.node_id,
                        labels=old.labels,
                        type=old.type,
                        name=old.name or n.name,
                        canonical_name=old.canonical_name or n.canonical_name,
                        aliases=_json(aliases),
                        properties=old.properties if old.properties != "{}" else n.properties,
                        evidence_text=old.evidence_text or n.evidence_text,
                        source_file=old.source_file,
                        record_index=old.record_index,
                        chunk_index=old.chunk_index,
                        source=old.source,
                        raw_text=old.raw_text,
                    )

            # relations
            for rel in _ensure_list(rec.get("relations")):
                if not isinstance(rel, dict):
                    continue
                rel_type = _safe_label(_norm(rel.get("type") or "RELATED_TO"))
                s_local = _norm(rel.get("source_id"))
                t_local = _norm(rel.get("target_id"))
                s_gid = local_to_global.get(s_local)
                t_gid = local_to_global.get(t_local)
                if not s_gid or not t_gid:
                    # skip dangling edges
                    continue
                edges.append(
                    EdgeRow(
                        start_id=s_gid,
                        end_id=t_gid,
                        rel_type=rel_type,
                        properties=_json(rel.get("properties") or {}),
                        evidence_text=_norm(rel.get("evidence_text") or ""),
                        source_file=path.name,
                        record_index=record_index,
                    )
                )

    return list(nodes.values()), edges


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_import_cypher(path: Path) -> None:
    # This assumes nodes.csv/edges.csv are placed under Neo4j import directory.
    path.write_text(
        """// 1) Put nodes.csv and edges.csv into Neo4j's import directory
// 2) Run this script in Neo4j Browser

// Constraints (Neo4j 5+)
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;

// Load nodes
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
CALL {
  WITH row
  WITH row, row.labels AS labels
  // We will always create a base :Entity label, and also the specific label in `labels`
  CALL apoc.create.node(['Entity', labels], {
    id: row.id,
    type: row.type,
    name: row.name,
    canonical_name: row.canonical_name,
    aliases: row.aliases,
    properties: row.properties,
    evidence_text: row.evidence_text,
    source_file: row.source_file,
    record_index: toInteger(row.record_index),
    chunk_index: row.chunk_index,
    source: row.source,
    raw_text: row.raw_text
  }) YIELD node
  RETURN node
}
RETURN count(*) AS nodes_loaded;

// Load edges
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {id: row.start_id})
MATCH (t:Entity {id: row.end_id})
CALL apoc.create.relationship(s, row.type, {
  properties: row.properties,
  evidence_text: row.evidence_text,
  source_file: row.source_file,
  record_index: toInteger(row.record_index)
}, t) YIELD rel
RETURN count(*) AS edges_loaded;
""",
        encoding="utf-8",
    )


def main() -> None:
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build Neo4j CSV import from LLM outputs")
    ap.add_argument("--output-dir", type=Path, default=base / "output", help="LLM extraction output directory")
    ap.add_argument("--out-dir", type=Path, default=base / "neo4j_import", help="Directory to write CSVs and cypher")
    args = ap.parse_args()

    nodes, edges = build_rows(args.output_dir)

    out_nodes = args.out_dir / "nodes.csv"
    out_edges = args.out_dir / "edges.csv"
    out_cypher = args.out_dir / "import.cypher"

    write_csv(
        out_nodes,
        header=[
            "id",
            "labels",
            "type",
            "name",
            "canonical_name",
            "aliases",
            "properties",
            "evidence_text",
            "source_file",
            "record_index",
            "chunk_index",
            "source",
            "raw_text",
        ],
        rows=(
            {
                "id": n.node_id,
                "labels": n.labels,
                "type": n.type,
                "name": n.name,
                "canonical_name": n.canonical_name,
                "aliases": n.aliases,
                "properties": n.properties,
                "evidence_text": n.evidence_text,
                "source_file": n.source_file,
                "record_index": n.record_index,
                "chunk_index": n.chunk_index,
                "source": n.source,
                "raw_text": n.raw_text,
            }
            for n in nodes
        ),
    )

    write_csv(
        out_edges,
        header=[
            "start_id",
            "end_id",
            "type",
            "properties",
            "evidence_text",
            "source_file",
            "record_index",
        ],
        rows=(
            {
                "start_id": e.start_id,
                "end_id": e.end_id,
                "type": e.rel_type,
                "properties": e.properties,
                "evidence_text": e.evidence_text,
                "source_file": e.source_file,
                "record_index": e.record_index,
            }
            for e in edges
        ),
    )

    write_import_cypher(out_cypher)
    print(f"Wrote: {out_nodes}")
    print(f"Wrote: {out_edges}")
    print(f"Wrote: {out_cypher}")
    print(f"Nodes: {len(nodes)}  Edges: {len(edges)}")


if __name__ == "__main__":
    main()

