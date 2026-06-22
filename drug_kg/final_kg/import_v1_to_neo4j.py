from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = FINAL_KG_DIR / "neo4j_import_v1"


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import v1 KG CSV files into Neo4j.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--uri", default="bolt://127.0.0.1:7688")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4j123456")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--wipe", action="store_true", help="Delete all existing nodes and relationships before import.")
    args = parser.parse_args()

    nodes = _load_csv(args.input_dir / "nodes.csv")
    relations = _load_csv(args.input_dir / "relations.csv")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as session:
        if args.wipe:
            session.run("MATCH (n) DETACH DELETE n")

        session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE")
        nodes_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in nodes:
            nodes_by_label[row[":LABEL"]].append(row)

        for label, rows in nodes_by_label.items():
            for batch in _chunked(rows, args.batch_size):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:Entity:{label} {{entity_id: row.entity_id}})
                    SET n.entity_type = row.entity_type,
                        n.canonical_name = row.canonical_name,
                        n.normalized_name = row.normalized_name,
                        n.aliases = row.aliases,
                        n.source_types = row.source_types,
                        n.source_records = toInteger(row.source_records),
                        n.properties_json = row.properties_json
                    """,
                    rows=[
                        {
                            "entity_id": row[":ID"],
                            "entity_type": row["entity_type"],
                            "canonical_name": row["canonical_name"],
                            "normalized_name": row["normalized_name"],
                            "aliases": row["aliases"],
                            "source_types": row["source_types"],
                            "source_records": row["source_records:int"],
                            "properties_json": row["properties_json"],
                        }
                        for row in batch
                    ],
                )

        session.run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)")
        session.run("CREATE INDEX canonical_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name)")

        rels_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in relations:
            rels_by_type[row[":TYPE"]].append(row)

        for rel_type, rows in rels_by_type.items():
            for batch in _chunked(rows, args.batch_size):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (s:Entity {{entity_id: row.start_id}})
                    MATCH (t:Entity {{entity_id: row.end_id}})
                    MERGE (s)-[r:{rel_type} {{relation_id: row.relation_id}}]->(t)
                    SET r.source_type = row.source_type,
                        r.source_name = row.source_name,
                        r.target_type = row.target_type,
                        r.target_name = row.target_name,
                        r.chunk_ids = row.chunk_ids,
                        r.source_types = row.source_types,
                        r.evidence_count = toInteger(row.evidence_count),
                        r.evidence_json = row.evidence_json
                    """,
                    rows=[
                        {
                            "start_id": row[":START_ID"],
                            "end_id": row[":END_ID"],
                            "relation_id": row["relation_id"],
                            "source_type": row["source_type"],
                            "source_name": row["source_name"],
                            "target_type": row["target_type"],
                            "target_name": row["target_name"],
                            "chunk_ids": row["chunk_ids"],
                            "source_types": row["source_types"],
                            "evidence_count": row["evidence_count:int"],
                            "evidence_json": row["evidence_json"],
                        }
                        for row in batch
                    ],
                )

    driver.close()


if __name__ == "__main__":
    main()
