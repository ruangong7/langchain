from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_QUERIES = [
    "糖尿病患者哪些药会影响肾功能",
    "华法林和什么药有相互作用",
    "孕妇慎用哪些药",
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_query(*, query: str, top_k: int, workdir: Path) -> dict:
    cmd = [
        sys.executable,
        "drug_kg/graphrag/search_graph_docs.py",
        query,
        "--top-k",
        str(top_k),
    ]
    completed = subprocess.run(
        cmd,
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return json.loads(completed.stdout)


def print_results(query: str, results: dict) -> None:
    print(f"\n=== Query: {query} ===")
    for doc_type in ("node_docs", "edge_docs", "subgraph_docs"):
        items = results.get(doc_type) or []
        print(f"\n[{doc_type}] top {len(items)}")
        for idx, item in enumerate(items, start=1):
            doc_id = item.get("doc_id", "")
            score = item.get("score", 0.0)
            text = str(item.get("text", "")).replace("\n", " ").strip()
            if len(text) > 180:
                text = text[:180] + "..."
            print(f"{idx}. score={score:.4f} doc_id={doc_id}")
            print(f"   text={text}")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Run quick GraphRAG retrieval tests against existing FAISS indexes")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Custom query to test. Can be provided multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top K per doc type")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parents[2]
    queries = args.query or DEFAULT_QUERIES

    for query in queries:
        results = run_query(query=query, top_k=args.top_k, workdir=workdir)
        print_results(query, results)


if __name__ == "__main__":
    main()
