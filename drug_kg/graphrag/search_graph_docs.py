from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import dashscope
import faiss
import numpy as np
from dashscope import TextEmbedding
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return str(value).strip()


class DashScopeEmbeddings:
    def __init__(self, model: str):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        resp = TextEmbedding.call(model=self.model, input=[text])
        if not resp or not getattr(resp, "output", None) or "embeddings" not in resp.output:
            raise RuntimeError(f"DashScope embedding failed: model={self.model}, resp={resp}")
        return resp.output["embeddings"][0]["embedding"]


def _load_meta(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _search_one(query_vec: np.ndarray, index_path: Path, meta_path: Path, top_k: int) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    meta = _load_meta(meta_path)
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = dict(meta[int(idx)])
        row["score"] = float(score)
        results.append(row)
    return results


def search_graph_docs(*, query: str, index_dir: Path, top_k: int) -> dict[str, list[dict[str, Any]]]:
    dashscope.api_key = _get_env("DASHSCOPE_API_KEY")
    embeddings = DashScopeEmbeddings(model=_get_env("EMBEDDING_MODEL"))
    query_vec = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    query_vec = _normalize(query_vec)

    results: dict[str, list[dict[str, Any]]] = {}
    for stem in ("node_docs", "edge_docs", "subgraph_docs", "community_docs"):
        index_path = index_dir / f"{stem}.index"
        meta_path = index_dir / f"{stem}_meta.jsonl"
        if not index_path.exists() or not meta_path.exists():
            continue
        results[stem] = _search_one(query_vec, index_path, meta_path, top_k)
    return results


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Search GraphRAG graph documents by semantic similarity")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("drug_kg/graphrag/index_standard"),
        help="Directory containing graph embedding matrices and metadata",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K results per index")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional UTF-8 JSON file path to write results for reliable inspection",
    )
    args = parser.parse_args()

    results = search_graph_docs(query=args.query, index_dir=args.index_dir, top_k=args.top_k)

    output_text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
