from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import dashscope
import faiss
import numpy as np
from dashscope import TextEmbedding

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class DashScopeEmbeddings:
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        max_batch = 10
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i : i + max_batch]
            resp = TextEmbedding.call(model=self.model, input=batch)
            if not resp or not getattr(resp, "output", None) or "embeddings" not in resp.output:
                raise RuntimeError(f"DashScope embedding failed: model={self.model}, batch={len(batch)}, resp={resp}")
            all_vectors.extend([item["embedding"] for item in resp.output["embeddings"]])
        return all_vectors


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_one_index(
    *,
    input_path: Path,
    output_dir: Path,
    embeddings: DashScopeEmbeddings,
    batch_size: int,
) -> dict[str, Any]:
    docs = _load_jsonl(input_path)
    texts = [_norm(doc.get("text")) for doc in docs]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        logger.info("Embedding %s: %s/%s", input_path.name, end, len(texts))
        vectors.extend(embeddings.embed_documents(texts[start:end]))
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = _l2_normalize(matrix)

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(output_dir / f"{stem}.index"))

    meta_rows = []
    for idx, doc in enumerate(docs):
        item = dict(doc)
        item["vector_row"] = idx
        meta_rows.append(item)

    with (output_dir / f"{stem}_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "name": stem,
        "count": len(docs),
        "dim": int(matrix.shape[1]) if matrix.size else 0,
        "faiss_file": str((output_dir / f"{stem}.index").as_posix()),
        "matrix_file": str((output_dir / f"{stem}.npy").as_posix()),
        "meta_file": str((output_dir / f"{stem}_meta.jsonl").as_posix()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings for GraphRAG graph documents")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("drug_kg/graphrag/output"),
        help="Directory containing node_docs.jsonl / edge_docs.jsonl / subgraph_docs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/index"),
        help="Directory for embedding matrices and metadata",
    )
    parser.add_argument(
        "--doc-types",
        nargs="*",
        default=["node_docs", "edge_docs", "subgraph_docs"],
        help="Document types to build: node_docs edge_docs subgraph_docs",
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Number of docs per embedding progress batch")
    args = parser.parse_args()

    dashscope.api_key = DASHSCOPE_API_KEY
    embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)

    summaries = []
    for stem in args.doc_types:
        name = f"{stem}.jsonl"
        input_path = args.input_dir / name
        if not input_path.exists():
            continue
        summaries.append(
            _build_one_index(
                input_path=input_path,
                output_dir=args.output_dir,
                embeddings=embeddings,
                batch_size=max(1, args.batch_size),
            )
        )

    with (args.output_dir / "index_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(json.dumps({"indexes": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
