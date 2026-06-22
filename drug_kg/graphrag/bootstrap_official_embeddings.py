from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from graphrag.config.embeddings import (
    community_full_content_embedding,
    entity_description_embedding,
    text_unit_text_embedding,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.load_config import load_config
from graphrag_vectors import VectorStoreDocument, create_vector_store


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")


def _build_documents(
    *,
    rows: list[dict[str, Any]],
    id_key: str,
    text_key: str,
    extra_fields: list[str] | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    docs: list[tuple[str, str, dict[str, Any]]] = []
    extra_fields = extra_fields or []
    for row in rows:
        doc_id = _norm(row.get(id_key))
        text = _norm(row.get(text_key))
        if not doc_id or not text:
            continue
        data = {field: row.get(field) for field in extra_fields if field in row}
        docs.append((doc_id, text, data))
    return docs


def _write_index(
    *,
    config: GraphRagConfig,
    embedding_name: str,
    docs: list[tuple[str, str, dict[str, Any]]],
    model: SentenceTransformer,
) -> int:
    if not docs:
        return 0
    schema = config.vector_store.index_schema[embedding_name]
    schema.vector_size = int(model.get_sentence_embedding_dimension())
    store = create_vector_store(config.vector_store, schema)
    store.connect()
    store.create_index()

    texts = [text for _, text, _ in docs]
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    documents = []
    for (doc_id, _, data), vector in zip(docs, vectors, strict=True):
        documents.append(
            VectorStoreDocument(
                id=doc_id,
                vector=vector.tolist(),
                data=data,
            )
        )
    store.load_documents(documents)
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap official GraphRAG LanceDB embeddings using a local sentence-transformers model.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("drug_kg/graphrag/official_project"),
        help="Official GraphRAG project root for loading settings.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/official_byog/output"),
        help="Directory containing official GraphRAG parquet outputs.",
    )
    parser.add_argument(
        "--model-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name for local bootstrap embeddings.",
    )
    args = parser.parse_args()

    config = load_config(root_dir=args.project_root)
    model = SentenceTransformer(args.model_name)

    entities = pd.read_parquet(args.output_dir / "entities.parquet")
    community_reports = pd.read_parquet(args.output_dir / "community_reports.parquet")
    text_units = pd.read_parquet(args.output_dir / "text_units.parquet")

    entity_docs = _build_documents(
        rows=_to_rows(entities),
        id_key="id",
        text_key="description",
        extra_fields=["title", "type"],
    )
    community_docs = _build_documents(
        rows=_to_rows(community_reports),
        id_key="id",
        text_key="full_content",
        extra_fields=["community", "title"],
    )
    text_unit_docs = _build_documents(
        rows=_to_rows(text_units),
        id_key="id",
        text_key="text",
        extra_fields=["document_id"],
    )

    summary = {
        entity_description_embedding: _write_index(
            config=config,
            embedding_name=entity_description_embedding,
            docs=entity_docs,
            model=model,
        ),
        community_full_content_embedding: _write_index(
            config=config,
            embedding_name=community_full_content_embedding,
            docs=community_docs,
            model=model,
        ),
        text_unit_text_embedding: _write_index(
            config=config,
            embedding_name=text_unit_text_embedding,
            docs=text_unit_docs,
            model=model,
        ),
        "vector_dim": int(model.get_sentence_embedding_dimension()),
        "db_uri": config.vector_store.db_uri,
        "model_name": args.model_name,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
