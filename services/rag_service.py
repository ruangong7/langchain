"""RAG服务 - 处理检索和上下文构建"""
from typing import Any, List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
from config import RAG_FINAL_TOP_K, RAG_MAX_CONTEXT_CHARS, RAG_MAX_DOC_CHARS
from services.cross_encoder_reranker import CrossEncoderReranker
from retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def document_redis_key(doc: Document) -> str:
    """从检索到的 Document 解析与 Redis 一致的 key（与测试集 CSV 中 redis_key 列对齐）。"""
    m = doc.metadata or {}
    return str(m.get("legacy_key") or m.get("id") or "").strip()


class RAGService:
    """RAG服务类"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        title_index: Dict[str, str],
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.retriever = retriever
        self.title_index = title_index
        self.reranker = reranker
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """检索相关文档（用于 Hit@K / 召回评测）。"""
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        return self.retriever.get_relevant_documents(query)
    
    def retrieve_context(self, query: str) -> str:
        """检索相关上下文"""
        context, _ = self.retrieve_context_with_meta(query)
        return context

    def retrieve_context_with_meta(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """检索相关上下文，并返回可观测的 chunk 元信息。"""
        documents = self.retrieve_documents(query)
        documents = HybridRetriever.score_documents(query, documents)
        documents = self._rerank_documents(query, documents)
        final_docs = documents[:RAG_FINAL_TOP_K]
        context, used_chunks = self.format_context(final_docs)
        return context, self._build_retrieval_meta(
            queries=[query],
            ranked_documents=documents,
            used_chunks=used_chunks,
        )

    def retrieve_context_multi(self, queries: List[str]) -> str:
        """基于多条改写 query 检索，并按文档 key 合并去重。"""
        context, _ = self.retrieve_context_multi_with_meta(queries)
        return context

    def retrieve_context_multi_with_meta(self, queries: List[str]) -> Tuple[str, Dict[str, Any]]:
        """多 query 检索，并返回最终使用 chunk 与高分 chunk 概览。"""
        clean_queries = []
        seen_queries = set()
        for query in queries:
            normalized = " ".join(str(query or "").split())
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                clean_queries.append(normalized)
        if not clean_queries:
            return "", self._empty_retrieval_meta()
        if len(clean_queries) == 1:
            return self.retrieve_context_with_meta(clean_queries[0])

        merged: Dict[str, Document] = {}
        for query_idx, query in enumerate(clean_queries, start=1):
            for doc in self.retrieve_documents(query):
                key = self._document_key(doc)
                metadata = dict(doc.metadata or {})
                matched_queries = metadata.get("matched_queries") or []
                if not isinstance(matched_queries, list):
                    matched_queries = []
                matched_queries.append(query)
                metadata["matched_queries"] = list(dict.fromkeys(matched_queries))
                metadata["query_variant_rank"] = query_idx
                if key in merged:
                    old_metadata = dict(merged[key].metadata or {})
                    old_queries = old_metadata.get("matched_queries") or []
                    if not isinstance(old_queries, list):
                        old_queries = []
                    old_metadata["matched_queries"] = list(dict.fromkeys(old_queries + metadata["matched_queries"]))
                    old_metadata["query_match_count"] = len(old_metadata["matched_queries"])
                    merged[key].metadata = old_metadata
                else:
                    metadata["query_match_count"] = len(metadata["matched_queries"])
                    merged[key] = Document(page_content=doc.page_content, metadata=metadata)

        ranking_query = " ".join(clean_queries)
        documents = HybridRetriever.score_documents(ranking_query, list(merged.values()))
        documents = self._rerank_documents(ranking_query, documents)
        logger.info("RAG多查询检索完成，queries=%d merged_documents=%d", len(clean_queries), len(documents))
        final_docs = documents[:RAG_FINAL_TOP_K]
        context, used_chunks = self.format_context(final_docs)
        return context, self._build_retrieval_meta(
            queries=clean_queries,
            ranked_documents=documents,
            used_chunks=used_chunks,
        )

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        if self.reranker is None:
            return documents
        return self.reranker.rerank(query, documents)
    
    def _document_source(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        text = doc.page_content
        return str(
            metadata.get("source")
            or metadata.get("source_file")
            or metadata.get("title")
            or self.title_index.get(text)
            or "未知来源"
        )

    def _document_key(self, doc: Document) -> str:
        return document_redis_key(doc) or doc.page_content.strip()

    def format_context(self, documents: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """将 Document 列表拼成受长度约束、可追踪来源的上下文字符串。"""
        context_parts = []
        seen_keys = set()
        total_chars = 0
        used_chunks: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents, start=1):
            key = self._document_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            text = " ".join(doc.page_content.split())
            if not text:
                continue
            if len(text) > RAG_MAX_DOC_CHARS:
                text = f"{text[:RAG_MAX_DOC_CHARS]}..."

            source = self._document_source(doc)
            metadata = doc.metadata or {}
            rank = (
                metadata.get("cross_encoder_rank")
                or metadata.get("hybrid_rank")
                or metadata.get("dense_rank")
                or metadata.get("sparse_rank")
                or idx
            )
            score = None
            for score_key in ("cross_encoder_score", "rrf_score", "bm25_score"):
                value = metadata.get(score_key)
                if isinstance(value, (int, float)):
                    score = float(value)
                    break
            score_text = f"\n【分数】{score:.6f}" if isinstance(score, float) else ""
            part = f"【证据{len(context_parts) + 1}】\n【来源】{source}\n【排序】{rank}{score_text}\n【内容】{text}"

            if total_chars + len(part) > RAG_MAX_CONTEXT_CHARS:
                remaining = RAG_MAX_CONTEXT_CHARS - total_chars
                if remaining > 200:
                    context_parts.append(part[:remaining])
                    used_chunks.append(self._serialize_chunk(doc, idx, text, truncated=True))
                break

            context_parts.append(part)
            total_chars += len(part)
            used_chunks.append(self._serialize_chunk(doc, idx, text, truncated=False))

        context = "\n\n---\n\n".join(context_parts)
        logger.info("RAG检索完成，documents=%d deduped=%d context长度=%d", len(documents), len(context_parts), len(context))
        return context, used_chunks

    def _build_retrieval_meta(
        self,
        *,
        queries: List[str],
        ranked_documents: List[Document],
        used_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        top_chunks = []
        for idx, doc in enumerate(ranked_documents[:5], start=1):
            text = " ".join((doc.page_content or "").split())
            if len(text) > 220:
                text = text[:220] + "..."
            top_chunks.append(self._serialize_chunk(doc, idx, text, truncated=False))

        return {
            "queries": list(queries),
            "used_chunks": used_chunks,
            "top_chunks": top_chunks,
            "used_chunk_count": len(used_chunks),
            "top_chunk_count": len(top_chunks),
        }

    @staticmethod
    def _empty_retrieval_meta() -> Dict[str, Any]:
        return {
            "queries": [],
            "used_chunks": [],
            "top_chunks": [],
            "used_chunk_count": 0,
            "top_chunk_count": 0,
        }

    def _serialize_chunk(self, doc: Document, fallback_rank: int, text: str, *, truncated: bool) -> Dict[str, Any]:
        metadata = dict(doc.metadata or {})
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("legacy_key")
            or metadata.get("id")
            or self._document_key(doc)
        )
        return {
            "chunk_id": str(chunk_id or "").strip(),
            "source": self._document_source(doc),
            "source_type": str(metadata.get("source_type") or "").strip(),
            "rank": (
                metadata.get("cross_encoder_rank")
                or metadata.get("rerank_rank")
                or metadata.get("hybrid_rank")
                or metadata.get("dense_rank")
                or metadata.get("sparse_rank")
                or fallback_rank
            ),
            "score": self._pick_score(metadata),
            "cross_encoder_score": self._maybe_float(metadata.get("cross_encoder_score")),
            "rrf_score": self._maybe_float(metadata.get("rrf_score")),
            "bm25_score": self._maybe_float(metadata.get("bm25_score")),
            "rerank_score": self._maybe_float(metadata.get("rerank_score")),
            "retrieval_sources": list(metadata.get("retrieval_sources") or []),
            "matched_queries": list(metadata.get("matched_queries") or []),
            "query_match_count": int(metadata.get("query_match_count") or 0),
            "text_preview": text,
            "truncated": bool(truncated),
        }

    @staticmethod
    def _pick_score(metadata: Dict[str, Any]) -> Optional[float]:
        for key in ("cross_encoder_score", "rerank_score", "rrf_score", "bm25_score"):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _maybe_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None
