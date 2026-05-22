"""RAG服务 - 处理检索和上下文构建"""
from typing import List, Dict, Optional
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
        documents = self.retrieve_documents(query)
        documents = HybridRetriever.score_documents(query, documents)
        documents = self._rerank_documents(query, documents)
        return self.format_context(documents[:RAG_FINAL_TOP_K])

    def retrieve_context_multi(self, queries: List[str]) -> str:
        """基于多条改写 query 检索，并按文档 key 合并去重。"""
        clean_queries = []
        seen_queries = set()
        for query in queries:
            normalized = " ".join(str(query or "").split())
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                clean_queries.append(normalized)
        if not clean_queries:
            return ""
        if len(clean_queries) == 1:
            return self.retrieve_context(clean_queries[0])

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
        return self.format_context(documents[:RAG_FINAL_TOP_K])

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

    def format_context(self, documents: List[Document]) -> str:
        """将 Document 列表拼成受长度约束、可追踪来源的上下文字符串。"""
        context_parts = []
        seen_keys = set()
        total_chars = 0

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
                break

            context_parts.append(part)
            total_chars += len(part)

        context = "\n\n---\n\n".join(context_parts)
        logger.info("RAG检索完成，documents=%d deduped=%d context长度=%d", len(documents), len(context_parts), len(context))
        
        return context
