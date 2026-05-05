"""RAG服务 - 处理检索和上下文构建"""
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging

logger = logging.getLogger(__name__)


def document_redis_key(doc: Document) -> str:
    """从检索到的 Document 解析与 Redis 一致的 key（与测试集 CSV 中 redis_key 列对齐）。"""
    m = doc.metadata or {}
    return str(m.get("legacy_key") or m.get("id") or "").strip()


class RAGService:
    """RAG服务类"""
    
    def __init__(self, retriever: BaseRetriever, title_index: Dict[str, str]):
        self.retriever = retriever
        self.title_index = title_index
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """检索相关文档（用于 Hit@K / 召回评测）。"""
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        return self.retriever.get_relevant_documents(query)
    
    def retrieve_context(self, query: str) -> str:
        """检索相关上下文"""
        documents = self.retrieve_documents(query)
        return self.format_context(documents)
    
    def format_context(self, documents: List[Document]) -> str:
        """将 Document 列表拼成与线上一致的上下文字符串。"""
        context_parts = []
        for doc in documents:
            text = doc.page_content
            title = self.title_index.get(text, doc.metadata.get("title", "未知来源"))
            context_parts.append(f"【来源】{title}\n【内容】{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"RAG检索完成，context长度: {len(context)}")
        
        return context
