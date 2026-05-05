"""混合检索器 - 结合稠密检索和稀疏检索"""
from typing import List, Any, ClassVar
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import logging

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """混合检索器，使用RRF算法合并稠密和稀疏检索结果"""
    
    RRF_K: ClassVar[float] = 60.0  # RRF参数

    # pydantic 模型字段定义，避免运行时动态添加属性报错
    dense_retriever: Any = None
    sparse_retriever: Any = None
    top_k: int = 10
    
    def __init__(self, dense_retriever, sparse_retriever, top_k=10):
        super().__init__()
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """执行混合检索"""
        # 1. 分别执行稠密和稀疏检索
        dense_results = self._call_retriever(self.dense_retriever, query)
        sparse_results = self._call_retriever(self.sparse_retriever, query)
        
        logger.debug(f"[HybridRetriever] 稠密检索: {len(dense_results)} 条, 稀疏检索: {len(sparse_results)} 条")
        
        # 2. 使用RRF合并结果
        rrf_scores = {}
        content_map = {}
        source_map = {}
        dense_rank_map = {}
        sparse_rank_map = {}
        
        # 处理稠密检索结果
        for i, doc in enumerate(dense_results):
            key = self._get_content_key(doc)
            rrf_score = 1.0 / (self.RRF_K + i + 1)
            rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score
            source_map.setdefault(key, set()).add("dense")
            dense_rank_map[key] = i + 1
            if key not in content_map:
                content_map[key] = doc
        
        # 处理稀疏检索结果
        for i, doc in enumerate(sparse_results):
            key = self._get_content_key(doc)
            rrf_score = 1.0 / (self.RRF_K + i + 1)
            rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score
            source_map.setdefault(key, set()).add("sparse")
            sparse_rank_map[key] = i + 1
            if key not in content_map:
                content_map[key] = doc
        
        # 3. 按RRF分数排序，返回Top-K
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]
        
        merged_results = []
        for rank, (key, score) in enumerate(sorted_results, 1):
            doc = content_map[key]
            metadata = dict(doc.metadata or {})
            metadata["hybrid_rank"] = rank
            metadata["rrf_score"] = float(score)
            metadata["retrieval_sources"] = sorted(source_map.get(key, set()))
            if key in dense_rank_map:
                metadata["dense_rank"] = dense_rank_map[key]
            if key in sparse_rank_map:
                metadata["sparse_rank"] = sparse_rank_map[key]
            merged_results.append(Document(page_content=doc.page_content, metadata=metadata))
        
        logger.info(f"[HybridRetriever] 混合检索完成，返回 {len(merged_results)} 条结果")
        
        return merged_results

    def _call_retriever(self, retriever: Any, query: str) -> List[Document]:
        """兼容不同版本检索器接口（优先 invoke，其次 get_relevant_documents）。"""
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        return retriever.get_relevant_documents(query)
    
    def _get_content_key(self, doc: Document) -> str:
        """生成文档的唯一键（用于去重）"""
        return doc.page_content
