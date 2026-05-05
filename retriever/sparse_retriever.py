"""稀疏检索器 - 使用BM25算法"""
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
import jieba
import logging
import os

logger = logging.getLogger(__name__)

class SparseRetriever(BaseRetriever):
    """基于BM25的稀疏检索器"""

    # pydantic 模型字段定义，避免运行时动态添加属性报错
    top_k: int = 10
    documents: List[Document] = []
    title_index: Dict[str, str] = {}
    bm25: Any = None

    def __init__(self, documents: List[Document], top_k=10):
        super().__init__()
        self.top_k = top_k
        self.documents = documents
        self.title_index = {}  # 文本 -> 标题的映射
        
        # 构建BM25索引
        self._build_index()
    
    def _build_index(self):
        """构建BM25索引"""
        # 分词处理
        tokenized_docs = []
        for doc in self.documents:
            # 使用jieba分词
            tokens = list(jieba.cut(doc.page_content))
            tokenized_docs.append(tokens)
        
        # 构建BM25模型
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"[SparseRetriever] BM25索引构建完成，共 {len(self.documents)} 条文档")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """执行稀疏检索"""
        # 对查询进行分词
        query_tokens = list(jieba.cut(query))
        
        # 计算BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取Top-K结果
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.top_k]
        
        results = []
        for rank, doc_index in enumerate(top_indices, 1):
            doc = self.documents[doc_index]
            metadata = dict(doc.metadata or {})
            metadata["sparse_rank"] = rank
            metadata["bm25_score"] = float(scores[doc_index])
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        logger.debug(f"[SparseRetriever] 检索完成，返回 {len(results)} 条结果")
        
        return results
    
    @classmethod
    def from_content_dir(cls, content_dir: str, top_k=10):
        """从content目录加载文档并构建检索器"""
        documents = []
        title_index = {}
        current_title = None
        
        # 读取markdown文件
        md_file = os.path.join(content_dir, "相互作用.md")
        if os.path.exists(md_file):
            with open(md_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 标题行：不以//开头
                    if not line.startswith("//"):
                        current_title = line
                        continue
                    
                    # 内容行：去掉//前缀
                    content = line[2:].strip()
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={"title": current_title or "未知来源"}
                        )
                        documents.append(doc)
                        title_index[content] = current_title or "未知来源"
        
        logger.info(f"[SparseRetriever] 从 {content_dir} 加载了 {len(documents)} 条文档")
        
        retriever = cls(documents, top_k)
        retriever.title_index = title_index
        return retriever
