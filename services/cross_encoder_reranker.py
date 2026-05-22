"""Optional cross-encoder reranker for RAG documents."""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from config import (
    CROSS_ENCODER_CANDIDATE_TOP_K,
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_MAX_LENGTH,
    CROSS_ENCODER_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved documents with a sequence-classification cross encoder."""

    def __init__(self) -> None:
        self.enabled = CROSS_ENCODER_ENABLED
        self.model_path = CROSS_ENCODER_MODEL_PATH.strip()
        self.max_length = CROSS_ENCODER_MAX_LENGTH
        self.candidate_top_k = CROSS_ENCODER_CANDIDATE_TOP_K
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.device: Optional[str] = None
        self._load_failed = False

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents or not self.enabled:
            return documents
        if not self._ensure_loaded():
            return documents

        candidate_count = min(max(1, self.candidate_top_k), len(documents))
        candidates = documents[:candidate_count]
        tail = documents[candidate_count:]
        pairs = [(query, doc.page_content or "") for doc in candidates]

        try:
            with self.torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits.detach().cpu()

            scores = []
            for row in logits:
                if row.numel() == 1:
                    scores.append(float(row.item()))
                else:
                    scores.append(float(row[-1].item()))

            scored = []
            for doc, score in zip(candidates, scores):
                metadata = dict(doc.metadata or {})
                metadata["cross_encoder_score"] = score
                doc.metadata = metadata
                scored.append(doc)

            reranked = sorted(scored, key=lambda doc: doc.metadata.get("cross_encoder_score", 0.0), reverse=True)
            for idx, doc in enumerate(reranked, start=1):
                metadata = dict(doc.metadata or {})
                metadata["cross_encoder_rank"] = idx
                doc.metadata = metadata
            logger.info("交叉编码器重排完成: candidates=%d", len(reranked))
            return reranked + tail
        except Exception as exc:
            logger.warning("交叉编码器重排失败，回退到原排序: %s", exc, exc_info=True)
            return documents

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if self.model is not None and self.tokenizer is not None:
            return True
        if self._load_failed:
            return False
        if not self.model_path:
            logger.warning("CROSS_ENCODER_ENABLED=true 但未配置 CROSS_ENCODER_MODEL_PATH，跳过交叉编码器重排")
            self._load_failed = True
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("交叉编码器加载完成: model=%s device=%s", self.model_path, self.device)
            return True
        except Exception as exc:
            logger.warning("交叉编码器加载失败，跳过重排: %s", exc, exc_info=True)
            self._load_failed = True
            return False
