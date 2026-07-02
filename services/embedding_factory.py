from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

from langchain_core.embeddings import Embeddings

from config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_BATCH_SIZE,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_DOCUMENT_PREFIX,
    LOCAL_EMBEDDING_MAX_LENGTH,
    LOCAL_EMBEDDING_MODEL_PATH,
    LOCAL_EMBEDDING_NORMALIZE,
    LOCAL_EMBEDDING_QUERY_PREFIX,
)

logger = logging.getLogger(__name__)


class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings wrapper."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from dashscope import TextEmbedding

        if not texts:
            return []
        max_batch = 10
        all_vectors: List[List[float]] = []
        for index in range(0, len(texts), max_batch):
            batch = texts[index : index + max_batch]
            response = TextEmbedding.call(model=self.model, input=batch)
            if not response or not getattr(response, "output", None) or "embeddings" not in response.output:
                raise ValueError(
                    f"DashScope TextEmbedding returned invalid response: model={self.model}, batch_size={len(batch)}"
                )
            all_vectors.extend([item["embedding"] for item in response.output["embeddings"]])
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LocalBGEEmbeddings(Embeddings):
    """Local HuggingFace embedding wrapper for BGE-style models."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        if not str(model_path).strip():
            raise ValueError("LOCAL_EMBEDDING_MODEL_PATH is required for local embeddings.")

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.normalize = bool(normalize)
        self.query_prefix = str(query_prefix or "")
        self.document_prefix = str(document_prefix or "")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Local BGE embeddings require torch and transformers in the current Python environment."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.pooling_mode = self._load_pooling_mode()

        logger.info(
            "本地 Embedding 已加载: model_path=%s device=%s pooling=%s batch_size=%s max_length=%s",
            self.model_path,
            self.device,
            self.pooling_mode,
            self.batch_size,
            self.max_length,
        )

    def _resolve_device(self, raw_device: str) -> str:
        device = str(raw_device or "auto").strip().lower()
        if device and device != "auto":
            return device

        if self._torch.cuda.is_available():
            return "cuda"
        mps = getattr(self._torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_pooling_mode(self) -> str:
        config_path = Path(self.model_path) / "1_Pooling" / "config.json"
        if not config_path.is_file():
            return "cls"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return "cls"
        if payload.get("pooling_mode_mean_tokens"):
            return "mean"
        return "cls"

    def _apply_prefix(self, texts: List[str], prefix: str) -> List[str]:
        clean_prefix = str(prefix or "")
        if not clean_prefix:
            return [str(text or "") for text in texts]
        return [clean_prefix + str(text or "") for text in texts]

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str], *, prefix: str) -> List[List[float]]:
        if not texts:
            return []

        prepared = self._apply_prefix(texts, prefix)
        vectors: List[List[float]] = []

        with self._torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                if self.pooling_mode == "mean":
                    pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                else:
                    pooled = outputs.last_hidden_state[:, 0]
                if self.normalize:
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, prefix=self.document_prefix)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], prefix=self.query_prefix)[0]


def build_embeddings() -> Embeddings:
    backend = str(EMBEDDING_BACKEND or "dashscope").strip().lower()
    if backend == "local":
        return LocalBGEEmbeddings(
            model_path=LOCAL_EMBEDDING_MODEL_PATH,
            device=LOCAL_EMBEDDING_DEVICE,
            batch_size=LOCAL_EMBEDDING_BATCH_SIZE,
            max_length=LOCAL_EMBEDDING_MAX_LENGTH,
            normalize=LOCAL_EMBEDDING_NORMALIZE,
            query_prefix=LOCAL_EMBEDDING_QUERY_PREFIX,
            document_prefix=LOCAL_EMBEDDING_DOCUMENT_PREFIX,
        )

    import dashscope

    dashscope.api_key = DASHSCOPE_API_KEY
    return DashScopeEmbeddings(model=EMBEDDING_MODEL)


def embedding_backend_summary() -> dict[str, Any]:
    backend = str(EMBEDDING_BACKEND or "dashscope").strip().lower()
    summary: dict[str, Any] = {
        "backend": backend,
        "embedding_model": EMBEDDING_MODEL,
    }
    if backend == "local":
        summary.update(
            {
                "model_path": LOCAL_EMBEDDING_MODEL_PATH,
                "device": LOCAL_EMBEDDING_DEVICE,
                "batch_size": LOCAL_EMBEDDING_BATCH_SIZE,
                "max_length": LOCAL_EMBEDDING_MAX_LENGTH,
                "normalize": LOCAL_EMBEDDING_NORMALIZE,
            }
        )
    return summary
