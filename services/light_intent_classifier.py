"""Local stage-1 intent classifier backed by a fine-tuned sequence classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore[assignment]

from config import (
    LIGHT_INTENT_MODEL_ENABLED,
    LIGHT_INTENT_MODEL_MAX_LENGTH,
    LIGHT_INTENT_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class LightIntentClassifierService:
    """Classifies a query as medical, general, or ambiguous."""

    def __init__(self) -> None:
        self.enabled = LIGHT_INTENT_MODEL_ENABLED
        self.model_path = Path(LIGHT_INTENT_MODEL_PATH)
        self.max_length = LIGHT_INTENT_MODEL_MAX_LENGTH
        self._loaded = False
        self._available = False
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None

    def classify(self, message: str) -> Dict[str, Any]:
        normalized = " ".join(str(message or "").split())
        if not normalized:
            return self._route("ambiguous", 0.5, "empty_query")

        if torch is None:
            return self._route("ambiguous", 0.5, "torch_unavailable")
        self._ensure_loaded()
        if not self.enabled or not self._available or self._tokenizer is None or self._model is None:
            return self._route("ambiguous", 0.5, "model_unavailable")

        try:
            label, confidence = self._predict_label(normalized)
        except Exception as exc:
            logger.warning("Light intent 推理失败: %s", exc, exc_info=True)
            return self._route("ambiguous", 0.5, "model_error")

        return self._route(label, confidence, "sequence_classifier")

    def warmup(self) -> bool:
        """Load the local model during application startup."""
        self._ensure_loaded()
        return self._available

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled or torch is None:
            return
        try:
            actual_path = self._resolve_model_path()
            if actual_path is None:
                logger.warning("Light intent 分类模型路径不存在: %s", self.model_path)
                return
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(actual_path), local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(str(actual_path), local_files_only=True)
            self._model.to(self._device)
            self._model.eval()
            self._available = True
            logger.info("Light intent 分类模型初始化完成: %s", actual_path)
        except Exception as exc:
            logger.warning("Light intent 分类模型初始化失败: %s", exc, exc_info=True)

    def _resolve_model_path(self) -> Optional[Path]:
        if not self.model_path.exists():
            return None
        if (self.model_path / "config.json").exists():
            return self.model_path
        snapshots_dir = self.model_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(item for item in snapshots_dir.iterdir() if item.is_dir())
            return snapshots[-1] if snapshots else None
        return None

    def _predict_label(self, text: str) -> tuple[str, float]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu()
        label_id = int(probs.argmax().item())
        confidence = float(probs[label_id].item())
        id2label = getattr(self._model.config, "id2label", {}) or {}
        label = str(id2label.get(label_id, id2label.get(str(label_id), "ambiguous"))).lower()
        if label not in {"medical", "general", "ambiguous"}:
            label = "ambiguous"
        return label, confidence

    @staticmethod
    def _route(label: str, confidence: float, reason: str) -> Dict[str, Any]:
        confidence = max(0.0, min(1.0, float(confidence)))
        if label == "medical":
            return {
                "domain": "medical",
                "route": "continue",
                "medical_score": round(confidence, 4),
                "confidence": round(confidence, 4),
                "intent": "unknown",
                "reason": reason,
            }
        if label == "general":
            return {
                "domain": "general",
                "route": "general_answer",
                "medical_score": round(1.0 - confidence, 4),
                "confidence": round(confidence, 4),
                "intent": "general_query",
                "reason": reason,
            }
        return {
            "domain": "ambiguous",
            "route": "stage2_review",
            "medical_score": None,
            "confidence": round(confidence, 4),
            "needs_review": True,
            "intent": "unknown",
            "reason": reason,
        }
