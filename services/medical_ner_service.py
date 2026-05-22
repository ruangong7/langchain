"""Medical NER service backed by a local token-classification model."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast

from config import MEDICAL_NER_ENABLED, MEDICAL_NER_MODEL_PATH

logger = logging.getLogger(__name__)

POPULATION_TERMS = {"孕妇", "儿童", "老人", "老年人", "小孩", "婴儿", "哺乳期", "怀孕"}
FOOD_TERMS = {"柚子", "牛奶", "酒", "葡萄柚", "橘子", "橙子", "咖啡", "茶", "水果", "食物"}
DISEASE_HINTS = ("病", "炎", "癌", "综合征", "高血压", "糖尿病", "冠心病", "鼻炎", "胃炎")
SYMPTOM_HINTS = ("痛", "晕", "热", "烧", "咳", "吐", "泻", "痒", "肿", "慌", "乏力", "失眠", "恶心")
DRUG_HINTS = ("片", "胶囊", "颗粒", "注射液", "口服液", "滴眼液", "他汀", "阿司匹林", "布洛芬", "阿莫西林", "头孢", "华法林")
DOSE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|ml|g|片|粒|次|天|毫克|克|毫升)", re.IGNORECASE)
LABEL_PREFIXES = ("B-", "I-", "E-", "S-", "M-")


def _normalize_label_type(label_type: str) -> Optional[str]:
    text = str(label_type or "").strip().lower()
    if not text:
        return None
    if any(key in text for key in ("drug", "medication", "medicine", "pharmacologic")):
        return "drug"
    if any(key in text for key in ("symptom", "sign")):
        return "symptom"
    if any(key in text for key in ("disease", "diagnosis", "disorder", "certificate")):
        return "disease"
    if any(key in text for key in ("population", "crowd", "person", "age", "gender")):
        return "population"
    if any(key in text for key in ("food", "diet", "nutrition", "allergen")):
        return "food"
    if any(key in text for key in ("dose", "dosage", "strength", "frequency", "duration", "form")):
        return "dose"
    return None


class MedicalNERService:
    """Extracts medical entity spans and lightweight typed slots."""

    def __init__(self) -> None:
        self.enabled = MEDICAL_NER_ENABLED
        self.model_path = Path(MEDICAL_NER_MODEL_PATH)
        self._loaded = False
        self._available = False
        self._tokenizer: Optional[BertTokenizerFast] = None
        self._model: Optional[AutoModelForTokenClassification] = None
        self._id2label: Dict[int, str] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract(self, message: str) -> Dict[str, Any]:
        normalized = " ".join(str(message or "").split())
        if not normalized or not self.enabled:
            return self._empty_result()

        self._ensure_loaded()
        if not self._available or self._tokenizer is None or self._model is None:
            return self._empty_result()

        entities = self._predict_entities(normalized)
        return self._build_result(normalized, entities)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            actual_path = self._resolve_model_path()
            if actual_path is None:
                logger.warning("Medical NER 模型路径不存在: %s", self.model_path)
                return
            self._tokenizer = BertTokenizerFast.from_pretrained(str(actual_path), local_files_only=True)
            self._model = AutoModelForTokenClassification.from_pretrained(str(actual_path), local_files_only=True)
            self._model.to(self._device)
            self._model.eval()
            label2id = getattr(self._model.config, "label2id", None) or {}
            if label2id:
                self._id2label = {value: key for key, value in label2id.items()}
            else:
                id2label = getattr(self._model.config, "id2label", None) or {}
                self._id2label = {int(key): value for key, value in id2label.items()}
            self._available = True
            logger.info("Medical NER 初始化完成: %s", actual_path)
        except Exception as exc:
            logger.warning("Medical NER 初始化失败: %s", exc, exc_info=True)

    def _resolve_model_path(self) -> Optional[Path]:
        if not self.model_path.exists():
            return None
        snapshots_dir = self.model_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(item for item in snapshots_dir.iterdir() if item.is_dir())
            return snapshots[-1] if snapshots else None
        return self.model_path

    def _predict_entities(self, sentence: str) -> List[Dict[str, str]]:
        assert self._tokenizer is not None
        assert self._model is not None
        inputs = self._tokenizer(
            sentence.replace(" ", "；").replace("\t", "；"),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            predictions = outputs.logits.argmax(-1) * inputs["attention_mask"]
        return self._extract_entities_from_outputs(
            self._tokenizer,
            inputs["input_ids"].cpu(),
            predictions.cpu(),
            inputs["attention_mask"].cpu(),
            self._id2label,
        )

    def _build_result(self, sentence: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
        drug_entities: List[str] = []
        symptom_entities: List[str] = []
        disease_entities: List[str] = []
        population_entities: List[str] = []
        food_entities: List[str] = []
        dose_entities: List[str] = []
        typed_entities: List[Dict[str, str]] = []

        for item in DOSE_PATTERN.findall(sentence):
            cleaned = " ".join(item.split())
            if cleaned and cleaned not in dose_entities:
                dose_entities.append(cleaned)

        for entity in entities:
            text = str(entity.get("text") or "").strip("，。；、（）()[]【】 ")
            if len(text) < 2:
                continue
            label_type = _normalize_label_type(entity.get("label_type") or "")
            if label_type == "drug" and text not in drug_entities:
                drug_entities.append(text)
            elif label_type == "symptom" and text not in symptom_entities:
                symptom_entities.append(text)
            elif label_type == "disease" and text not in disease_entities:
                disease_entities.append(text)
            elif label_type == "population" and text not in population_entities:
                population_entities.append(text)
            elif label_type == "food" and text not in food_entities:
                food_entities.append(text)
            elif label_type == "dose" and text not in dose_entities:
                dose_entities.append(text)
            elif text in POPULATION_TERMS and text not in population_entities:
                population_entities.append(text)
            elif text in FOOD_TERMS and text not in food_entities:
                food_entities.append(text)
            elif any(hint in text for hint in DRUG_HINTS) and text not in drug_entities:
                drug_entities.append(text)
            elif any(hint in text for hint in DISEASE_HINTS) and text not in disease_entities:
                disease_entities.append(text)
            elif any(hint in text for hint in SYMPTOM_HINTS) and text not in symptom_entities:
                symptom_entities.append(text)
            typed_type = (
                "drug" if text in drug_entities else
                "symptom" if text in symptom_entities else
                "disease" if text in disease_entities else
                "population" if text in population_entities else
                "food" if text in food_entities else
                "dose" if text in dose_entities else
                None
            )
            if typed_type:
                typed_entities.append({
                    "text": text,
                    "type": typed_type,
                    "raw_label": str(entity.get("label_type") or ""),
                })

        return {
            "spans": [item.get("text", "") for item in entities],
            "entities": typed_entities,
            "drug_entities": drug_entities[:8],
            "symptom_entities": symptom_entities[:8],
            "disease_entities": disease_entities[:8],
            "population_entities": population_entities[:8],
            "food_entities": food_entities[:8],
            "dose_entities": dose_entities[:8],
        }

    @staticmethod
    def _extract_entities_from_outputs(
        tokenizer: BertTokenizerFast,
        input_ids: torch.Tensor,
        predictions: torch.Tensor,
        attention_mask: torch.Tensor,
        id2label: Dict[int, str],
    ) -> List[Dict[str, str]]:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        entities: List[Dict[str, str]] = []
        current_entity: List[str] = []
        current_label_type = ""
        collecting = False

        for token, pred, mask in zip(tokens, predictions[0], attention_mask[0]):
            if mask.item() == 0:
                break
            label = id2label.get(int(pred.item()), "O")
            if token in {"[UNK]", "[CLS]", "[SEP]", "[PAD]"}:
                current_entity = []
                current_label_type = ""
                collecting = False
                continue

            prefix, label_type = MedicalNERService._split_label(label)
            if prefix == "S":
                entity = MedicalNERService._reconstruct_entity_text([token])
                if entity:
                    entities.append({"text": entity, "label_type": label_type})
                current_entity = []
                current_label_type = ""
                collecting = False
            elif prefix == "B":
                current_entity = [token]
                current_label_type = label_type
                collecting = True
            elif prefix in {"I", "M"} and collecting:
                current_entity.append(token)
            elif prefix == "E" and collecting:
                current_entity.append(token)
                entity = MedicalNERService._reconstruct_entity_text(current_entity)
                if entity:
                    entities.append({"text": entity, "label_type": current_label_type or label_type})
                current_entity = []
                current_label_type = ""
                collecting = False
            else:
                current_entity = []
                current_label_type = ""
                collecting = False
        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in entities:
            key = (item.get("text", ""), item.get("label_type", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    @staticmethod
    def _split_label(label: str) -> Tuple[str, str]:
        raw = str(label or "").strip()
        for prefix in LABEL_PREFIXES:
            if raw.startswith(prefix):
                return prefix[0], raw[len(prefix):]
        if raw in {"B", "I", "E", "S", "M"}:
            return raw, ""
        return "O", ""

    @staticmethod
    def _reconstruct_entity_text(tokens: List[str]) -> str:
        text = ""
        for token in tokens:
            text += token[2:] if token.startswith("##") else token
        text = re.sub(r"\s+", "", text)
        return text if len(text) >= 2 else ""

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "spans": [],
            "entities": [],
            "drug_entities": [],
            "symptom_entities": [],
            "disease_entities": [],
            "population_entities": [],
            "food_entities": [],
            "dose_entities": [],
        }
