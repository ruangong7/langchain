"""Query understanding driven by a two-stage model pipeline."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
    QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK,
)
from services.light_intent_classifier import LightIntentClassifierService

logger = logging.getLogger(__name__)

STAGE2_AMBIGUOUS_PROMPT = """你是一个中文问题分流器。只输出 JSON，不要解释。
你会收到：
1. 用户原问题
2. 第一阶段路由结果

请输出：
{
  "domain": "medical_related|drug_related|general|ambiguous",
  "route": "continue|ask_user|general_answer|out_of_scope",
  "reason": "drug_query|medical_query|ambiguous_reference|general_query|clear_query",
  "intent": "interaction|contraindication|side_effect|dosage|population|drug_info|medical_query|general_query|unknown",
  "rewrite_queries": ["..."],
  "need_rewrite": true
}

要求：
- 只判断第一阶段 ambiguous 的问题，不要默认把问题解释成医疗问题
- 如果是明确普通问题（如天气、新闻、编程、数学、翻译、闲聊等），domain=general 且 route=general_answer
- 如果可以明确视为医疗/用药问题，route=continue，并给出 1 到 4 条适合检索的短 query
- 如果像在追问前文药物但缺少指代对象，route=ask_user 且 reason=ambiguous_reference
- rewrite_queries 仅在 route=continue 时填写，优先保留药名、症状、人群、剂量、食物等关键信息
- 不要臆造药名
"""

MEDICAL_REWRITE_PROMPT = """你是一个中文医疗检索查询改写器。只输出 JSON，不要解释。
你会收到用户原问题和第一阶段路由结果。

请输出：
{
  "domain": "medical_related|drug_related",
  "reason": "medical_query|drug_query|clear_query",
  "intent": "interaction|contraindication|side_effect|dosage|population|drug_info|medical_query|unknown",
  "rewrite_queries": ["..."],
  "need_rewrite": true
}

要求：
- 只为医疗/用药知识库检索改写 query，不抽取实体、不输出实体字段
- 保留原问题的关键药名、症状、人群、剂量、时间、食物等词
- 给出 1 到 4 条短 query，适合知识库检索
- 如果原问题已经很清楚，也至少把原问题作为 rewrite_queries 的一项
- 不要输出 JSON 以外的内容
"""

CONTEXT_REWRITE_PROMPT = """你是一个中文医疗对话上下文消歧器。只输出 JSON，不要解释。
你会收到最近五轮对话和当前用户问题。

请输出：
{
  "action": "rewrite|ask_user|no_change",
  "reason": "coreference_resolved|typo_or_unclear|not_enough_context|not_ambiguous",
  "rewritten_query": "...",
  "clarification": "..."
}

要求：
- 只有当当前问题存在「这个、这个东西、它、那个、一起、还能、这样吃」等指代或省略，并且最近对话能唯一确定指代对象时，action=rewrite
- rewrite 时保留原问题原文，并把指代内容追加在原问题后面，例如：这个东西能晚饭后吃吗（这个东西指阿莫西林）
- 如果最近对话不能唯一确定指代对象，action=ask_user，并在 clarification 中直接问用户要补充什么
- 如果像药名错别字、对象不清楚、多个候选都可能，action=ask_user
- 如果当前问题不需要上下文改写，action=no_change
- 不要凭空加入最近对话里没有出现过的药名、症状或疾病
"""

INTENT_REWRITE_TERMS = {
    "interaction": ("相互作用", "联合用药", "同服"),
    "contraindication": ("禁忌", "慎用", "注意事项"),
    "side_effect": ("副作用", "不良反应"),
    "dosage": ("用法用量", "剂量"),
    "population": ("特殊人群用药",),
    "drug_info": ("说明书", "作用", "适应症"),
    "medical_query": ("症状", "处理"),
}
VAGUE_MEDICAL_REFERENCE_CUES = (
    "能吃吗", "可以吃吗", "还能吃", "怎么吃", "一起吃", "同服", "服用", "用吗",
)


class QueryUnderstandingService:
    """Two-stage query understanding: stage1 routing, stage2 medical resolution."""

    def __init__(self):
        self.light_intent_classifier = LightIntentClassifierService()
        self.llm = self._init_llm()
        self.light_intent_classifier.warmup()

    def normalize_query(self, message: str) -> str:
        return " ".join(str(message or "").strip().split())

    def _init_llm(self) -> Optional[ChatOpenAI]:
        if not QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK or not DASHSCOPE_API_KEY:
            return None
        try:
            return ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL,
                temperature=0,
                streaming=False,
            )
        except Exception as exc:
            logger.warning("查询理解 LLM 初始化失败: %s", exc, exc_info=True)
            return None

    def _parse_jsonish(self, raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def _call_llm_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            return {}
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            return self._parse_jsonish(response.content if hasattr(response, "content") else str(response))
        except Exception as exc:
            logger.warning("查询理解 LLM 调用失败: %s", exc, exc_info=True)
            return {}

    def resolve_with_history(self, message: str, history: List[Dict[str, str]]) -> Dict[str, str]:
        normalized = self.normalize_query(message)
        parsed = self._call_llm_json(
            CONTEXT_REWRITE_PROMPT,
            {
                "query": normalized,
                "history": history[-10:],
            },
        )
        action = str(parsed.get("action") or "").strip()
        if action not in {"rewrite", "ask_user", "no_change"}:
            return {
                "action": "ask_user",
                "reason": "not_enough_context",
                "rewritten_query": "",
                "clarification": "这里的指代不够明确。请补充你说的是哪种药或哪一个问题，我再继续查。",
            }
        return {
            "action": action,
            "reason": str(parsed.get("reason") or "").strip(),
            "rewritten_query": str(parsed.get("rewritten_query") or "").strip(),
            "clarification": str(parsed.get("clarification") or "").strip(),
        }

    def _stage1_route(self, normalized: str) -> Dict[str, Any]:
        try:
            routed = self.light_intent_classifier.classify(normalized)
            if routed:
                return routed
        except Exception as exc:
            logger.warning("轻量意图分类器调用失败: %s", exc, exc_info=True)
        return {
            "domain": "ambiguous",
            "route": "stage2_review",
            "medical_score": None,
            "confidence": 0.0,
            "needs_review": True,
            "intent": "unknown",
            "reason": "classifier_unavailable",
        }

    def _resolve_ambiguous_stage(
        self,
        normalized: str,
        stage1: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "query": normalized,
            "stage1": stage1,
        }
        parsed = self._call_llm_json(STAGE2_AMBIGUOUS_PROMPT, payload)
        if parsed:
            normalized_result = self._normalize_stage2_result(normalized, parsed)
            if normalized_result:
                logger.info(
                    "查询理解二阶段LLM分流: route=%s domain=%s intent=%s rewrite_queries=%s",
                    normalized_result.get("route"),
                    normalized_result.get("domain"),
                    normalized_result.get("intent"),
                    normalized_result.get("rewrite_queries"),
                )
                return normalized_result
        result = self._fallback_ambiguous_stage(normalized, stage1)
        logger.info(
            "查询理解二阶段fallback分流: route=%s domain=%s intent=%s rewrite_queries=%s",
            result.get("route"),
            result.get("domain"),
            result.get("intent"),
            result.get("rewrite_queries"),
        )
        return result

    def _normalize_stage2_result(self, normalized: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        domain = str(parsed.get("domain") or "").strip()
        route = str(parsed.get("route") or "").strip()
        if domain not in {"medical_related", "drug_related", "general", "ambiguous"}:
            return None
        if route not in {"continue", "ask_user", "general_answer", "out_of_scope"}:
            return None
        if route == "general_answer":
            return self._general_result(normalized, {"reason": str(parsed.get("reason") or "general_query").strip()})

        result = {
            "domain": domain,
            "route": route,
            "reason": str(parsed.get("reason") or "clear_query").strip(),
            "normalized_query": normalized,
            "intent": str(parsed.get("intent") or "unknown").strip(),
            "drug_entities": [],
            "symptom_entities": [],
            "disease_entities": [],
            "population_entities": [],
            "food_entities": [],
            "dose_entities": [],
        }
        rewrite_queries = self._clean_list(parsed.get("rewrite_queries"))
        result["need_rewrite"] = route == "continue" and bool(parsed.get("need_rewrite", bool(rewrite_queries)))
        result["constraints"] = self._extract_constraints(normalized)
        if result["need_rewrite"]:
            result["rewrite_queries"] = rewrite_queries or self._build_rewrite_queries_from_slots(result)
        else:
            result["rewrite_queries"] = []
        return result

    def _fallback_ambiguous_stage(
        self,
        normalized: str,
        stage1: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._looks_like_vague_medical_reference(normalized):
            return self._result(
                route="ask_user",
                reason="ambiguous_reference",
                normalized_query=normalized,
                domain="ambiguous",
                intent="unknown",
            )
        return self._general_result(normalized, {"reason": "ambiguous_general_fallback"})

    def _build_medical_stage_result(
        self,
        normalized: str,
        stage1: Dict[str, Any],
    ) -> Dict[str, Any]:
        parsed = self._call_llm_json(
            MEDICAL_REWRITE_PROMPT,
            {
                "query": normalized,
                "stage1": stage1,
            },
        )
        domain = str(parsed.get("domain") or "medical_related").strip()
        if domain not in {"medical_related", "drug_related"}:
            domain = "medical_related"
        intent = str(parsed.get("intent") or "medical_query").strip()
        rewrite_queries = self._clean_list(parsed.get("rewrite_queries"))
        result = self._result(
            route="continue",
            reason=str(parsed.get("reason") or "medical_query").strip(),
            normalized_query=normalized,
            domain=domain,
            intent=intent,
        )
        result["rewrite_queries"] = rewrite_queries or [normalized]
        result["need_rewrite"] = True
        logger.info(
            "医疗查询LLM改写: domain=%s intent=%s rewrite_queries=%s",
            result.get("domain"),
            result.get("intent"),
            result.get("rewrite_queries"),
        )
        return result

    @staticmethod
    def _looks_like_vague_medical_reference(normalized: str) -> bool:
        compact = re.sub(r"\s+", "", normalized)
        return any(term in compact for term in VAGUE_MEDICAL_REFERENCE_CUES)

    def analyze(self, message: str) -> Dict[str, Any]:
        normalized = self.normalize_query(message)
        stage1 = self._stage1_route(normalized)
        logger.info(
            "意图识别第一阶段: query=%s domain=%s route=%s intent=%s reason=%s medical_score=%s confidence=%s needs_review=%s probabilities=%s",
            normalized,
            stage1.get("domain"),
            stage1.get("route"),
            stage1.get("intent"),
            stage1.get("reason"),
            stage1.get("medical_score"),
            stage1.get("confidence"),
            stage1.get("needs_review", False),
            stage1.get("probabilities"),
        )
        if stage1.get("route") == "general_answer":
            result = self._general_result(normalized, stage1)
            logger.info(
                "查询理解最终结果: query=%s route=%s domain=%s intent=%s rewrite_queries=%s",
                normalized,
                result.get("route"),
                result.get("domain"),
                result.get("intent"),
                result.get("rewrite_queries"),
            )
            return result

        if stage1.get("route") == "medical_rewrite":
            result = self._build_medical_stage_result(normalized, stage1)
        else:
            result = self._resolve_ambiguous_stage(normalized, stage1)
        logger.info(
            "查询理解最终结果: query=%s route=%s domain=%s intent=%s rewrite_queries=%s",
            normalized,
            result.get("route"),
            result.get("domain"),
            result.get("intent"),
            result.get("rewrite_queries"),
        )
        return result

    def build_retrieval_queries(self, analysis: Dict[str, Any], fallback: str = "") -> List[str]:
        normalized = analysis.get("normalized_query") or self.normalize_query(fallback)
        if analysis.get("route") != "continue":
            return [normalized] if normalized else []
        rewrite_queries = analysis.get("rewrite_queries") or []
        return rewrite_queries if rewrite_queries else ([normalized] if normalized else [])

    def _general_result(self, normalized_query: str, stage1: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain": "general",
            "route": "general_answer",
            "reason": stage1.get("reason", "general_query"),
            "normalized_query": normalized_query,
            "intent": "general_query",
            "drug_entities": [],
            "symptom_entities": [],
            "disease_entities": [],
            "population_entities": [],
            "food_entities": [],
            "dose_entities": [],
            "constraints": [],
            "need_rewrite": False,
            "rewrite_queries": [],
        }

    def _result(
        self,
        route: str,
        reason: str,
        normalized_query: str,
        domain: str,
        intent: str,
        drug_entities: Optional[List[str]] = None,
        symptom_entities: Optional[List[str]] = None,
        disease_entities: Optional[List[str]] = None,
        population_entities: Optional[List[str]] = None,
        food_entities: Optional[List[str]] = None,
        dose_entities: Optional[List[str]] = None,
        unknown_mentions: Optional[List[str]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        result = {
            "domain": domain,
            "route": route,
            "reason": reason,
            "normalized_query": normalized_query,
            "intent": intent,
            "drug_entities": self._clean_list(drug_entities),
            "symptom_entities": self._clean_list(symptom_entities),
            "disease_entities": self._clean_list(disease_entities),
            "population_entities": self._clean_list(population_entities),
            "food_entities": self._clean_list(food_entities),
            "dose_entities": self._clean_list(dose_entities),
            "constraints": self._extract_constraints(normalized_query),
        }
        if unknown_mentions:
            result["unknown_mentions"] = self._clean_list(unknown_mentions)
        if candidates:
            result["drug_candidates"] = candidates

        can_rewrite = route == "continue" and domain in {"drug_related", "medical_related"}
        result["need_rewrite"] = can_rewrite
        result["rewrite_queries"] = self._build_rewrite_queries_from_slots(result) if can_rewrite else []
        return result

    def _build_rewrite_queries_from_slots(self, analysis: Dict[str, Any]) -> List[str]:
        normalized = analysis.get("normalized_query", "")
        intent = str(analysis.get("intent") or "unknown")
        drug_entities = analysis.get("drug_entities") or []
        symptom_entities = analysis.get("symptom_entities") or []
        disease_entities = analysis.get("disease_entities") or []
        population_entities = analysis.get("population_entities") or []
        food_entities = analysis.get("food_entities") or []
        dose_entities = analysis.get("dose_entities") or []
        constraints = analysis.get("constraints") or []

        if drug_entities:
            names = " ".join(drug_entities)
            extra = " ".join(self._merge_unique(population_entities, food_entities, dose_entities, [item.get("value", "") for item in constraints]))
            intent_terms = list(INTENT_REWRITE_TERMS.get(intent, ("说明书", "作用")))
            queries = [
                " ".join([names, extra, intent_terms[0]]).strip(),
                " ".join([names, extra, " ".join(intent_terms[1:])]).strip() if len(intent_terms) > 1 else "",
                " ".join([names, extra]).strip(),
            ]
            if len(drug_entities) >= 2:
                queries.append(" ".join([names, extra, "配伍禁忌"]).strip())
            return self._dedupe_queries(queries, normalized)

        entity_text = " ".join(self._merge_unique(symptom_entities, disease_entities, population_entities))
        if entity_text:
            terms = list(INTENT_REWRITE_TERMS.get("medical_query", ("症状", "处理")))
            queries = [
                " ".join([entity_text, terms[0]]).strip(),
                " ".join([entity_text, terms[-1]]).strip(),
                entity_text,
            ]
            return self._dedupe_queries(queries, normalized)

        return [normalized] if normalized else []

    @staticmethod
    def _extract_constraints(message: str) -> List[Dict[str, str]]:
        compact = re.sub(r"\s+", "", message)
        constraints = []
        for term in ("孕妇", "怀孕", "哺乳", "儿童", "小孩", "老人", "老年人", "肝功能", "肾功能", "肝肾", "过敏", "高血压", "糖尿病", "饭前", "饭后", "空腹"):
            if term in compact:
                kind = "dose" if term in {"饭前", "饭后", "空腹"} else "population"
                constraints.append({"type": kind, "value": term})
        for item in re.findall(r"\d+(?:\.\d+)?\s*(?:mg|ml|g|片|粒|次|天|毫克|克|毫升)", compact, flags=re.IGNORECASE):
            constraints.append({"type": "dose", "value": item})
        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in constraints:
            key = (item["type"], item["value"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped[:8]

    @staticmethod
    def _merge_unique(*groups: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for group in groups:
            for item in group or []:
                text = str(item or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    merged.append(text)
        return merged

    @staticmethod
    def _clean_list(values: Optional[List[Any]]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        return cleaned[:8]

    @staticmethod
    def _dedupe_queries(queries: List[str], normalized: str) -> List[str]:
        seen = set()
        deduped = []
        for query in queries + [normalized]:
            compact = " ".join(str(query or "").split())
            if compact and compact not in seen:
                seen.add(compact)
                deduped.append(compact)
        return deduped[:4]
