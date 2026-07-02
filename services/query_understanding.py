"""Query understanding driven by a two-stage model pipeline."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
    QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK,
)
from services.drug_lexicon import DrugLexiconService
from services.light_intent_classifier import LightIntentClassifierService
from services.medical_ner_service import MedicalNERService

logger = logging.getLogger(__name__)

AMBIGUOUS_TOOL_MAX_ROUNDS = 2
AMBIGUOUS_TOOL_NAMES = {"queryUserHealthProfile", "queryUserMedicationSummary"}

STAGE2_AMBIGUOUS_PROMPT = """你是一个中文问题分流器。你可以先调用少量工具，再输出最终 JSON；也可以不调用工具。
你会收到：
1. 用户原问题
2. 第一阶段路由结果
3. 长期记忆摘要
4. 最近几轮对话
5. 当前是否登录
6. 当前可用工具列表
7. 用户档案是否可用

如果你要调用工具，请只调用当前 available_tools 列表中真的存在的工具。
最多调用 2 轮工具。工具调用结束后，必须输出最终 JSON，不要解释，不要 Markdown。

最终输出格式：
{
  "domain": "medical_related|drug_related|general|ambiguous",
  "route": "continue|ask_user|general_answer",
  "reason": "drug_query|medical_query|ambiguous_reference|general_query|clear_query",
  "intent": "interaction|contraindication|side_effect|dosage|population|drug_info|medical_query|general_query|unknown",
  "resolved_query": "...",
  "clarification": "...",
  "rewrite_queries": ["..."],
  "need_rewrite": true,
  "need_tool": true,
  "tool_candidates": ["queryUserMedicationSummary"],
  "skip_retrieval": false
}

要求：
- 只判断第一阶段 ambiguous 的问题，不要默认把问题解释成医疗问题
- 如果用户是在问对话历史本身，例如“我上句话说了什么”“你刚才说了什么”，并且最近对话足以回答，route=general_answer
- 如果是明确普通问题（如天气、新闻、编程、数学、翻译、闲聊等），domain=general 且 route=general_answer
- 如果可以明确视为医疗/用药问题，route=continue，并给出 1 到 4 条适合检索的短 query
- 如果当前问题是在问用户自己的健康档案或当前用药，并且当前已登录且对应工具可用，route=continue；同时设置 need_tool=true，并在 tool_candidates 中给出合适工具名
- `queryUserHealthProfile` 只用于读取用户个人档案，不包含当前用药
- `queryUserMedicationSummary` 只用于读取当前登记的用药摘要
- 如果工具结果本身已经足够支撑最终回答，例如“我现在在吃什么药”“我的基础病是什么”，可设置 skip_retrieval=true；这样后续回答阶段会直接看到工具结果，不再走知识库检索
- 如果像在追问前文药物但缺少指代对象，route=ask_user 且 reason=ambiguous_reference
- 如果 route=ask_user，clarification 里直接给出追问句子
- 如果 route=continue 且需要结合上下文消歧，可在 resolved_query 中给出补全后的问题；rewrite_queries 优先基于 resolved_query 生成
- rewrite_queries 仅在 route=continue 时填写，优先保留药名、症状、人群、剂量、食物等关键信息
- 不要臆造药名
"""

MEDICAL_REWRITE_PROMPT = """你是一个中文医疗检索查询改写器。只输出 JSON，不要解释。
你会收到用户原问题、第一阶段路由结果、用户个人档案上下文、当前用药摘要。

请输出：
{
  "domain": "medical_related|drug_related",
  "reason": "medical_query|drug_query|clear_query",
  "intent": "interaction|contraindication|side_effect|dosage|population|drug_info|medical_query|unknown",
  "personalized": true,
  "rewrite_queries": ["..."],
  "need_rewrite": true
}

要求：
- 只为医疗/用药知识库检索改写 query，不抽取实体、不输出实体字段
- 如果问题明显与“用户本人能不能吃/能不能用/当前是否适合”有关，请结合用户个人档案和当前用药摘要进行个体化改写，并设置 personalized=true
- 如果当前用药与原问题无关，不要强行把所有用药都塞进 query，只保留和判断最相关的 1 到 3 个药物/风险点
- 保留原问题的关键药名、症状、人群、剂量、时间、食物等词
- 给出 1 到 4 条短 query，适合知识库检索
- 如果原问题已经很清楚，也至少把原问题作为 rewrite_queries 的一项
- 不要输出 JSON 以外的内容
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

    def __init__(self, database_tool: Any | None = None):
        self.light_intent_classifier = LightIntentClassifierService()
        self.llm = self._init_llm()
        self.medical_ner = MedicalNERService()
        self.database_tool = database_tool
        self.drug_lexicon = DrugLexiconService(database_tool)
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
        history: Optional[List[Dict[str, str]]] = None,
        memory_summary: str = "",
        user_logged_in: bool = False,
        available_tools: Optional[List[str]] = None,
        profile_available: bool = False,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "query": normalized,
            "stage1": stage1,
            "memory_summary": str(memory_summary or "").strip(),
            "history": list(history or [])[-10:],
            "user_logged_in": bool(user_logged_in),
            "available_tools": self._clean_list(available_tools),
            "profile_available": bool(profile_available),
        }
        parsed, tool_observations = self._resolve_ambiguous_with_tools(
            payload,
            user_id=user_id,
            available_tools=available_tools,
        )
        if parsed:
            normalized_result = self._normalize_stage2_result(normalized, parsed)
            if normalized_result:
                normalized_result = self._apply_prefetched_tool_context(normalized_result, tool_observations)
                logger.info(
                    "查询理解二阶段LLM分流: route=%s domain=%s intent=%s rewrite_queries=%s tool_candidates=%s skip_retrieval=%s",
                    normalized_result.get("route"),
                    normalized_result.get("domain"),
                    normalized_result.get("intent"),
                    normalized_result.get("rewrite_queries"),
                    normalized_result.get("tool_candidates"),
                    normalized_result.get("skip_retrieval", False),
                )
                return normalized_result
        result = self._fallback_ambiguous_stage(normalized, stage1)
        logger.info(
            "查询理解二阶段fallback分流: route=%s domain=%s intent=%s rewrite_queries=%s tool_candidates=%s",
            result.get("route"),
            result.get("domain"),
            result.get("intent"),
            result.get("rewrite_queries"),
            result.get("tool_candidates"),
        )
        return result

    def _resolve_ambiguous_with_tools(
        self,
        payload: Dict[str, Any],
        *,
        user_id: Optional[int],
        available_tools: Optional[List[str]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.llm is None:
            return {}, []

        selected_tool_names = [
            name for name in self._clean_list(available_tools)
            if name in AMBIGUOUS_TOOL_NAMES
        ]
        selected_tool_schemas = self._build_ambiguous_tool_schemas(selected_tool_names)
        messages = [
            SystemMessage(content=STAGE2_AMBIGUOUS_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        tool_observations: List[Dict[str, Any]] = []

        if not selected_tool_schemas:
            return self._call_llm_json(STAGE2_AMBIGUOUS_PROMPT, payload), tool_observations

        try:
            for _ in range(AMBIGUOUS_TOOL_MAX_ROUNDS):
                response = self.llm.bind_tools(selected_tool_schemas).invoke(messages)
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    return self._parse_jsonish(response.content if hasattr(response, "content") else str(response)), tool_observations

                messages.append(response)
                for tool_call in tool_calls:
                    tool_message, metadata = self._run_ambiguous_tool_call(tool_call, user_id=user_id)
                    tool_observations.append(metadata)
                    messages.append(tool_message)

            messages.append(SystemMessage(content="你已经完成最多 2 轮工具调用。现在必须直接输出最终 JSON，不能再调用工具。"))
            final_response = self.llm.invoke(messages)
            return self._parse_jsonish(final_response.content if hasattr(final_response, "content") else str(final_response)), tool_observations
        except Exception as exc:
            logger.warning("查询理解二阶段工具闭环失败: %s", exc, exc_info=True)
            return self._call_llm_json(STAGE2_AMBIGUOUS_PROMPT, payload), tool_observations

    @staticmethod
    def _build_ambiguous_tool_schemas(tool_names: List[str]) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        if "queryUserHealthProfile" in tool_names:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "queryUserHealthProfile",
                        "description": "读取当前登录用户的个人健康档案，不包含当前用药。",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
        if "queryUserMedicationSummary" in tool_names:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "queryUserMedicationSummary",
                        "description": "读取当前登录用户登记的当前用药摘要。",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
        return schemas

    def _run_ambiguous_tool_call(
        self,
        tool_call: Dict[str, Any],
        *,
        user_id: Optional[int],
    ) -> tuple[ToolMessage, Dict[str, Any]]:
        name = str(tool_call.get("name") or "").strip()
        call_id = str(tool_call.get("id") or name or "tool_call")

        if user_id is None or self.database_tool is None:
            message = "当前没有可用的登录用户信息，无法执行该工具。"
            return ToolMessage(content=message, tool_call_id=call_id), {
                "name": name,
                "ok": False,
                "reason": "missing_user_context",
                "message": message,
                "rendered": message,
            }

        try:
            if name == "queryUserHealthProfile":
                payload = self.database_tool.query_user_health_profile_payload(int(user_id))
            elif name == "queryUserMedicationSummary":
                payload = self.database_tool.query_user_medication_summary_payload(int(user_id))
            else:
                message = f"工具 {name} 当前不可用。"
                return ToolMessage(content=message, tool_call_id=call_id), {
                    "name": name,
                    "ok": False,
                    "reason": "tool_unavailable",
                    "message": message,
                    "rendered": message,
                }

            formatter = getattr(self.database_tool, "format_tool_result", None)
            rendered = str(formatter(payload) if callable(formatter) else payload.get("message") or "").strip()
            rendered = rendered or str(payload.get("message") or "").strip() or "工具未返回可读结果。"
            metadata = {
                "name": name,
                "ok": bool(payload.get("ok")),
                "reason": str(payload.get("reason") or ""),
                "message": str(payload.get("message") or rendered),
                "rendered": rendered,
                "count": int(payload.get("count") or 0),
            }
            return ToolMessage(content=rendered[:6000], tool_call_id=call_id), metadata
        except Exception as exc:
            message = f"工具 {name} 调用失败：{exc}"
            logger.warning("查询理解工具执行失败: tool=%s error=%s", name, exc, exc_info=True)
            return ToolMessage(content=message, tool_call_id=call_id), {
                "name": name,
                "ok": False,
                "reason": "exception",
                "message": message,
                "rendered": message,
            }

    def _normalize_stage2_result(self, normalized: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        domain = str(parsed.get("domain") or "").strip()
        route = str(parsed.get("route") or "").strip()
        if domain not in {"medical_related", "drug_related", "general", "ambiguous"}:
            return None
        if route not in {"continue", "ask_user", "general_answer"}:
            return None
        if route == "general_answer":
            return self._general_result(normalized, {"reason": str(parsed.get("reason") or "general_query").strip()})

        result = {
            "domain": domain,
            "route": route,
            "reason": str(parsed.get("reason") or "clear_query").strip(),
            "normalized_query": normalized,
            "intent": str(parsed.get("intent") or "unknown").strip(),
            "resolved_query": str(parsed.get("resolved_query") or "").strip(),
            "clarification": str(parsed.get("clarification") or "").strip(),
            "need_tool": bool(parsed.get("need_tool")),
            "tool_candidates": self._clean_list(parsed.get("tool_candidates")),
            "skip_retrieval": route == "continue" and bool(parsed.get("skip_retrieval")),
            "drug_entities": [],
            "symptom_entities": [],
            "disease_entities": [],
            "population_entities": [],
            "food_entities": [],
            "dose_entities": [],
        }
        rewrite_queries = self._clean_list(parsed.get("rewrite_queries"))
        if route == "continue" and result["resolved_query"]:
            resolved_query = result["resolved_query"]
            if resolved_query != normalized:
                result["context_resolved"] = True
        result["need_rewrite"] = route == "continue" and bool(parsed.get("need_rewrite", bool(rewrite_queries)))
        result["constraints"] = self._extract_constraints(normalized)
        if result["need_rewrite"]:
            result["rewrite_queries"] = rewrite_queries or self._build_rewrite_queries_from_slots(result)
        else:
            result["rewrite_queries"] = []
        return result

    def _apply_prefetched_tool_context(
        self,
        result: Dict[str, Any],
        tool_observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if result.get("route") != "continue" or not tool_observations:
            return result

        if not result.get("tool_candidates"):
            result["tool_candidates"] = self._clean_list([item.get("name") for item in tool_observations])

        if not result.get("skip_retrieval"):
            return result

        rendered_blocks = []
        seen = set()
        for item in tool_observations:
            text = str(item.get("rendered") or item.get("message") or "").strip()
            if text and text not in seen:
                seen.add(text)
                rendered_blocks.append(text)
        if rendered_blocks:
            result["prefetched_context"] = "\n\n".join(rendered_blocks)
            result["prefetched_tools"] = self._clean_list([item.get("name") for item in tool_observations])
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
        *,
        personal_profile_context: str = "",
        medication_summary_context: str = "",
    ) -> Dict[str, Any]:
        parsed = self._call_llm_json(
            MEDICAL_REWRITE_PROMPT,
            {
                "query": normalized,
                "stage1": stage1,
                "personal_profile_context": str(personal_profile_context or "").strip(),
                "medication_summary_context": str(medication_summary_context or "").strip(),
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
        result["personalized"] = bool(parsed.get("personalized"))
        if medication_summary_context:
            result["prefetched_medication_context"] = str(medication_summary_context).strip()
            result["prefetched_tools"] = ["queryUserMedicationSummary"]
        logger.info(
            "continue链路LLM改写: domain=%s intent=%s personalized=%s rewrite_queries=%s",
            result.get("domain"),
            result.get("intent"),
            result.get("personalized", False),
            result.get("rewrite_queries"),
        )
        return result

    @staticmethod
    def _looks_like_vague_medical_reference(normalized: str) -> bool:
        compact = re.sub(r"\s+", "", normalized)
        return any(term in compact for term in VAGUE_MEDICAL_REFERENCE_CUES)

    def analyze(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        memory_summary: str = "",
        user_logged_in: bool = False,
        available_tools: Optional[List[str]] = None,
        profile_available: bool = False,
        user_id: Optional[int] = None,
        personal_profile_context: str = "",
    ) -> Dict[str, Any]:
        normalized = self.normalize_query(message)
        extracted = self._extract_entities(normalized)
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
            result = self._apply_extracted_entities(self._general_result(normalized, stage1), extracted)
            logger.info(
                "查询理解最终结果: query=%s route=%s domain=%s intent=%s rewrite_queries=%s",
                normalized,
                result.get("route"),
                result.get("domain"),
                result.get("intent"),
                result.get("rewrite_queries"),
            )
            return result

        if stage1.get("route") == "continue" and stage1.get("domain") == "medical":
            medication_summary_context = self._prefetch_medication_summary_context(
                user_id=user_id,
                available_tools=available_tools,
            )
            result = self._build_medical_stage_result(
                normalized,
                stage1,
                personal_profile_context=personal_profile_context,
                medication_summary_context=medication_summary_context,
            )
        else:
            result = self._resolve_ambiguous_stage(
                normalized,
                stage1,
                history=history,
                memory_summary=memory_summary,
                user_logged_in=user_logged_in,
                available_tools=available_tools,
                profile_available=profile_available,
                user_id=user_id,
            )
        result = self._apply_extracted_entities(result, extracted)
        logger.info(
            "查询理解最终结果: query=%s route=%s domain=%s intent=%s rewrite_queries=%s",
            normalized,
            result.get("route"),
            result.get("domain"),
            result.get("intent"),
            result.get("rewrite_queries"),
        )
        return result

    def _prefetch_medication_summary_context(
        self,
        *,
        user_id: Optional[int],
        available_tools: Optional[List[str]],
    ) -> str:
        if (
            user_id is None
            or self.database_tool is None
            or "queryUserMedicationSummary" not in self._clean_list(available_tools)
        ):
            return ""
        try:
            payload = self.database_tool.query_user_medication_summary_payload(int(user_id))
            formatter = getattr(self.database_tool, "format_tool_result", None)
            rendered = str(formatter(payload) if callable(formatter) else payload.get("message") or "").strip()
            if not rendered:
                rendered = str(payload.get("message") or "").strip()
            logger.info(
                "continue链路已预取当前用药摘要: user_id=%s count=%s ok=%s",
                user_id,
                payload.get("count"),
                payload.get("ok"),
            )
            return rendered
        except Exception as exc:
            logger.warning("预取当前用药摘要失败: user_id=%s error=%s", user_id, exc, exc_info=True)
            return ""

    def _extract_entities(self, normalized: str) -> Dict[str, Any]:
        ner_result = self.medical_ner.extract(normalized)
        alias_matches = self.drug_lexicon.match_mentions(normalized)

        canonical_drugs = []
        alias_drugs = []
        for item in alias_matches:
            mention = str(item.get("mention") or "").strip()
            canonical = str(item.get("canonical") or "").strip()
            if mention:
                alias_drugs.append(mention)
            if canonical and canonical not in canonical_drugs:
                canonical_drugs.append(canonical)

        ner_drugs = self._clean_list(ner_result.get("drug_entities"))
        drug_entities = self._merge_unique(canonical_drugs, ner_drugs)

        return {
            "drug_entities": drug_entities[:8],
            "drug_mentions": self._clean_list(alias_drugs),
            "symptom_entities": self._clean_list(ner_result.get("symptom_entities")),
            "disease_entities": self._clean_list(ner_result.get("disease_entities")),
            "population_entities": self._clean_list(ner_result.get("population_entities")),
            "food_entities": self._clean_list(ner_result.get("food_entities")),
            "dose_entities": self._clean_list(ner_result.get("dose_entities")),
        }

    def _apply_extracted_entities(self, result: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
        result["drug_entities"] = self._merge_unique(result.get("drug_entities") or [], extracted.get("drug_entities") or [])[:8]
        result["symptom_entities"] = self._merge_unique(result.get("symptom_entities") or [], extracted.get("symptom_entities") or [])[:8]
        result["disease_entities"] = self._merge_unique(result.get("disease_entities") or [], extracted.get("disease_entities") or [])[:8]
        result["population_entities"] = self._merge_unique(result.get("population_entities") or [], extracted.get("population_entities") or [])[:8]
        result["food_entities"] = self._merge_unique(result.get("food_entities") or [], extracted.get("food_entities") or [])[:8]
        result["dose_entities"] = self._merge_unique(result.get("dose_entities") or [], extracted.get("dose_entities") or [])[:8]
        result["constraints"] = self._extract_constraints(result.get("normalized_query") or "")

        if result.get("route") == "continue":
            llm_queries = self._clean_list(result.get("rewrite_queries"))
            slot_queries = self._build_rewrite_queries_from_slots(result)
            result["rewrite_queries"] = self._dedupe_queries(llm_queries + slot_queries, result.get("normalized_query") or "")
            if result.get("drug_entities"):
                result["domain"] = "drug_related"
        return result

    def build_retrieval_queries(self, analysis: Dict[str, Any], fallback: str = "") -> List[str]:
        normalized = analysis.get("resolved_query") or analysis.get("normalized_query") or self.normalize_query(fallback)
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
            "resolved_query": "",
            "clarification": "",
            "need_tool": False,
            "tool_candidates": [],
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
            "resolved_query": "",
            "clarification": "",
            "need_tool": False,
            "tool_candidates": [],
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
        normalized = analysis.get("resolved_query") or analysis.get("normalized_query", "")
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
