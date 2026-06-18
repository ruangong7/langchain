"""Chat request orchestration service."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from services.llm_service import LLMService
from services.query_understanding import QueryUnderstandingService
from services.rag_service import RAGService
from tools.database_tool import DatabaseTool

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Coordinate query understanding, retrieval, and answer generation."""

    def __init__(
        self,
        query_understanding: QueryUnderstandingService,
        rag_service: RAGService,
        llm_service: LLMService,
        database_tool: Optional[DatabaseTool] = None,
    ):
        self.query_understanding = query_understanding
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.database_tool = database_tool

    def answer(self, memory_id: str, message: str, user_id: Optional[int] = None) -> str:
        analysis = self._prepare_analysis(memory_id, message)
        route = self._route_kind(analysis)

        if route == "ask_user":
            logger.info("意图分流: route=ask_user，返回澄清问题")
            return build_clarification_reply(analysis)
        if route == "general_answer":
            logger.info("意图分流: route=general_answer，直接调用通用大模型")
            return self.llm_service.chat_direct(memory_id, message)
        if route == "out_of_scope":
            logger.info("意图分流: route=out_of_scope，返回范围外提示")
            return build_out_of_scope_reply(analysis)

        context = self._retrieve_context(analysis, message)
        response = self.llm_service.chat(memory_id, message, context, self._build_personal_context(user_id))
        return response

    async def answer_stream(self, memory_id: str, message: str, user_id: Optional[int] = None) -> AsyncIterator[str]:
        analysis = self._prepare_analysis(memory_id, message)
        route = self._route_kind(analysis)

        if route == "ask_user":
            logger.info("意图分流: route=ask_user，返回澄清问题")
            yield build_clarification_reply(analysis)
            return
        if route == "general_answer":
            logger.info("意图分流: route=general_answer，直接调用通用大模型")
            async for chunk in self.llm_service.chat_stream_direct(memory_id, message):
                yield chunk
            return
        if route == "out_of_scope":
            logger.info("意图分流: route=out_of_scope，返回范围外提示")
            yield build_out_of_scope_reply(analysis)
            return

        context = self._retrieve_context(analysis, message)
        async for chunk in self.llm_service.chat_stream(
            memory_id,
            message,
            context,
            self._build_personal_context(user_id),
        ):
            yield chunk

    def _build_personal_context(self, user_id: Optional[int]) -> str:
        if user_id is None or self.database_tool is None:
            return ""
        try:
            return self.database_tool.build_user_personal_context(user_id)
        except Exception as exc:
            logger.warning("构建个体化上下文失败: user_id=%s error=%s", user_id, exc)
            return ""

    def _prepare_analysis(self, memory_id: str, message: str) -> Dict[str, Any]:
        initial = self.query_understanding.analyze(message)
        self._log_route("initial_route", initial)

        if not self._should_try_context_resolution(initial):
            self._log_route("final_route", initial)
            return initial

        resolved = self._resolve_with_history(memory_id, message, initial)
        self._log_route("final_route", resolved)
        return resolved

    @staticmethod
    def _should_try_context_resolution(analysis: Dict[str, Any]) -> bool:
        return analysis.get("route") == "ask_user" and analysis.get("reason") in {"ambiguous_reference", "unknown"}

    @staticmethod
    def _route_kind(analysis: Dict[str, Any]) -> str:
        route = str(analysis.get("route") or "")
        if route in {"ask_user", "general_answer", "out_of_scope", "continue"}:
            return route
        logger.warning("未知路由，按 ask_user 处理: route=%s analysis=%s", route, analysis)
        analysis["clarification"] = "这个问题我暂时没判断清楚。请补充你想查询的药品、症状或用药场景。"
        return "ask_user"

    def _retrieve_context(self, analysis: Dict[str, Any], message: str) -> str:
        retrieval_queries = self._query_texts(analysis, message)
        if analysis.get("context_resolved"):
            logger.info(
                "上下文补全检索: original=%s resolved=%s rewrite_queries=%s",
                analysis.get("original_query"),
                analysis.get("resolved_query"),
                retrieval_queries,
            )
        logger.info("意图分流: route=continue，进入医疗RAG，queries=%s", retrieval_queries)
        return self.rag_service.retrieve_context_multi(retrieval_queries)

    def _resolve_with_history(self, memory_id: str, message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        history = self.llm_service.memory_service.get_recent_turns(memory_id, turns=5)
        memory_summary = self.llm_service.memory_service.get_summary(memory_id)
        resolved_context = self.query_understanding.resolve_with_history(message, history, memory_summary)
        action = resolved_context.get("action")
        if action != "rewrite":
            clarification = resolved_context.get("clarification")
            if clarification:
                analysis["clarification"] = clarification
            logger.info(
                "上下文消歧未改写: action=%s reason=%s history_messages=%d",
                action,
                resolved_context.get("reason"),
                len(history),
            )
            return analysis

        resolved_query = resolved_context.get("rewritten_query", "").strip()
        if not resolved_query:
            logger.info("上下文消歧失败: LLM未返回改写问题 history_messages=%d", len(history))
            return analysis

        logger.info("上下文消歧尝试: original=%s resolved=%s history_messages=%d", message, resolved_query, len(history))
        resolved = self.query_understanding.analyze(resolved_query)
        self._log_route("context_resolved_analysis", resolved)
        if resolved.get("route") != "continue":
            logger.info("上下文消歧未采用: resolved_route=%s reason=%s", resolved.get("route"), resolved.get("reason"))
            return analysis
        resolved["original_query"] = message
        resolved["resolved_query"] = resolved_query
        resolved["context_resolved"] = True
        logger.info("上下文消歧成功: original=%s resolved=%s", message, resolved_query)
        return resolved

    def _query_texts(self, analysis: Dict, fallback: str) -> List[str]:
        if analysis.get("context_resolved") and analysis.get("resolved_query"):
            fallback = str(analysis.get("resolved_query") or fallback)
        return self.query_understanding.build_retrieval_queries(analysis, fallback)

    @staticmethod
    def _log_route(stage: str, analysis: Dict[str, Any]) -> None:
        logger.info(
            "%s: domain=%s route=%s intent=%s reason=%s normalized=%s context_resolved=%s resolved_query=%s",
            stage,
            analysis.get("domain"),
            analysis.get("route"),
            analysis.get("intent"),
            analysis.get("reason"),
            analysis.get("normalized_query"),
            analysis.get("context_resolved", False),
            analysis.get("resolved_query", ""),
        )

    @staticmethod
    def _clean_list(values: Any) -> List[str]:
        cleaned = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        return cleaned[:8]

    @staticmethod
    def _merge_unique(*groups: List[str]) -> List[str]:
        merged = []
        seen = set()
        for group in groups:
            for item in group or []:
                text = str(item or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    merged.append(text)
        return merged[:12]


def build_clarification_reply(analysis: Dict) -> str:
    if analysis.get("clarification"):
        return str(analysis.get("clarification"))
    reason = analysis.get("reason", "need_clarification")
    entities = analysis.get("drug_entities") or []
    candidates = analysis.get("drug_candidates") or []
    unknown_mentions = analysis.get("unknown_mentions") or []
    if reason == "possible_typo":
        if candidates:
            names = "、".join(item.get("canonical", "") for item in candidates if item.get("canonical"))
            return f"这看起来像是药名拼写有误。你说的是不是“{names}”？请确认后我再继续查。"
        return "这看起来像是药名拼写有误，请确认一下具体药名。"
    if reason == "confirm_drug_candidate":
        names = "、".join(item.get("canonical", "") for item in candidates if item.get("canonical")) or "、".join(entities)
        return f"我不太确定药名。你说的是不是“{names}”？请确认准确药名后我再查。"
    if reason == "unknown_drug":
        mention = "、".join(unknown_mentions)
        if mention:
            return f"我识别到“{mention}”像药品名，但药品字典里没有匹配项。请确认准确药名或补充通用名后我再查。"
        return "这个名称像药品名，但药品字典里没有匹配项。请确认准确药名或补充通用名后我再查。"
    if reason == "ambiguous_reference":
        return "这里的指代不够明确。请补充你说的是哪种药或哪一个问题，我再继续查。"
    if entities:
        return f"我识别到药品：{', '.join(entities)}。如果不是这个，请补充准确药名。"
    return "你说的药品不够明确，麻烦补充具体药名后我再查。"


def build_out_of_scope_reply(analysis: Dict) -> str:
    if analysis.get("domain") == "general":
        return "这个问题不属于用药或医疗查询范围。请换成药品、症状、疾病或用药相关问题。"
    return "这个问题暂时不在当前查询范围内。请补充更具体的医疗或用药信息。"
