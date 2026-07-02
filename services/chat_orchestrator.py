"""Chat request orchestration service."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from services.llm_service import LLMService
from services.query_understanding import QueryUnderstandingService
from tools.database_tool import DatabaseTool
from config import GRAPHRAG_FALLBACK_TO_LEGACY_RAG

if TYPE_CHECKING:
    from services.graphrag_service import GraphRAGService
    from services.rag_service import RAGService

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Coordinate query understanding, retrieval, and answer generation."""

    def __init__(
        self,
        query_understanding: QueryUnderstandingService,
        rag_service: Optional[RAGService],
        llm_service: LLMService,
        graphrag_service: Optional[GraphRAGService] = None,
        database_tool: Optional[DatabaseTool] = None,
    ):
        self.query_understanding = query_understanding
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.graphrag_service = graphrag_service
        self.database_tool = database_tool

    def answer(
        self,
        memory_id: str,
        message: str,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
    ) -> str:
        answer_text, _ = self.answer_with_meta(memory_id, message, user_id=user_id, tool_calls_enabled=tool_calls_enabled)
        return answer_text

    def build_context_snapshot(self, memory_id: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        personal_profile = self._load_personal_profile(user_id)
        effective_background_context, effective_background_snapshot = self._build_effective_background(
            user_id,
            personal_profile=personal_profile,
        )
        background_meta = self._background_meta(
            effective_background_snapshot,
            memory_available=self._has_memory_context(memory_id),
        )
        return {
            "memory_id": memory_id,
            "user_logged_in": user_id is not None,
            "profile_available": bool(background_meta["profile_available"]),
            "memory_available": bool(background_meta["memory_available"]),
            "effective_context": effective_background_snapshot,
            "effective_context_text": effective_background_context,
        }

    def answer_with_meta(
        self,
        memory_id: str,
        message: str,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
    ) -> tuple[str, Dict[str, Any]]:
        prepared = self.prepare_turn(memory_id, message, user_id=user_id, tool_calls_enabled=tool_calls_enabled)
        analysis = prepared["analysis"]
        route = prepared["route"]
        meta = prepared["meta"]

        if route == "ask_user":
            logger.info("意图分流: route=ask_user，返回澄清问题")
            meta["response_mode"] = "clarification"
            reply = build_clarification_reply(analysis)
            self.llm_service.memory_service.append_exchange(memory_id, message, reply, assistant_meta=meta)
            return reply, meta
        if route == "general_answer":
            logger.info("意图分流: route=general_answer，直接调用通用大模型")
            meta["response_mode"] = "general_answer"
            reply = self.llm_service.chat_direct(
                memory_id,
                message,
                personal_context=prepared.get("personal_profile_context", ""),
            )
            meta["tooling"] = self.llm_service.get_last_run_metadata()
            return reply, meta

        response = self.llm_service.chat(
            memory_id,
            message,
            prepared["context"],
            prepared.get("personal_profile_context", ""),
            prepared["allowed_tool_names"],
            prepared.get("runtime_tool_kwargs"),
        )
        meta["response_mode"] = "medical_answer"
        meta["tooling"] = self.llm_service.get_last_run_metadata()
        return response, meta

    async def answer_stream(
        self,
        memory_id: str,
        message: str,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
    ) -> AsyncIterator[str]:
        prepared = self.prepare_turn(memory_id, message, user_id=user_id, tool_calls_enabled=tool_calls_enabled)
        async for chunk in self.answer_stream_prepared(
            prepared,
            memory_id,
            message,
            user_id=user_id,
            tool_calls_enabled=tool_calls_enabled,
        ):
            yield chunk

    async def answer_stream_prepared(
        self,
        prepared: Dict[str, Any],
        memory_id: str,
        message: str,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
    ) -> AsyncIterator[str]:
        analysis = prepared["analysis"]
        route = prepared["route"]

        if route == "ask_user":
            logger.info("意图分流: route=ask_user，返回澄清问题")
            reply = build_clarification_reply(analysis)
            self.llm_service.memory_service.append_exchange(memory_id, message, reply, assistant_meta=prepared["meta"])
            yield reply
            return
        if route == "general_answer":
            logger.info("意图分流: route=general_answer，直接调用通用大模型")
            async for chunk in self.llm_service.chat_stream_direct(
                memory_id,
                message,
                personal_context=prepared.get("personal_profile_context", ""),
            ):
                yield chunk
            return

        async for chunk in self.llm_service.chat_stream(
            memory_id,
            message,
            prepared["context"],
            prepared.get("personal_profile_context", ""),
            prepared["allowed_tool_names"],
            prepared.get("runtime_tool_kwargs"),
        ):
            yield chunk

    def prepare_turn(
        self,
        memory_id: str,
        message: str,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
    ) -> Dict[str, Any]:
        personal_profile = self._load_personal_profile(user_id)
        analysis = self._prepare_analysis(
            memory_id,
            message,
            user_id=user_id,
            tool_calls_enabled=tool_calls_enabled,
            personal_profile=personal_profile,
        )
        personal_profile_context = self._merge_personal_contexts(
            self._build_personal_profile_context(personal_profile),
            str(analysis.get("prefetched_medication_context") or "").strip(),
        )
        route = self._route_kind(analysis)
        meta = self._base_meta(analysis, user_id)
        meta["tool_calls_effective"] = bool(tool_calls_enabled)
        if route == "ask_user":
            meta["response_mode"] = "clarification"
        elif route == "general_answer":
            meta["response_mode"] = "general_answer"
        else:
            meta["response_mode"] = "medical_answer"
        prepared: Dict[str, Any] = {
            "analysis": analysis,
            "route": route,
            "meta": meta,
            "context": "",
            "effective_background_context": "",
            "effective_background_snapshot": {},
            "personal_profile_context": personal_profile_context,
            "allowed_tool_names": [],
            "runtime_tool_kwargs": {},
        }
        if route == "continue":
            context, retrieval_meta = self._retrieve_context_with_meta(analysis, message)
            effective_background_context, effective_background_snapshot = self._build_effective_background(
                user_id,
                personal_profile=personal_profile,
            )
            allowed_tool_names = self._select_allowed_tools(analysis, message, user_id, tool_calls_enabled=tool_calls_enabled)
            runtime_tool_kwargs = self._build_runtime_tool_kwargs(allowed_tool_names, user_id)
            meta["retrieval"] = retrieval_meta
            meta["allowed_tool_names"] = allowed_tool_names
            meta["background"] = self._background_meta(
                effective_background_snapshot,
                memory_available=self._has_memory_context(memory_id),
            )
            prepared["context"] = context
            prepared["effective_background_context"] = effective_background_context
            prepared["effective_background_snapshot"] = effective_background_snapshot
            prepared["allowed_tool_names"] = allowed_tool_names
            prepared["runtime_tool_kwargs"] = runtime_tool_kwargs
        elif personal_profile_context:
            snapshot = self.llm_service.memory_service.build_effective_context_snapshot(personal_profile)
            meta["background"] = {
                "profile_available": bool(personal_profile_context),
                "memory_available": self._has_memory_context(memory_id),
                "conditions": list(snapshot.get("conditions") or [])[:3],
                "current_medications": [],
            }
        return prepared

    @staticmethod
    def _merge_personal_contexts(*parts: str) -> str:
        merged = []
        seen = set()
        for part in parts:
            text = str(part or "").strip()
            if text and text not in seen:
                seen.add(text)
                merged.append(text)
        return "\n\n".join(merged)

    def _load_personal_profile(self, user_id: Optional[int]) -> Dict[str, Any]:
        if user_id is None or self.database_tool is None:
            return {}
        memory_service = getattr(self.llm_service, "memory_service", None)
        get_cache = getattr(memory_service, "get_user_profile_cache", None)
        set_cache = getattr(memory_service, "set_user_profile_cache", None)
        if callable(get_cache):
            cached = get_cache(int(user_id))
            profile = cached.get("profile") if isinstance(cached, dict) else {}
            if isinstance(profile, dict) and profile:
                return profile
        try:
            profile = self.database_tool.get_user_health_profile(user_id, include_medications=False)
            if callable(set_cache):
                set_cache(
                    int(user_id),
                    profile,
                    self._build_personal_profile_context(profile),
                )
            return profile
        except Exception as exc:
            logger.warning("构建个体化上下文失败: user_id=%s error=%s", user_id, exc)
            return {}

    def _build_effective_background(
        self,
        user_id: Optional[int],
        personal_profile: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        profile = personal_profile if personal_profile is not None else self._load_personal_profile(user_id)
        snapshot = self.llm_service.memory_service.build_effective_context_snapshot(profile)
        return self.llm_service.memory_service.format_effective_context_snapshot(snapshot), snapshot

    def _build_personal_profile_context(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return ""
        renderer = getattr(self.database_tool, "render_profile_context", None)
        if callable(renderer):
            return str(renderer(profile) or "")

        lines = ["[用户个人档案]"]
        if profile.get("display_name"):
            lines.append("称呼: " + str(profile["display_name"]))
        if profile.get("gender"):
            lines.append("性别: " + str(profile["gender"]))
        if profile.get("age") is not None:
            lines.append("年龄: " + str(profile["age"]))
        if profile.get("conditions"):
            lines.append("基础病: " + "、".join(self._clean_list(profile.get("conditions"))))
        if profile.get("allergies"):
            lines.append("过敏史: " + "、".join(self._clean_list(profile.get("allergies"))))
        if profile.get("notes"):
            lines.append("备注: " + str(profile["notes"]))
        return "" if len(lines) == 1 else "\n".join(lines)

    @staticmethod
    def _background_meta(snapshot: Dict[str, Any], memory_available: bool = False) -> Dict[str, Any]:
        if not snapshot:
            return {
                "profile_available": False,
                "memory_available": memory_available,
                "conditions": [],
                "current_medications": [],
            }
        return {
            "profile_available": True,
            "memory_available": memory_available,
            "conditions": list(snapshot.get("conditions") or [])[:3],
            "current_medications": [],
        }

    def _has_memory_context(self, memory_id: str) -> bool:
        if self.llm_service.memory_service.get_summary(memory_id):
            return True
        return bool(self.llm_service.memory_service.get_recent_turns(memory_id, turns=1))

    def _prepare_analysis(
        self,
        memory_id: str,
        message: str,
        *,
        user_id: Optional[int] = None,
        tool_calls_enabled: bool = True,
        personal_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history = self.llm_service.memory_service.get_recent_turns(memory_id, turns=5)
        memory_summary = self.llm_service.memory_service.get_summary(memory_id)
        understanding_tools = self._available_user_tools_for_understanding(user_id)
        profile = personal_profile if personal_profile is not None else self._load_personal_profile(user_id)
        result = self.query_understanding.analyze(
            message,
            history=history,
            memory_summary=memory_summary,
            user_logged_in=user_id is not None,
            available_tools=understanding_tools,
            profile_available=self._profile_has_context(profile),
            user_id=user_id,
            personal_profile_context=self._build_personal_profile_context(profile),
        )
        self._log_route("final_route", result)
        return result

    def _select_allowed_tools(
        self,
        analysis: Dict[str, Any],
        message: str,
        user_id: Optional[int],
        tool_calls_enabled: bool = True,
    ) -> List[str]:
        if self.database_tool is None or analysis.get("route") != "continue" or not tool_calls_enabled:
            return []
        if analysis.get("skip_retrieval") and analysis.get("prefetched_context"):
            logger.info("本轮工具暴露: 已有预取工具上下文，跳过重复暴露 tools=%s", analysis.get("prefetched_tools") or [])
            return []
        allowed = []
        available_tools = self._available_user_tools(user_id, tool_calls_enabled=tool_calls_enabled)
        tool_candidates = [name for name in (analysis.get("tool_candidates") or []) if name in available_tools]
        if tool_candidates:
            allowed = list(tool_candidates)
        else:
            allowed = list(available_tools)
        prefetched_tools = set(analysis.get("prefetched_tools") or [])
        if prefetched_tools:
            allowed = [name for name in allowed if name not in prefetched_tools]
        compact = re.sub(r"\s+", "", str(message or ""))
        personal_query = user_id is not None and any(term in compact for term in ("我", "本人", "自己", "正在吃", "在吃", "长期吃"))

        seen = set()
        deduped = []
        for name in allowed:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        logger.info(
            "本轮工具暴露: personal_query=%s user_id=%s tools=%s tool_candidates=%s",
            personal_query,
            user_id,
            deduped,
            analysis.get("tool_candidates") or [],
        )
        return deduped

    def _available_user_tools(self, user_id: Optional[int], *, tool_calls_enabled: bool = True) -> List[str]:
        if self.database_tool is None or user_id is None or not tool_calls_enabled:
            return []
        return self._available_user_tools_for_understanding(user_id)

    def _available_user_tools_for_understanding(self, user_id: Optional[int]) -> List[str]:
        if self.database_tool is None or user_id is None:
            return []
        capabilities = self.database_tool.get_capabilities()
        allowed: List[str] = []
        if capabilities.get("user_health_profile_table"):
            allowed.append("queryUserHealthProfile")
        if capabilities.get("user_medications_table"):
            allowed.append("queryUserMedicationSummary")
        return allowed

    @staticmethod
    def _build_runtime_tool_kwargs(allowed_tool_names: List[str], user_id: Optional[int]) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        if user_id is not None and "queryUserHealthProfile" in allowed_tool_names:
            payload["queryUserHealthProfile"] = {"user_id": int(user_id)}
        if user_id is not None and "queryUserMedicationSummary" in allowed_tool_names:
            payload["queryUserMedicationSummary"] = {"user_id": int(user_id)}
        return payload

    @staticmethod
    def _route_kind(analysis: Dict[str, Any]) -> str:
        route = str(analysis.get("route") or "")
        if route in {"ask_user", "general_answer", "continue"}:
            return route
        logger.warning("未知路由，按 ask_user 处理: route=%s analysis=%s", route, analysis)
        analysis["clarification"] = "这个问题我暂时没判断清楚。请补充你想查询的药品、症状或用药场景。"
        return "ask_user"

    def _retrieve_context(self, analysis: Dict[str, Any], message: str) -> str:
        context, _ = self._retrieve_context_with_meta(analysis, message)
        return context

    def _retrieve_context_with_meta(self, analysis: Dict[str, Any], message: str) -> tuple[str, Dict[str, Any]]:
        if analysis.get("skip_retrieval"):
            return (
                str(analysis.get("prefetched_context") or ""),
                {
                    "queries": [],
                    "backend": "prefetched_context",
                    "method": "direct",
                    "prefetched_tools": list(analysis.get("prefetched_tools") or []),
                },
            )
        retrieval_queries = self._query_texts(analysis, message)
        meta = {
            "queries": retrieval_queries,
            "backend": "none",
            "method": "",
        }
        if analysis.get("context_resolved"):
            logger.info(
                "上下文补全检索: original=%s resolved=%s rewrite_queries=%s",
                analysis.get("original_query"),
                analysis.get("resolved_query"),
                retrieval_queries,
            )
        logger.info("意图分流: route=continue，进入医疗RAG，queries=%s", retrieval_queries)

        primary_query = str(analysis.get("resolved_query") or analysis.get("normalized_query") or message).strip()
        if self.graphrag_service is not None:
            try:
                meta["backend"] = "graphrag"
                meta["method"] = "local" if str(analysis.get("intent") or "") in {"interaction", "contraindication", "side_effect", "dosage", "population", "drug_info"} else "auto"
                return self.graphrag_service.retrieve_context_for_analysis(analysis, primary_query), meta
            except Exception as exc:
                if not GRAPHRAG_FALLBACK_TO_LEGACY_RAG:
                    logger.exception("GraphRAG 检索失败，且已禁用旧RAG回退")
                    raise RuntimeError("GraphRAG 检索失败，已禁用旧向量检索/BM25回退") from exc
                logger.warning("GraphRAG 检索失败，按配置回退旧RAG: error=%s", exc, exc_info=True)

        if self.rag_service is None:
            raise RuntimeError("未初始化可用检索服务")
        meta["backend"] = "legacy_rag"
        meta["method"] = "hybrid"
        context, rag_meta = self.rag_service.retrieve_context_multi_with_meta(retrieval_queries)
        if rag_meta:
            meta.update(rag_meta)
        return context, meta

    @staticmethod
    def _base_meta(analysis: Dict[str, Any], user_id: Optional[int]) -> Dict[str, Any]:
        return {
            "route": str(analysis.get("route") or ""),
            "domain": str(analysis.get("domain") or ""),
            "intent": str(analysis.get("intent") or ""),
            "context_resolved": bool(analysis.get("context_resolved")),
            "user_logged_in": user_id is not None,
            "need_tool": bool(analysis.get("need_tool")),
            "tool_candidates": list(analysis.get("tool_candidates") or []),
            "skip_retrieval": bool(analysis.get("skip_retrieval")),
            "drug_entities": list(analysis.get("drug_entities") or []),
            "disease_entities": list(analysis.get("disease_entities") or []),
            "symptom_entities": list(analysis.get("symptom_entities") or []),
        }

    def _query_texts(self, analysis: Dict, fallback: str) -> List[str]:
        if analysis.get("context_resolved") and analysis.get("resolved_query"):
            fallback = str(analysis.get("resolved_query") or fallback)
        return self.query_understanding.build_retrieval_queries(analysis, fallback)

    @classmethod
    def _profile_has_context(cls, profile: Dict[str, Any]) -> bool:
        return any(
            [
                str(profile.get("display_name") or "").strip(),
                str(profile.get("gender") or "").strip(),
                profile.get("age") is not None,
                bool(cls._clean_list(profile.get("conditions"))),
                bool(cls._clean_list(profile.get("allergies"))),
                str(profile.get("notes") or "").strip(),
                bool(profile.get("is_pregnant")),
                bool(profile.get("is_breastfeeding")),
            ]
        )

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
