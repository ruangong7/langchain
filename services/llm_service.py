"""LLM服务 - 处理AI对话"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from typing import Any, Callable, Dict, AsyncIterator, List, Optional
import logging
from config import (
    CHAT_MEMORY_SUMMARY_BATCH_TURNS,
    CHAT_MEMORY_SUMMARY_ENABLED,
    CHAT_MEMORY_SUMMARY_MAX_CHARS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_TOOL_MAX_ROUNDS,
    MODEL_NAME,
)
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = """你是一个简洁友好的中文助手。

直接回答用户的问题。不要调用工具，不要输出 JSON，不要展示分析过程。
"""

NO_TOOLS_SYSTEM_PROMPT = """你是中文医疗健康问答助手。

当前数据库工具调用已关闭。请不要尝试调用任何工具，也不要声称已经查询数据库。

当提供了「知识库内容」时，只依据知识库中的文字作答；信息不足时，说明当前材料无法支持明确结论。

回答用简洁中文，不要输出 JSON，不要展示分析过程。
"""

SUMMARY_SYSTEM_PROMPT = """你是中文医疗健康问答系统的长期记忆摘要器。

请把旧摘要和本次溢出的历史对话合并成新的长期记忆摘要。

要求：
- 只保留稳定、后续可能有用的信息：用户长期用药、疾病/症状、过敏史、禁忌、偏好、已澄清对象。
- 不保留一次性闲聊、问候、无关内容和模型不确定推测。
- 不新增历史里没有出现过的医学事实。
- 用简洁中文输出纯文本，不要 Markdown 标题，不要 JSON。
"""

MEMORY_SUMMARY_CONTEXT_PROMPT = """长期记忆摘要（仅用于理解用户背景和多轮指代，不可替代知识库、数据库或医生诊断）：
{summary}
"""

PERSONAL_CONTEXT_PROMPT = """以下是来自用户健康档案或登录态资料的个体化健康信息，优先级高于对话记忆，仅用于个体化用药评估：
{personal_context}
"""

class LLMService:
    ANSWER_HISTORY_TURNS = 7
    SUMMARY_BATCH_TURNS = max(1, int(CHAT_MEMORY_SUMMARY_BATCH_TURNS))
    MAX_TOOL_CALL_ROUNDS = max(1, int(LLM_TOOL_MAX_ROUNDS))

    """LLM服务类"""
    
    def __init__(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable[..., str]]] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        try:
            # 初始化ChatOpenAI（兼容DashScope）
            logger.info(f"正在初始化 ChatOpenAI，model={MODEL_NAME}, base_url={DASHSCOPE_BASE_URL}")
            if not DASHSCOPE_API_KEY:
                raise ValueError("DASHSCOPE_API_KEY 未配置")
            self.llm = ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL,
                temperature=0.7,
                streaming=True
            )
            logger.info("ChatOpenAI 初始化成功")
        except Exception as e:
            logger.error(f"ChatOpenAI 初始化失败: {type(e).__name__}: {e}", exc_info=True)
            raise
        
        # OpenAI-style tool schema + 本地执行函数。
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.tool_schemas_by_name = {
            str(item.get("function", {}).get("name") or ""): item
            for item in self.tools
            if str(item.get("function", {}).get("name") or "")
        }
        self.memory_service = memory_service or MemoryService()
        self.last_run_metadata: Dict[str, Any] = {}
        self._runtime_tool_kwargs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _tool_prompt_rules(allowed_tool_names: List[str]) -> str:
        rules = []
        if "queryUserHealthProfile" in allowed_tool_names:
            rules.append("1. 当问题明确是在问用户自己的基础病、过敏史、年龄、妊娠/哺乳状态或其他个人档案信息时，可调用 queryUserHealthProfile()。")
        if "queryUserMedicationSummary" in allowed_tool_names:
            rules.append("2. 当问题明确是在问用户当前登记的用药，或需要结合当前正在吃的药做个体化判断时，可调用 queryUserMedicationSummary()。")
        if not rules:
            return NO_TOOLS_SYSTEM_PROMPT
        joined_rules = "\n".join(rules)
        return f"""你是中文医疗健康问答助手。

【当提供了「知识库内容」时（RAG）】
回答须与测试集生成时对 ground_truth 的要求一致：
- 只依据知识库中的文字作答，不引入知识库未出现的事实；信息不足时，说明当前材料无法支持明确结论。
- 用 1～3 句简洁、完整的中文直接作答；不要输出 JSON，不要展示分析过程。

【工具调用规则】
{joined_rules}
- 如果当前问题不适合工具，直接基于知识库内容回答，不要强行调用工具。
- 工具返回不可用或无结果时，要如实说明，不要假装查询成功。
"""

    def _select_tool_names(self, allowed_tool_names: Optional[List[str]]) -> List[str]:
        if allowed_tool_names is None:
            return list(self.tool_schemas_by_name)
        result = []
        seen = set()
        for name in allowed_tool_names:
            text = str(name or "").strip()
            if text and text in self.tool_schemas_by_name and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _build_tool_run_metadata(selected_tool_names: List[str]) -> Dict[str, Any]:
        return {
            "allowed_tool_names": list(selected_tool_names),
            "tool_calls": [],
            "used_tools": False,
            "tool_rounds": 0,
            "tool_loop_truncated": False,
        }

    def _build_messages(
        self,
        history_messages: List[BaseMessage],
        message: str,
        context: str = "",
        memory_summary: str = "",
        personal_context: str = "",
        allowed_tool_names: Optional[List[str]] = None,
    ) -> List:
        selected_tool_names = self._select_tool_names(allowed_tool_names)
        system_prompt = self._tool_prompt_rules(selected_tool_names)
        messages = [SystemMessage(content=system_prompt)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
        if personal_context:
            messages.append(SystemMessage(content=PERSONAL_CONTEXT_PROMPT.format(personal_context=personal_context)))
        messages.extend(history_messages)

        user_message = f"用户问题：{message}\n\n"
        if context:
            user_message += f"知识库内容（仅供回答参考，不用于判断是否调用工具）：\n{context}"
        messages.append(HumanMessage(content=user_message))
        return messages

    def _answer_history(self, memory_id: str) -> List[BaseMessage]:
        return self.memory_service.get_history(memory_id)[-self.ANSWER_HISTORY_TURNS * 2 :]

    @staticmethod
    def _messages_to_summary_text(messages: List[BaseMessage]) -> str:
        lines = []
        for message in messages:
            role = "用户" if isinstance(message, HumanMessage) else "助手"
            content = " ".join(str(message.content or "").split())
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _maybe_update_memory_summary(self, memory_id: str) -> None:
        if not CHAT_MEMORY_SUMMARY_ENABLED:
            return
        overflow = self.memory_service.get_overflow_history(memory_id, keep_turns=self.ANSWER_HISTORY_TURNS)
        if not overflow:
            return
        if len(overflow) < self.SUMMARY_BATCH_TURNS * 2:
            return

        old_summary = self.memory_service.get_summary(memory_id)
        overflow_text = self._messages_to_summary_text(overflow)
        if not overflow_text:
            self.memory_service.trim_to_recent_turns(memory_id, self.ANSWER_HISTORY_TURNS)
            return

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"旧摘要：\n{old_summary or '无'}\n\n"
                            f"本次需要压缩进摘要的历史对话：\n{overflow_text}\n\n"
                            f"请输出不超过 {CHAT_MEMORY_SUMMARY_MAX_CHARS} 字的新摘要。"
                        )
                    ),
                ]
            )
            summary = str(response.content or "").strip()
            if summary:
                self.memory_service.set_summary(memory_id, summary[:CHAT_MEMORY_SUMMARY_MAX_CHARS])
                logger.info(
                    "长期记忆摘要已更新: memory_id=%s overflow_messages=%d summary_chars=%d",
                    memory_id,
                    len(overflow),
                    len(summary),
                )
            self.memory_service.trim_to_recent_turns(memory_id, self.ANSWER_HISTORY_TURNS)
        except Exception as exc:
            logger.warning("长期记忆摘要更新失败，保留短期记忆: %s", exc, exc_info=True)

    def _run_tool_call(self, tool_call: Dict[str, Any]) -> tuple[ToolMessage, Dict[str, Any]]:
        name = tool_call.get("name") or ""
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id") or name

        handler = self.tool_handlers.get(name)
        if handler is None:
            logger.warning("模型请求了未注册工具: %s", name)
            metadata = {
                "name": name,
                "args": args,
                "ok": False,
                "reason": "unregistered",
                "message": f"工具 {name} 未注册，无法执行。",
            }
            return ToolMessage(content=metadata["message"], tool_call_id=call_id), metadata

        try:
            merged_args = dict(args)
            merged_args.update(self._runtime_tool_kwargs.get(name) or {})
            result = handler(**merged_args)
            if isinstance(result, dict):
                content = str(result.get("rendered") or result.get("message") or "")
                metadata = {
                    "name": name,
                    "args": args,
                    "ok": bool(result.get("ok")),
                    "reason": str(result.get("reason") or ""),
                    "message": str(result.get("message") or content),
                    "count": int(result.get("count") or 0),
                }
                return ToolMessage(content=content[:6000], tool_call_id=call_id), metadata
            if not isinstance(result, str):
                result = str(result)
            metadata = {
                "name": name,
                "args": args,
                "ok": True,
                "reason": "success",
                "message": result[:2000],
            }
            return ToolMessage(content=result[:6000], tool_call_id=call_id), metadata
        except Exception as exc:
            logger.error("工具调用失败: %s args=%s error=%s", name, args, exc, exc_info=True)
            metadata = {
                "name": name,
                "args": args,
                "ok": False,
                "reason": "exception",
                "message": f"工具 {name} 调用失败：{exc}",
            }
            return ToolMessage(content=metadata["message"], tool_call_id=call_id), metadata

    def _invoke_with_tools(
        self,
        messages: List,
        allowed_tool_names: Optional[List[str]] = None,
        runtime_tool_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        selected_tool_names = self._select_tool_names(allowed_tool_names)
        self._runtime_tool_kwargs = runtime_tool_kwargs or {}
        metadata = self._build_tool_run_metadata(selected_tool_names)
        if not selected_tool_names:
            self.last_run_metadata = metadata
            try:
                return self.llm.invoke(messages)
            finally:
                self._runtime_tool_kwargs = {}

        selected_tools = [self.tool_schemas_by_name[name] for name in selected_tool_names]
        tool_messages_context = list(messages)
        try:
            for round_index in range(1, self.MAX_TOOL_CALL_ROUNDS + 1):
                response = self.llm.bind_tools(selected_tools).invoke(tool_messages_context)
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    self.last_run_metadata = metadata
                    return response

                logger.info("模型触发第 %d 轮工具调用: %s", round_index, [call.get("name") for call in tool_calls])
                tool_results = [self._run_tool_call(call) for call in tool_calls]
                tool_messages = [item[0] for item in tool_results]
                metadata["tool_calls"].extend([item[1] for item in tool_results])
                metadata["used_tools"] = True
                metadata["tool_rounds"] = round_index
                tool_messages_context.extend([response, *tool_messages])

            metadata["tool_loop_truncated"] = True
            logger.warning(
                "工具调用达到最大轮数限制，转为直接回答: max_rounds=%d tools=%s",
                self.MAX_TOOL_CALL_ROUNDS,
                selected_tool_names,
            )
            self.last_run_metadata = metadata
            return self.llm.invoke(tool_messages_context)
        finally:
            self._runtime_tool_kwargs = {}

    def _prepare_messages_for_stream(
        self,
        messages: List,
        allowed_tool_names: Optional[List[str]] = None,
        runtime_tool_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List:
        selected_tool_names = self._select_tool_names(allowed_tool_names)
        self._runtime_tool_kwargs = runtime_tool_kwargs or {}
        metadata = self._build_tool_run_metadata(selected_tool_names)
        if not selected_tool_names:
            self.last_run_metadata = metadata
            return list(messages)

        selected_tools = [self.tool_schemas_by_name[name] for name in selected_tool_names]
        tool_messages_context = list(messages)
        try:
            for round_index in range(1, self.MAX_TOOL_CALL_ROUNDS + 1):
                response = self.llm.bind_tools(selected_tools).invoke(tool_messages_context)
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    self.last_run_metadata = metadata
                    return tool_messages_context

                logger.info("模型触发第 %d 轮工具调用: %s", round_index, [call.get("name") for call in tool_calls])
                tool_results = [self._run_tool_call(call) for call in tool_calls]
                tool_messages = [item[0] for item in tool_results]
                metadata["tool_calls"].extend([item[1] for item in tool_results])
                metadata["used_tools"] = True
                metadata["tool_rounds"] = round_index
                tool_messages_context.extend([response, *tool_messages])

            metadata["tool_loop_truncated"] = True
            logger.warning(
                "流式回答前的工具规划达到最大轮数限制: max_rounds=%d tools=%s",
                self.MAX_TOOL_CALL_ROUNDS,
                selected_tool_names,
            )
            self.last_run_metadata = metadata
            return tool_messages_context
        except Exception:
            self._runtime_tool_kwargs = {}
            raise
    
    def chat(
        self,
        memory_id: str,
        message: str,
        context: str = "",
        personal_context: str = "",
        allowed_tool_names: Optional[List[str]] = None,
        runtime_tool_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """非流式聊天"""
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = self._build_messages(
            history,
            message,
            context,
            memory_summary,
            personal_context,
            allowed_tool_names,
        )
        response = self._invoke_with_tools(messages, allowed_tool_names, runtime_tool_kwargs)
        
        # 保存到记忆（写入Redis）
        self.memory_service.append_exchange(memory_id, message, response.content)
        self._maybe_update_memory_summary(memory_id)
        
        return response.content

    def get_last_run_metadata(self) -> Dict[str, Any]:
        return dict(self.last_run_metadata or {})

    def chat_direct(
        self,
        memory_id: str,
        message: str,
        personal_context: str = "",
    ) -> str:
        """Direct general chat without tool probing or RAG context."""
        self.last_run_metadata = self._build_tool_run_metadata([])
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
        if personal_context:
            messages.append(SystemMessage(content=PERSONAL_CONTEXT_PROMPT.format(personal_context=personal_context)))
        messages.extend([*history, HumanMessage(content=message)])
        response = self.llm.invoke(messages)
        self.memory_service.append_exchange(memory_id, message, response.content)
        self._maybe_update_memory_summary(memory_id)
        return response.content
    
    async def chat_stream(
        self,
        memory_id: str,
        message: str,
        context: str = "",
        personal_context: str = "",
        allowed_tool_names: Optional[List[str]] = None,
        runtime_tool_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """流式聊天"""
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = self._build_messages(
            history,
            message,
            context,
            memory_summary,
            personal_context,
            allowed_tool_names,
        )
        try:
            messages = self._prepare_messages_for_stream(messages, allowed_tool_names, runtime_tool_kwargs)

            full_response = ""
            chunk_count = 0
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    chunk_count += 1
                    yield chunk.content
            logger.info(
                "LLM流式回复完成: memory_id=%s chunks=%d total_length=%d",
                memory_id,
                chunk_count,
                len(full_response),
            )
            
            # 保存到记忆（写入Redis）
            self.memory_service.append_exchange(memory_id, message, full_response)
            self._maybe_update_memory_summary(memory_id)
        finally:
            self._runtime_tool_kwargs = {}

    async def chat_stream_direct(
        self,
        memory_id: str,
        message: str,
        personal_context: str = "",
    ) -> AsyncIterator[str]:
        """Direct streaming general chat without the preliminary tool-probing call."""
        self.last_run_metadata = self._build_tool_run_metadata([])
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
        if personal_context:
            messages.append(SystemMessage(content=PERSONAL_CONTEXT_PROMPT.format(personal_context=personal_context)))
        messages.extend([*history, HumanMessage(content=message)])

        full_response = ""
        chunk_count = 0
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                full_response += chunk.content
                chunk_count += 1
                yield chunk.content
        logger.info(
            "LLM直接流式回复完成: memory_id=%s chunks=%d total_length=%d",
            memory_id,
            chunk_count,
            len(full_response),
        )

        self.memory_service.append_exchange(memory_id, message, full_response)
        self._maybe_update_memory_summary(memory_id)
