"""LLM服务 - 处理AI对话"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from typing import Any, Callable, Dict, AsyncIterator, List, Optional
import logging
from config import (
    CHAT_MEMORY_SUMMARY_ENABLED,
    CHAT_MEMORY_SUMMARY_MAX_CHARS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
)
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)

# 系统提示词：与 evaluation/build_testset_from_redis.py 中 ground_truth 规则对齐（便于 RAG 评测可比）
SYSTEM_PROMPT = """你是中文医疗健康问答助手。

【当提供了「知识库内容」时（RAG）】
回答须与测试集生成时对 ground_truth 的要求一致：
- 只依据知识库中的文字作答，不引入知识库未出现的事实；信息不足、无关或为引导语时，用一两句说明材料无法支持或仅含何种信息即可。
- 用 1～3 句简洁、完整的中文直接作答；不要使用「结论：」「建议：」等模板，不要列举【参考一】或任何参考文献、来源列表。
- 若问题针对「该片段」「本材料」等，只概括知识库里实际写到的内容。

【处理规则】（是否调用工具只看「用户问题：」这一段，不看知识库内容）
1. 个人用药或个人饮食（含「我」「能不能」「我想」等）：应调用 queryJointData()，再结合知识库作答。
2. 药物相互作用、具体药名等：应调用 queryRealDrugDatabase()，再结合知识库作答。
3. 非药物类食品问题：可直接基于常识简短回答，不调用工具。

【未提供知识库时】
按上述规则决定是否调用工具；回答仍保持简短中文，不要编造专业细节。

【输出要求】
不要输出 JSON；不要展示分析过程；不要使用 Markdown 标题层级；语言简洁友好。
"""

GENERAL_SYSTEM_PROMPT = """你是一个简洁友好的中文助手。

直接回答用户的问题。不要调用工具，不要输出 JSON，不要展示分析过程。
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

class LLMService:
    ANSWER_HISTORY_TURNS = 7

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
        self.tool_llm = self.llm.bind_tools(self.tools) if self.tools else self.llm
        self.memory_service = memory_service or MemoryService()

    def _build_messages(
        self,
        history_messages: List[BaseMessage],
        message: str,
        context: str = "",
        memory_summary: str = "",
    ) -> List:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
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

    def _run_tool_call(self, tool_call: Dict[str, Any]) -> ToolMessage:
        name = tool_call.get("name") or ""
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id") or name

        handler = self.tool_handlers.get(name)
        if handler is None:
            logger.warning("模型请求了未注册工具: %s", name)
            return ToolMessage(content=f"工具 {name} 未注册，无法执行。", tool_call_id=call_id)

        try:
            result = handler(**args)
            if not isinstance(result, str):
                result = str(result)
            return ToolMessage(content=result[:6000], tool_call_id=call_id)
        except Exception as exc:
            logger.error("工具调用失败: %s args=%s error=%s", name, args, exc, exc_info=True)
            return ToolMessage(content=f"工具 {name} 调用失败：{exc}", tool_call_id=call_id)

    def _invoke_with_tools(self, messages: List):
        first_response = self.tool_llm.invoke(messages)
        tool_calls = getattr(first_response, "tool_calls", None) or []
        if not tool_calls:
            return first_response

        logger.info("模型触发工具调用: %s", [call.get("name") for call in tool_calls])
        tool_messages = [self._run_tool_call(call) for call in tool_calls]
        return self.llm.invoke([*messages, first_response, *tool_messages])
    
    def chat(self, memory_id: str, message: str, context: str = "") -> str:
        """非流式聊天"""
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = self._build_messages(history, message, context, memory_summary)
        response = self._invoke_with_tools(messages)
        
        # 保存到记忆（写入Redis）
        self.memory_service.append_exchange(memory_id, message, response.content)
        self._maybe_update_memory_summary(memory_id)
        
        return response.content

    def chat_direct(self, memory_id: str, message: str) -> str:
        """Direct general chat without tool probing or RAG context."""
        history = self.memory_service.get_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
        messages.extend([*history, HumanMessage(content=message)])
        response = self.llm.invoke(messages)
        self.memory_service.append_exchange(memory_id, message, response.content)
        self._maybe_update_memory_summary(memory_id)
        return response.content
    
    async def chat_stream(self, memory_id: str, message: str, context: str = "") -> AsyncIterator[str]:
        """流式聊天"""
        history = self._answer_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = self._build_messages(history, message, context, memory_summary)

        # 先用非流式探测工具调用，工具执行完后再流式输出最终答案。
        first_response = self.tool_llm.invoke(messages)
        tool_calls = getattr(first_response, "tool_calls", None) or []
        if tool_calls:
            logger.info("模型触发工具调用: %s", [call.get("name") for call in tool_calls])
            tool_messages = [self._run_tool_call(call) for call in tool_calls]
            messages = [*messages, first_response, *tool_messages]

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

    async def chat_stream_direct(self, memory_id: str, message: str) -> AsyncIterator[str]:
        """Direct streaming general chat without the preliminary tool-probing call."""
        history = self.memory_service.get_history(memory_id)
        memory_summary = self.memory_service.get_summary(memory_id)
        messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)]
        if memory_summary:
            messages.append(SystemMessage(content=MEMORY_SUMMARY_CONTEXT_PROMPT.format(summary=memory_summary)))
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
