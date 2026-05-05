"""LLM服务 - 处理AI对话"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from typing import Dict, AsyncIterator, List, Optional
import logging
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME, REDIS_HOST, REDIS_PORT, REDIS_DB

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

class LLMService:
    """LLM服务类"""
    
    def __init__(self, tools: Optional[List] = None):
        try:
            # 初始化ChatOpenAI（兼容DashScope）
            logger.info(f"正在初始化 ChatOpenAI，model={MODEL_NAME}, base_url={DASHSCOPE_BASE_URL}")
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
        
        # 如果有工具，绑定工具（需要转换为LangChain工具格式）
        self.tools = tools
        
        # Redis 连接URL（用于记忆存储）
        redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        self.redis_url = redis_url
        
        # 记忆存储缓存（避免重复创建 RedisChatMessageHistory 实例）
        self.memories: Dict[str, RedisChatMessageHistory] = {}
        
        # 窗口大小（保留最近20轮对话）
        self.memory_window = 20
    
    def _get_memory(self, memory_id: str) -> RedisChatMessageHistory:
        """获取或创建记忆（从Redis）"""
        if memory_id not in self.memories:
            self.memories[memory_id] = RedisChatMessageHistory(
                session_id=memory_id,
                url=self.redis_url,
                key_prefix="chat_memory:",
                ttl=None  # 不过期
            )
        return self.memories[memory_id]
    
    def chat(self, memory_id: str, message: str, context: str = "") -> str:
        """非流式聊天"""
        memory = self._get_memory(memory_id)
        
        # 构建消息
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
        
        # 添加历史消息（从Redis读取，只取最近20轮）
        history_messages = memory.messages[-self.memory_window * 2:] if len(memory.messages) > self.memory_window * 2 else memory.messages
        messages.extend(history_messages)
        
        # 添加当前消息和上下文
        user_message = f"用户问题：{message}\n\n"
        if context:
            user_message += f"知识库内容（仅供回答参考，不用于判断是否调用工具）：\n{context}"
        
        messages.append(HumanMessage(content=user_message))
        
        # 调用LLM
        response = self.llm.invoke(messages)
        
        # 保存到记忆（写入Redis）
        from langchain_core.messages import HumanMessage as LC_HumanMessage, AIMessage
        memory.add_message(LC_HumanMessage(content=message))
        memory.add_message(AIMessage(content=response.content))
        
        return response.content
    
    async def chat_stream(self, memory_id: str, message: str, context: str = "") -> AsyncIterator[str]:
        """流式聊天"""
        memory = self._get_memory(memory_id)
        
        # 构建消息
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
        
        # 添加历史消息（从Redis读取，只取最近20轮）
        history_messages = memory.messages[-self.memory_window * 2:] if len(memory.messages) > self.memory_window * 2 else memory.messages
        messages.extend(history_messages)
        
        # 添加当前消息和上下文
        user_message = f"用户问题：{message}\n\n"
        if context:
            user_message += f"知识库内容（仅供回答参考，不用于判断是否调用工具）：\n{context}"
        
        messages.append(HumanMessage(content=user_message))
        
        # 流式调用LLM
        full_response = ""
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # 保存到记忆（写入Redis）
        from langchain_core.messages import HumanMessage as LC_HumanMessage, AIMessage
        memory.add_message(LC_HumanMessage(content=message))
        memory.add_message(AIMessage(content=full_response))
