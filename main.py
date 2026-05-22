"""FastAPI主应用"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import logging
from pathlib import Path
from typing import AsyncIterator, List
from pydantic import BaseModel

from config import *
from services.chat_orchestrator import ChatOrchestrator
from services.rag_service import RAGService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.query_understanding import QueryUnderstandingService
from services.user_auth_service import AuthError, UserAuthService
from tools.database_tool import DatabaseTool
from retriever.hybrid_retriever import HybridRetriever
from retriever.sparse_retriever import SparseRetriever
from langchain_community.vectorstores import Redis
from langchain_core.embeddings import Embeddings
import dashscope
from dashscope import TextEmbedding

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
auth_service = UserAuthService()


class AuthRequest(BaseModel):
    username: str
    password: str


class DashScopeEmbeddings(Embeddings):
    """使用 DashScope 官方 SDK 的 Embeddings 封装"""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # DashScope 端对 batch size 有限制（常见为 <=10），这里做分批，避免 400
        max_batch = 10
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i : i + max_batch]
            resp = TextEmbedding.call(model=self.model, input=batch)
            if not resp or not getattr(resp, "output", None) or "embeddings" not in resp.output:
                raise ValueError(
                    f"DashScope TextEmbedding 返回异常: model={self.model}, batch_size={len(batch)}, resp={resp}"
                )
            embeddings = resp.output["embeddings"]
            all_vectors.extend([item["embedding"] for item in embeddings])
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期（替代 on_event 的 startup/shutdown）"""
    global rag_service, llm_service, query_understanding_service, chat_orchestrator, database_tool

    logger.info("正在初始化服务...")

    try:
        # 配置 DashScope API Key（用于 Chat 与 Embedding）
        dashscope.api_key = DASHSCOPE_API_KEY

        # 初始化数据库工具。数据库不可用不阻断 RAG 服务启动，工具调用时会明确返回不可用。
        try:
            database_tool = DatabaseTool()
        except Exception as exc:
            database_tool = None
            logger.warning("数据库工具初始化失败，工具调用将返回不可用: %s", exc, exc_info=True)

        # 初始化工具列表（用于LLM）
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "queryRealDrugDatabase",
                    "description": "查询real_drug数据库，获取药物的详细信息，包括饮食禁忌、相互作用等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drugName": {
                                "type": "string",
                                "description": "药物名称",
                            }
                        },
                        "required": ["drugName"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "queryJointData",
                    "description": "联合查询：先查yinshi获取用户用药，再以药物名称为外键查询real_drug，一次调用获取所有相关数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "查询问题",
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
        ]

        def query_real_drug_database(drugName: str) -> str:
            if database_tool is None:
                return "数据库工具当前不可用，无法查询 real_drug。"
            return database_tool.query_real_drug_database(drugName)

        def query_joint_data(question: str) -> str:
            if database_tool is None:
                return "数据库工具当前不可用，无法执行联合查询。"
            return database_tool.query_joint_data(question)

        tool_handlers = {
            "queryRealDrugDatabase": query_real_drug_database,
            "queryJointData": query_joint_data,
        }

        # 初始化LLM服务
        try:
            logger.info("正在初始化 LLMService...")
            memory_service = MemoryService()
            llm_service = LLMService(tools=tools, tool_handlers=tool_handlers, memory_service=memory_service)
            logger.info("LLMService 初始化成功")
        except Exception as e:
            logger.error(f"LLMService 初始化失败: {type(e).__name__}: {e}", exc_info=True)
            raise

        # 初始化向量存储（Redis）——使用 DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)

        # 新版 Redis 向量库需要使用 redis_url 参数
        redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        vectorstore = Redis(
            redis_url=redis_url,
            index_name="drug_vectors",
            embedding=embeddings,
        )

        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # 初始化稀疏检索器
        content_dir = "content"
        sparse_retriever = SparseRetriever.from_content_dir(content_dir, top_k=10)

        # 首次启动时：Redis 里可能还没有创建 drug_vectors 索引，也没有写入向量数据
        # Redis 向量库会在首次 add_texts 时自动创建索引（_create_index_if_not_exist）
        try:
            vectorstore.client.ft("drug_vectors").info()  # type: ignore[attr-defined]
        except Exception:
            texts = [d.page_content for d in sparse_retriever.documents]
            metadatas = [d.metadata for d in sparse_retriever.documents]
            if texts:
                logger.info(f"检测到 Redis 索引不存在，开始写入向量数据并自动创建索引: {len(texts)} 条")
                # 注意：DashScope embedding 对批量大小有限制，这里按小批次写入
                chunk_size = 10
                for start in range(0, len(texts), chunk_size):
                    end = min(start + chunk_size, len(texts))
                    vectorstore.add_texts(
                        texts=texts[start:end],
                        metadatas=metadatas[start:end],
                        batch_size=chunk_size,
                    )
                    if end % 1000 == 0 or end == len(texts):
                        logger.info(f"Redis 向量数据写入进度: {end}/{len(texts)}")
                logger.info("Redis 向量索引/数据初始化完成")
            else:
                logger.warning("content 文档为空，跳过 Redis 向量索引初始化")

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            top_k=10,
        )

        title_index = sparse_retriever.title_index
        rag_service = RAGService(hybrid_retriever, title_index)

        query_understanding_service = QueryUnderstandingService()
        chat_orchestrator = ChatOrchestrator(
            query_understanding=query_understanding_service,
            rag_service=rag_service,
            llm_service=llm_service,
        )

        logger.info("服务初始化完成")

        yield

    finally:
        if database_tool:
            database_tool.close()
        logger.info("应用已关闭")


app = FastAPI(title="健康用药助手API", lifespan=lifespan)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return PlainTextResponse("frontend not found", status_code=404)


@app.post("/auth/register")
async def register(payload: AuthRequest):
    try:
        return auth_service.register(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("用户注册失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"注册失败：{exc}") from exc


@app.post("/auth/login")
async def login(payload: AuthRequest):
    try:
        return auth_service.login(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("用户登录失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"登录失败：{exc}") from exc


@app.get("/chat")
async def chat(
    memory_id: str = Query(..., description="记忆ID"),
    message: str = Query(..., description="用户消息")
):
    """非流式聊天接口"""
    try:
        logger.info(f"开始处理用户问题: {message}")
        response = chat_orchestrator.answer(memory_id, message)
        
        logger.info(f"AI服务处理完成，结果长度: {len(response)}")
        
        return PlainTextResponse(response, media_type="text/plain;charset=UTF-8")
        
    except Exception as e:
        logger.error(f"处理用户问题失败: {e}", exc_info=True)
        return PlainTextResponse(f"抱歉，服务暂时不可用。错误信息：{str(e)}", status_code=500)

@app.get("/chat-stream")
async def chat_stream(
    memory_id: str = Query(..., description="记忆ID"),
    message: str = Query(..., description="用户消息")
):
    """流式聊天接口 - 使用SSE"""
    try:
        logger.info(f"开始处理流式用户问题: {message}")
        
        # 执行RAG检索
        context = rag_service.retrieve_context(message)
        
        # 流式调用LLM
        async def event_generator() -> AsyncIterator[str]:
            async for chunk in llm_service.chat_stream(memory_id, message, context):
                yield chunk
            yield "[DONE]"
        
        return EventSourceResponse(event_generator())
        
    except Exception as e:
        logger.error(f"流式处理异常: {e}", exc_info=True)
        async def error_generator():
            yield f"data: 异常: {str(e)}\n\n"
        return EventSourceResponse(error_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
