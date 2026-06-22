"""FastAPI main application."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import dashscope
from dashscope import TextEmbedding
from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import Redis
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import *
from retriever.hybrid_retriever import HybridRetriever
from retriever.sparse_retriever import SparseRetriever
from services.chat_orchestrator import ChatOrchestrator
from services.cross_encoder_reranker import CrossEncoderReranker
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.query_understanding import QueryUnderstandingService
from services.rag_service import RAGService
from services.user_auth_service import AuthError, UserAuthService
from tools.database_tool import DatabaseTool

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
auth_service = UserAuthService()


class AuthRequest(BaseModel):
    username: str
    password: str


class UserMedicationPayload(BaseModel):
    drug_name: str
    dosage: str = ""
    purpose: str = ""
    frequency: str = ""
    times_per_day: Optional[int] = None
    administration_time: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class HealthProfilePayload(BaseModel):
    display_name: str = ""
    gender: str = ""
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medications: List[UserMedicationPayload] = Field(default_factory=list)
    notes: str = ""


class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings wrapper."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        max_batch = 10
        all_vectors: List[List[float]] = []
        for index in range(0, len(texts), max_batch):
            batch = texts[index : index + max_batch]
            response = TextEmbedding.call(model=self.model, input=batch)
            if not response or not getattr(response, "output", None) or "embeddings" not in response.output:
                raise ValueError(
                    f"DashScope TextEmbedding returned invalid response: model={self.model}, batch_size={len(batch)}"
                )
            all_vectors.extend([item["embedding"] for item in response.output["embeddings"]])
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle."""

    global rag_service, llm_service, query_understanding_service, chat_orchestrator, database_tool

    logger.info("正在初始化服务...")
    try:
        dashscope.api_key = DASHSCOPE_API_KEY

        try:
            database_tool = DatabaseTool()
        except Exception as exc:
            database_tool = None
            logger.warning("数据库工具初始化失败，工具调用将返回不可用: %s", exc, exc_info=True)

        tools = []
        if LLM_TOOL_CALLS_ENABLED:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "queryRealDrugDatabase",
                        "description": "查询 real_drug 数据库，获取药物详细信息，包括饮食禁忌、相互作用等。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "drugName": {"type": "string", "description": "药物名称"}
                            },
                            "required": ["drugName"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "queryJointData",
                        "description": "联合查询：先查 yinshi 获取用户用药，再按药物名称查询 real_drug。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string", "description": "查询问题"}
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

        tool_handlers = {}
        if LLM_TOOL_CALLS_ENABLED:
            tool_handlers = {
                "queryRealDrugDatabase": query_real_drug_database,
                "queryJointData": query_joint_data,
            }

        logger.info("LLM 工具调用开关: enabled=%s", LLM_TOOL_CALLS_ENABLED)

        memory_service = MemoryService()
        llm_service = LLMService(tools=tools, tool_handlers=tool_handlers, memory_service=memory_service)

        embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
        redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        vectorstore = Redis(redis_url=redis_url, index_name="drug_vectors", embedding=embeddings)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        sparse_retriever = SparseRetriever.from_content_dir("content", top_k=10)
        try:
            vectorstore.client.ft("drug_vectors").info()  # type: ignore[attr-defined]
        except Exception:
            texts = [doc.page_content for doc in sparse_retriever.documents]
            metadatas = [doc.metadata for doc in sparse_retriever.documents]
            if texts:
                logger.info("检测到 Redis 索引不存在，开始写入向量数据并自动创建索引: %d 条", len(texts))
                chunk_size = 10
                for start in range(0, len(texts), chunk_size):
                    end = min(start + chunk_size, len(texts))
                    vectorstore.add_texts(
                        texts=texts[start:end],
                        metadatas=metadatas[start:end],
                        batch_size=chunk_size,
                    )
                    if end % 1000 == 0 or end == len(texts):
                        logger.info("Redis 向量数据写入进度: %d/%d", end, len(texts))

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            top_k=10,
        )
        rag_service = RAGService(
            hybrid_retriever,
            sparse_retriever.title_index,
            reranker=CrossEncoderReranker(),
        )
        graphrag_service = None
        try:
            from services.graphrag_service import GraphRAGService

            graphrag_service = GraphRAGService(project_root="drug_kg/graphrag/official_project")
            logger.info("官方 GraphRAG 检索已启用")
        except Exception as exc:
            logger.warning("官方 GraphRAG 初始化失败，将回退到旧RAG: %s", exc, exc_info=True)
        query_understanding_service = QueryUnderstandingService()
        chat_orchestrator = ChatOrchestrator(
            query_understanding=query_understanding_service,
            rag_service=rag_service,
            llm_service=llm_service,
            graphrag_service=graphrag_service,
            database_tool=database_tool,
        )

        logger.info("服务初始化完成")
        yield
    finally:
        if database_tool is not None:
            database_tool.close()
        logger.info("应用已关闭")


app = FastAPI(title="健康用药助手 API", lifespan=lifespan)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
STATIC_DIR = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else BASE_DIR / "static"
ASSETS_DIR = STATIC_DIR / "assets"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_bearer_token(authorization: Optional[str]) -> str:
    header = str(authorization or "").strip()
    if not header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少登录凭证")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证格式无效")
    token = header[7:].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少登录凭证")
    return token


def _require_user_payload(authorization: Optional[str]) -> Dict[str, Any]:
    try:
        return auth_service.verify_token(_extract_bearer_token(authorization))
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def _optional_user_id(authorization: Optional[str]) -> Optional[int]:
    if not str(authorization or "").strip():
        return None
    return int(_require_user_payload(authorization)["uid"])


@app.get("/", include_in_schema=False)
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return PlainTextResponse("frontend not found", status_code=404)


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    favicon_file = STATIC_DIR / "favicon.svg"
    if favicon_file.exists():
        return FileResponse(favicon_file)
    return PlainTextResponse("favicon not found", status_code=404)


@app.post("/auth/register")
async def register(payload: AuthRequest):
    try:
        return auth_service.register(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("用户注册失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"注册失败: {exc}") from exc


@app.post("/auth/login")
async def login(payload: AuthRequest):
    try:
        return auth_service.login(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("用户登录失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"登录失败: {exc}") from exc


@app.get("/me/health-profile")
async def get_health_profile(authorization: Optional[str] = Header(default=None)):
    if database_tool is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据库工具当前不可用")
    user_payload = _require_user_payload(authorization)
    try:
        return database_tool.get_user_health_profile(int(user_payload["uid"]))
    except Exception as exc:
        logger.error("读取健康档案失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"读取健康档案失败: {type(exc).__name__}: {exc}") from exc


@app.put("/me/health-profile")
async def update_health_profile(
    payload: HealthProfilePayload,
    authorization: Optional[str] = Header(default=None),
):
    if database_tool is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据库工具当前不可用")
    user_payload = _require_user_payload(authorization)
    try:
        profile_data = payload.model_dump()
        profile_data["medications"] = [item.model_dump() for item in payload.medications]
        return database_tool.upsert_user_health_profile(int(user_payload["uid"]), profile_data)
    except Exception as exc:
        logger.error("更新健康档案失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"更新健康档案失败: {type(exc).__name__}: {exc}") from exc


@app.get("/chat")
async def chat(
    memory_id: str = Query(..., description="记忆 ID"),
    message: str = Query(..., description="用户消息"),
    authorization: Optional[str] = Header(default=None),
):
    try:
        logger.info("开始处理用户问题: %s", message)
        response = chat_orchestrator.answer(memory_id, message, user_id=_optional_user_id(authorization))
        logger.info("AI 服务处理完成，结果长度: %d", len(response))
        return PlainTextResponse(response, media_type="text/plain;charset=UTF-8")
    except Exception as exc:
        logger.error("处理用户问题失败: %s", exc, exc_info=True)
        return PlainTextResponse(f"抱歉，服务暂时不可用。错误信息：{exc}", status_code=500)


@app.get("/chat-stream")
async def chat_stream(
    memory_id: str = Query(..., description="记忆 ID"),
    message: str = Query(..., description="用户消息"),
    authorization: Optional[str] = Header(default=None),
):
    try:
        logger.info("开始处理流式用户问题: %s", message)
        user_id = _optional_user_id(authorization)

        async def event_generator() -> AsyncIterator[str]:
            async for chunk in chat_orchestrator.answer_stream(memory_id, message, user_id=user_id):
                yield json.dumps({"data": chunk}, ensure_ascii=False)
            yield json.dumps({"done": True}, ensure_ascii=False)

        return EventSourceResponse(event_generator())
    except Exception as exc:
        logger.error("流式处理异常: %s", exc, exc_info=True)

        async def error_generator():
            yield json.dumps({"error": str(exc)}, ensure_ascii=False)

        return EventSourceResponse(error_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
