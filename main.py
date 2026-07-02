"""FastAPI main application."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import dashscope
from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import *
from services.embedding_factory import DashScopeEmbeddings, build_embeddings, embedding_backend_summary
from services.chat_orchestrator import ChatOrchestrator
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.query_understanding import QueryUnderstandingService
from services.user_auth_service import AuthError, UserAuthService
from tools.database_tool import DatabaseTool

def _configure_logging() -> Path | None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    log_file_path: Path | None = None

    if LOG_TO_FILE:
        log_dir = Path(LOG_DIR)
        if not log_dir.is_absolute():
            log_dir = Path(__file__).resolve().parent / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / LOG_FILE_NAME
        handlers.append(
            RotatingFileHandler(
                log_file_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file_path


LOG_FILE_PATH = _configure_logging()
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


class ChatResponsePayload(BaseModel):
    answer: str
    meta: Dict[str, Any]


class ChatHistoryMessagePayload(BaseModel):
    role: str
    content: str
    meta: Optional[Dict[str, Any]] = None


class ChatHistoryPayload(BaseModel):
    memory_id: str
    messages: List[ChatHistoryMessagePayload]
    summary: str


class ChatHistoryClearPayload(BaseModel):
    memory_id: str
    cleared: bool


class ChatContextPayload(BaseModel):
    memory_id: str
    user_logged_in: bool
    profile_available: bool
    memory_available: bool
    effective_context: Dict[str, Any]
    effective_context_text: str


class SessionPayload(BaseModel):
    title: str = ""


class SessionPatchPayload(BaseModel):
    title: str


class ChatSessionItemPayload(BaseModel):
    id: str
    title: str
    preview: str
    updated_at: int


class ChatSessionListPayload(BaseModel):
    sessions: List[ChatSessionItemPayload]


class RuntimeStatusPayload(BaseModel):
    tool_calls_available: bool
    tool_calls_enabled: bool
    tool_calls_runtime_override_enabled: bool
    available_tools: List[str]
    database: Dict[str, Any]
    retrieval: Dict[str, Any]
    memory: Dict[str, Any]
    auth: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle."""

    global rag_service, llm_service, query_understanding_service, chat_orchestrator, database_tool

    logger.info("正在初始化服务...")
    if LOG_FILE_PATH is not None:
        logger.info("应用日志已落盘: path=%s", LOG_FILE_PATH)
    try:
        dashscope.api_key = DASHSCOPE_API_KEY

        try:
            database_tool = DatabaseTool()
        except Exception as exc:
            database_tool = None
            logger.warning("数据库工具初始化失败，工具调用将返回不可用: %s", exc, exc_info=True)

        tools = []
        database_capabilities = database_tool.get_capabilities() if database_tool is not None else {}
        if LLM_TOOL_CALLS_AVAILABLE:
            if database_capabilities.get("user_health_profile_table") or database_capabilities.get("user_medications_table"):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "queryUserHealthProfile",
                            "description": "读取当前登录用户的健康档案，不包含当前用药，用于个体化问答。",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )
            if database_capabilities.get("user_medications_table"):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "queryUserMedicationSummary",
                            "description": "读取当前登录用户登记的当前用药摘要，用于回答“我现在在吃什么药”或结合当前用药评估风险。",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )

        def query_user_health_profile(user_id: int) -> str:
            if database_tool is None:
                return "数据库工具当前不可用，无法读取健康档案。"
            return database_tool.query_user_health_profile(user_id)

        def query_user_medication_summary(user_id: int) -> str:
            if database_tool is None:
                return "数据库工具当前不可用，无法读取当前用药。"
            return database_tool.query_user_medication_summary(user_id)

        tool_handlers = {}
        if LLM_TOOL_CALLS_AVAILABLE:
            tool_handlers = {
                "queryUserHealthProfile": query_user_health_profile,
                "queryUserMedicationSummary": query_user_medication_summary,
            }

        logger.info(
            "LLM 工具调用开关: enabled=%s available_tools=%s db_capabilities=%s",
            LLM_TOOL_CALLS_ENABLED,
            [item.get("function", {}).get("name") for item in tools],
            database_capabilities,
        )

        memory_service = MemoryService()
        llm_service = LLMService(tools=tools, tool_handlers=tool_handlers, memory_service=memory_service)

        graphrag_service = None
        rag_service = None
        if GRAPHRAG_ENABLED:
            try:
                from services.graphrag_service import GraphRAGService

                graphrag_service = GraphRAGService(project_root=GRAPHRAG_PROJECT_ROOT)
                logger.info("官方 GraphRAG 检索已启用，跳过旧向量检索和 BM25 初始化")
            except Exception as exc:
                if not GRAPHRAG_FALLBACK_TO_LEGACY_RAG:
                    raise RuntimeError(
                        "GraphRAG 初始化失败，且 GRAPHRAG_FALLBACK_TO_LEGACY_RAG=false，"
                        "不会回退到旧向量检索/BM25"
                    ) from exc
                logger.warning("官方 GraphRAG 初始化失败，将按配置回退到旧RAG: %s", exc, exc_info=True)

        if graphrag_service is None:
            from langchain_community.vectorstores import Redis

            from retriever.hybrid_retriever import HybridRetriever
            from retriever.sparse_retriever import SparseRetriever
            from services.cross_encoder_reranker import CrossEncoderReranker
            from services.rag_service import RAGService

            logger.info("初始化旧RAG：Redis 向量检索 + BM25")
            logger.info("Embedding 配置: %s", embedding_backend_summary())
            embeddings = build_embeddings()
            redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
            vectorstore = Redis(redis_url=redis_url, index_name=VECTOR_INDEX_NAME, embedding=embeddings)
            dense_retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_DENSE_TOP_K})

            unified_chunks_path = str(BASE_DIR / "content" / "unified_chunks.jsonl")
            sparse_retriever = SparseRetriever.from_jsonl(unified_chunks_path, top_k=RAG_SPARSE_TOP_K)
            try:
                vectorstore.client.ft(VECTOR_INDEX_NAME).info()  # type: ignore[attr-defined]
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

            hybrid_top_k = max(
                10,
                RAG_FINAL_TOP_K,
                CROSS_ENCODER_CANDIDATE_TOP_K if CROSS_ENCODER_ENABLED else 10,
            )
            logger.info(
                "旧RAG检索参数: dense_top_k=%d sparse_top_k=%d hybrid_top_k=%d rerank_enabled=%s rerank_candidate_top_k=%d",
                RAG_DENSE_TOP_K,
                RAG_SPARSE_TOP_K,
                hybrid_top_k,
                CROSS_ENCODER_ENABLED,
                CROSS_ENCODER_CANDIDATE_TOP_K,
            )
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                top_k=hybrid_top_k,
            )
            rag_service = RAGService(
                hybrid_retriever,
                sparse_retriever.title_index,
                reranker=CrossEncoderReranker(),
            )

        query_understanding_service = QueryUnderstandingService(database_tool=database_tool)
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
STATIC_DIR = FRONTEND_DIST_DIR
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


def _require_memory_access(memory_id: str, authorization: Optional[str]) -> None:
    text = str(memory_id or "").strip()
    if text.startswith("user_id_"):
        user_payload = _require_user_payload(authorization)
        expected = f"user_id_{int(user_payload['uid'])}"
        if text != expected and not text.startswith(f"{expected}_session_"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该会话")


def _get_memory_service() -> MemoryService:
    service = getattr(chat_orchestrator, "llm_service", None)
    memory_service = getattr(service, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="记忆服务当前不可用")
    return memory_service


def _persist_last_assistant_meta(memory_id: str, meta: Dict[str, Any]) -> None:
    try:
        _get_memory_service().update_last_assistant_meta(memory_id, meta)
    except Exception as exc:
        logger.warning("更新会话元数据失败: memory_id=%s error=%s", memory_id, exc)


def _sync_user_profile_cache(user_id: int, profile: Dict[str, Any]) -> None:
    if database_tool is None:
        return
    try:
        renderer = getattr(database_tool, "render_profile_context", None)
        if callable(renderer):
            context = renderer(profile)
        else:
            lines = ["[用户个人档案]"]
            if profile.get("display_name"):
                lines.append("称呼: " + str(profile["display_name"]))
            if profile.get("conditions"):
                lines.append("基础病: " + "、".join([str(item).strip() for item in profile.get("conditions") or [] if str(item).strip()]))
            if profile.get("allergies"):
                lines.append("过敏史: " + "、".join([str(item).strip() for item in profile.get("allergies") or [] if str(item).strip()]))
            if profile.get("notes"):
                lines.append("备注: " + str(profile["notes"]))
            context = "" if len(lines) == 1 else "\n".join(lines)
        _get_memory_service().set_user_profile_cache(int(user_id), profile, context)
    except Exception as exc:
        logger.warning("同步用户个人档案缓存失败: user_id=%s error=%s", user_id, exc)


def _session_scope_for_user(user_id: int) -> str:
    return f"user:{int(user_id)}"


def _validate_user_session_id(user_id: int, session_id: str) -> str:
    text = str(session_id or "").strip()
    base = f"user_id_{int(user_id)}"
    if text == base or text.startswith(f"{base}_session_"):
        return text
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="会话 ID 不属于当前用户")


def _runtime_status_snapshot() -> RuntimeStatusPayload:
    llm = getattr(chat_orchestrator, "llm_service", None)
    available_tools = []
    if llm is not None:
        available_tools = sorted(
            [
                str(item.get("function", {}).get("name") or "").strip()
                for item in getattr(llm, "tools", []) or []
                if str(item.get("function", {}).get("name") or "").strip()
            ]
        )
    db_capabilities = database_tool.get_capabilities() if database_tool is not None else {}
    graphrag_ready = bool(getattr(chat_orchestrator, "graphrag_service", None))
    legacy_rag_ready = bool(getattr(chat_orchestrator, "rag_service", None))
    primary_backend = "graphrag" if graphrag_ready else ("legacy_rag" if legacy_rag_ready else "none")
    return RuntimeStatusPayload(
        tool_calls_available=bool(LLM_TOOL_CALLS_AVAILABLE),
        tool_calls_enabled=bool(LLM_TOOL_CALLS_ENABLED),
        tool_calls_runtime_override_enabled=bool(LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED),
        available_tools=available_tools,
        database={
            "available": database_tool is not None,
            "capabilities": db_capabilities,
        },
        retrieval={
            "graphrag_enabled": bool(GRAPHRAG_ENABLED),
            "graphrag_ready": graphrag_ready,
            "legacy_rag_ready": legacy_rag_ready,
            "primary_backend": primary_backend,
        },
        memory={
            "available": llm is not None and getattr(llm, "memory_service", None) is not None,
            "summary_enabled": bool(CHAT_MEMORY_SUMMARY_ENABLED),
            "backend": "redis",
        },
        auth={
            "available": auth_service is not None,
            "token_ttl_seconds": int(AUTH_TOKEN_TTL_SECONDS),
        },
    )


def _resolve_tool_policy(tool_policy: str) -> tuple[str, bool]:
    raw_value = tool_policy
    if not isinstance(raw_value, str):
        raw_value = getattr(raw_value, "default", "default")
    normalized = str(raw_value or "default").strip().lower()
    if normalized not in {"default", "force_on", "force_off"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="tool_policy 必须是 default、force_on 或 force_off")
    if not LLM_TOOL_CALLS_AVAILABLE:
        return normalized, False
    if normalized == "force_off":
        return normalized, False
    if normalized == "force_on":
        return normalized, bool(LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED)
    return normalized, bool(LLM_TOOL_CALLS_ENABLED)


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
        profile = database_tool.get_user_health_profile(int(user_payload["uid"]))
        _sync_user_profile_cache(int(user_payload["uid"]), profile)
        return profile
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
        profile = database_tool.upsert_user_health_profile(int(user_payload["uid"]), profile_data)
        _sync_user_profile_cache(int(user_payload["uid"]), profile)
        return profile
    except Exception as exc:
        logger.error("更新健康档案失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"更新健康档案失败: {type(exc).__name__}: {exc}") from exc


@app.get("/me/chat-sessions")
async def list_chat_sessions(authorization: Optional[str] = Header(default=None)):
    user_payload = _require_user_payload(authorization)
    memory_service = _get_memory_service()
    sessions = memory_service.list_sessions(_session_scope_for_user(int(user_payload["uid"])))
    return ChatSessionListPayload(sessions=[ChatSessionItemPayload(**item) for item in sessions])


@app.post("/me/chat-sessions")
async def create_chat_session(payload: SessionPayload, authorization: Optional[str] = Header(default=None)):
    user_payload = _require_user_payload(authorization)
    memory_service = _get_memory_service()
    user_id = int(user_payload["uid"])
    session_id = f"user_id_{user_id}_session_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    session = memory_service.upsert_session(
        _session_scope_for_user(user_id),
        {
            "id": session_id,
            "title": str(payload.title or "").strip() or "新会话",
            "preview": "",
            "updated_at": int(datetime.now(timezone.utc).timestamp() * 1000),
        },
    )
    return ChatSessionItemPayload(**session)


@app.patch("/me/chat-sessions/{session_id}")
async def rename_chat_session(
    session_id: str,
    payload: SessionPatchPayload,
    authorization: Optional[str] = Header(default=None),
):
    user_payload = _require_user_payload(authorization)
    memory_service = _get_memory_service()
    user_id = int(user_payload["uid"])
    validated_session_id = _validate_user_session_id(user_id, session_id)
    renamed = memory_service.rename_session(_session_scope_for_user(user_id), validated_session_id, payload.title)
    if renamed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return ChatSessionItemPayload(**renamed)


@app.delete("/me/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str, authorization: Optional[str] = Header(default=None)):
    user_payload = _require_user_payload(authorization)
    memory_service = _get_memory_service()
    user_id = int(user_payload["uid"])
    validated_session_id = _validate_user_session_id(user_id, session_id)
    deleted = memory_service.delete_session(_session_scope_for_user(user_id), validated_session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return {"deleted": True, "session_id": validated_session_id}


@app.get("/chat-history")
async def get_chat_history(
    memory_id: str = Query(..., description="记忆 ID"),
    turns: int = Query(20, ge=1, le=100, description="返回最近多少轮对话"),
    authorization: Optional[str] = Header(default=None),
):
    _require_memory_access(memory_id, authorization)
    memory_service = _get_memory_service()
    return ChatHistoryPayload(
        memory_id=memory_id,
        messages=[ChatHistoryMessagePayload(**item) for item in memory_service.export_history(memory_id, turns=turns)],
        summary=memory_service.get_summary(memory_id),
    )


@app.delete("/chat-history")
async def clear_chat_history(
    memory_id: str = Query(..., description="记忆 ID"),
    authorization: Optional[str] = Header(default=None),
):
    _require_memory_access(memory_id, authorization)
    memory_service = _get_memory_service()
    memory_service.clear_memory(memory_id)
    return ChatHistoryClearPayload(memory_id=memory_id, cleared=True)


@app.get("/chat-context")
async def get_chat_context(
    memory_id: str = Query(..., description="记忆 ID"),
    authorization: Optional[str] = Header(default=None),
):
    _require_memory_access(memory_id, authorization)
    payload = chat_orchestrator.build_context_snapshot(memory_id, user_id=_optional_user_id(authorization))
    return ChatContextPayload(**payload)


@app.get("/system/runtime-status")
async def get_runtime_status():
    return _runtime_status_snapshot()


@app.get("/chat")
async def chat(
    memory_id: str = Query(..., description="记忆 ID"),
    message: str = Query(..., description="用户消息"),
    include_meta: bool = Query(False, description="是否返回元数据 JSON"),
    tool_policy: str = Query("default", description="工具调用策略：default|force_on|force_off"),
    authorization: Optional[str] = Header(default=None),
):
    try:
        _require_memory_access(memory_id, authorization)
        normalized_tool_policy, tool_calls_enabled = _resolve_tool_policy(tool_policy)
        current_user_id = _optional_user_id(authorization)
        if current_user_id is not None:
            _get_memory_service().touch_session(
                _session_scope_for_user(current_user_id),
                {
                    "id": _validate_user_session_id(current_user_id, memory_id),
                    "title": str(message or "").strip()[:18] or "最近会话",
                    "preview": str(message or "").strip()[:80],
                    "updated_at": int(datetime.now(timezone.utc).timestamp() * 1000),
                },
            )
        logger.info("开始处理用户问题: %s", message)
        response, meta = chat_orchestrator.answer_with_meta(
            memory_id,
            message,
            user_id=current_user_id,
            tool_calls_enabled=tool_calls_enabled,
        )
        meta["tool_policy"] = normalized_tool_policy
        _persist_last_assistant_meta(memory_id, meta)
        logger.info("AI 服务处理完成，结果长度: %d", len(response))
        if include_meta:
            return ChatResponsePayload(answer=response, meta=meta)
        return PlainTextResponse(response, media_type="text/plain;charset=UTF-8")
    except Exception as exc:
        logger.error("处理用户问题失败: %s", exc, exc_info=True)
        return PlainTextResponse(f"抱歉，服务暂时不可用。错误信息：{exc}", status_code=500)


@app.get("/chat-stream")
async def chat_stream(
    memory_id: str = Query(..., description="记忆 ID"),
    message: str = Query(..., description="用户消息"),
    tool_policy: str = Query("default", description="工具调用策略：default|force_on|force_off"),
    authorization: Optional[str] = Header(default=None),
):
    try:
        _require_memory_access(memory_id, authorization)
        logger.info("开始处理流式用户问题: %s", message)
        user_id = _optional_user_id(authorization)
        normalized_tool_policy, tool_calls_enabled = _resolve_tool_policy(tool_policy)
        if user_id is not None:
            _get_memory_service().touch_session(
                _session_scope_for_user(user_id),
                {
                    "id": _validate_user_session_id(user_id, memory_id),
                    "title": str(message or "").strip()[:18] or "最近会话",
                    "preview": str(message or "").strip()[:80],
                    "updated_at": int(datetime.now(timezone.utc).timestamp() * 1000),
                },
            )
        prepared = chat_orchestrator.prepare_turn(
            memory_id,
            message,
            user_id=user_id,
            tool_calls_enabled=tool_calls_enabled,
        )
        meta = prepared["meta"]
        meta["tool_policy"] = normalized_tool_policy

        async def event_generator() -> AsyncIterator[str]:
            yield json.dumps({"meta": meta}, ensure_ascii=False)
            async for chunk in chat_orchestrator.answer_stream_prepared(
                prepared,
                memory_id,
                message,
                user_id=user_id,
                tool_calls_enabled=tool_calls_enabled,
            ):
                yield json.dumps({"data": chunk}, ensure_ascii=False)
            final_tooling = chat_orchestrator.llm_service.get_last_run_metadata()
            if final_tooling:
                final_meta = dict(meta)
                final_meta["tooling"] = final_tooling
                _persist_last_assistant_meta(memory_id, final_meta)
                yield json.dumps({"meta_update": final_meta}, ensure_ascii=False)
            else:
                _persist_last_assistant_meta(memory_id, meta)
            yield json.dumps({"done": True}, ensure_ascii=False)

        return EventSourceResponse(event_generator())
    except Exception as exc:
        logger.error("流式处理异常: %s", exc, exc_info=True)
        error_text = str(exc)

        async def error_generator():
            yield json.dumps({"error": error_text}, ensure_ascii=False)

        return EventSourceResponse(error_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, loop="asyncio")
