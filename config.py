"""配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, *, allow_empty: bool = False, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or (not allow_empty and value == ""):
        if default is not None:
            return default
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def _get_int_env(name: str, *, default: int | None = None) -> int:
    return int(_get_env(name, default=str(default) if default is not None else None))


def _get_bool_env(name: str, *, default: bool | None = None) -> bool:
    default_text = None if default is None else ("true" if default else "false")
    value = _get_env(name, default=default_text).lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"环境变量 {name} 必须是 true/false")


def _get_csv_env(name: str) -> list[str]:
    values = [item.strip() for item in _get_env(name).split(",") if item.strip()]
    if not values:
        raise RuntimeError(f"环境变量 {name} 至少需要一个值")
    return values


# DashScope配置
DASHSCOPE_API_KEY = _get_env("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = _get_env("DASHSCOPE_BASE_URL")
MODEL_NAME = _get_env("MODEL_NAME")
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL")
EMBEDDING_BACKEND = _get_env("EMBEDDING_BACKEND", default="dashscope").strip().lower()
VECTOR_INDEX_NAME = _get_env("VECTOR_INDEX_NAME", default="drug_vectors")
VECTOR_KEY_PREFIX = _get_env("VECTOR_KEY_PREFIX", default="qwen3:")
LOCAL_EMBEDDING_MODEL_PATH = _get_env("LOCAL_EMBEDDING_MODEL_PATH", default="")
LOCAL_EMBEDDING_DEVICE = _get_env("LOCAL_EMBEDDING_DEVICE", default="auto")
LOCAL_EMBEDDING_BATCH_SIZE = _get_int_env("LOCAL_EMBEDDING_BATCH_SIZE", default=8)
LOCAL_EMBEDDING_MAX_LENGTH = _get_int_env("LOCAL_EMBEDDING_MAX_LENGTH", default=512)
LOCAL_EMBEDDING_NORMALIZE = _get_bool_env("LOCAL_EMBEDDING_NORMALIZE", default=True)
LOCAL_EMBEDDING_QUERY_PREFIX = _get_env("LOCAL_EMBEDDING_QUERY_PREFIX", allow_empty=True, default="")
LOCAL_EMBEDDING_DOCUMENT_PREFIX = _get_env("LOCAL_EMBEDDING_DOCUMENT_PREFIX", allow_empty=True, default="")
if EMBEDDING_BACKEND == "local" and not LOCAL_EMBEDDING_MODEL_PATH.strip():
    raise RuntimeError("EMBEDDING_BACKEND=local 时必须配置 LOCAL_EMBEDDING_MODEL_PATH")

# Redis配置
REDIS_HOST = _get_env("REDIS_HOST")
REDIS_PORT = _get_int_env("REDIS_PORT")
REDIS_DB = _get_int_env("REDIS_DB")

# MySQL配置
MYSQL_HOST = _get_env("MYSQL_HOST")
MYSQL_PORT = _get_int_env("MYSQL_PORT")
MYSQL_DATABASE = _get_env("MYSQL_DATABASE")
MYSQL_USER = _get_env("MYSQL_USER")
MYSQL_PASSWORD = _get_env("MYSQL_PASSWORD", allow_empty=True)

# 应用配置
LOG_LEVEL = _get_env("LOG_LEVEL")
LOG_TO_FILE = _get_bool_env("LOG_TO_FILE", default=True)
LOG_DIR = _get_env("LOG_DIR", default="logs")
LOG_FILE_NAME = _get_env("LOG_FILE_NAME", default="app.log")
LOG_MAX_BYTES = _get_int_env("LOG_MAX_BYTES", default=10 * 1024 * 1024)
LOG_BACKUP_COUNT = _get_int_env("LOG_BACKUP_COUNT", default=5)
ALLOWED_ORIGINS = _get_csv_env("ALLOWED_ORIGINS")
CHAT_MEMORY_TTL_SECONDS = _get_int_env("CHAT_MEMORY_TTL_SECONDS")
CHAT_MEMORY_MAX_MESSAGES = _get_int_env("CHAT_MEMORY_MAX_MESSAGES")
CHAT_MEMORY_SUMMARY_ENABLED = _get_bool_env("CHAT_MEMORY_SUMMARY_ENABLED", default=True)
CHAT_MEMORY_SUMMARY_MAX_CHARS = _get_int_env("CHAT_MEMORY_SUMMARY_MAX_CHARS", default=1000)
CHAT_MEMORY_SUMMARY_BATCH_TURNS = _get_int_env("CHAT_MEMORY_SUMMARY_BATCH_TURNS", default=3)
CHAT_MEMORY_SUMMARY_TTL_SECONDS = _get_int_env(
    "CHAT_MEMORY_SUMMARY_TTL_SECONDS",
    default=max(CHAT_MEMORY_TTL_SECONDS, 30 * 24 * 60 * 60),
)
AUTH_TOKEN_SECRET = _get_env("AUTH_TOKEN_SECRET")
AUTH_TOKEN_TTL_SECONDS = _get_int_env("AUTH_TOKEN_TTL_SECONDS")
RAG_MAX_CONTEXT_CHARS = _get_int_env("RAG_MAX_CONTEXT_CHARS")
RAG_MAX_DOC_CHARS = _get_int_env("RAG_MAX_DOC_CHARS")
RAG_FINAL_TOP_K = _get_int_env("RAG_FINAL_TOP_K")
RAG_DENSE_TOP_K = _get_int_env("RAG_DENSE_TOP_K", default=10)
RAG_SPARSE_TOP_K = _get_int_env("RAG_SPARSE_TOP_K", default=10)
GRAPHRAG_ENABLED = _get_bool_env("GRAPHRAG_ENABLED", default=True)
GRAPHRAG_PROJECT_ROOT = _get_env("GRAPHRAG_PROJECT_ROOT", default="drug_kg/graphrag/official_project")
GRAPHRAG_FALLBACK_TO_LEGACY_RAG = _get_bool_env("GRAPHRAG_FALLBACK_TO_LEGACY_RAG", default=False)
CROSS_ENCODER_ENABLED = _get_bool_env("CROSS_ENCODER_ENABLED", default=False)
CROSS_ENCODER_MODEL_PATH = _get_env("CROSS_ENCODER_MODEL_PATH", default="")
CROSS_ENCODER_MAX_LENGTH = _get_int_env("CROSS_ENCODER_MAX_LENGTH", default=512)
CROSS_ENCODER_CANDIDATE_TOP_K = _get_int_env("CROSS_ENCODER_CANDIDATE_TOP_K", default=20)
REWRITE_SHORT_QUERY_CHARS = _get_int_env("REWRITE_SHORT_QUERY_CHARS")
DRUG_ALIAS_FILE = _get_env("DRUG_ALIAS_FILE")
QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK = _get_bool_env("QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK")
LIGHT_INTENT_MODEL_ENABLED = _get_bool_env("LIGHT_INTENT_MODEL_ENABLED")
LIGHT_INTENT_MODEL_PATH = _get_env("LIGHT_INTENT_MODEL_PATH")
LIGHT_INTENT_MODEL_MAX_LENGTH = _get_int_env("LIGHT_INTENT_MODEL_MAX_LENGTH")
MEDICAL_NER_ENABLED = _get_bool_env("MEDICAL_NER_ENABLED")
MEDICAL_NER_MODEL_PATH = _get_env("MEDICAL_NER_MODEL_PATH", default="")
if MEDICAL_NER_ENABLED and not MEDICAL_NER_MODEL_PATH.strip():
    raise RuntimeError("MEDICAL_NER_ENABLED=true 时必须配置 MEDICAL_NER_MODEL_PATH")
LLM_TOOL_CALLS_AVAILABLE = _get_bool_env("LLM_TOOL_CALLS_AVAILABLE", default=True)
LLM_TOOL_CALLS_ENABLED = _get_bool_env("LLM_TOOL_CALLS_ENABLED", default=True)
LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED = _get_bool_env("LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED", default=True)
LLM_TOOL_MAX_ROUNDS = _get_int_env("LLM_TOOL_MAX_ROUNDS", default=3)
