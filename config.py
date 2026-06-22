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
ALLOWED_ORIGINS = _get_csv_env("ALLOWED_ORIGINS")
CHAT_MEMORY_TTL_SECONDS = _get_int_env("CHAT_MEMORY_TTL_SECONDS")
CHAT_MEMORY_MAX_MESSAGES = _get_int_env("CHAT_MEMORY_MAX_MESSAGES")
CHAT_MEMORY_SUMMARY_ENABLED = _get_bool_env("CHAT_MEMORY_SUMMARY_ENABLED", default=True)
CHAT_MEMORY_SUMMARY_MAX_CHARS = _get_int_env("CHAT_MEMORY_SUMMARY_MAX_CHARS", default=1000)
AUTH_TOKEN_SECRET = _get_env("AUTH_TOKEN_SECRET")
AUTH_TOKEN_TTL_SECONDS = _get_int_env("AUTH_TOKEN_TTL_SECONDS")
RAG_MAX_CONTEXT_CHARS = _get_int_env("RAG_MAX_CONTEXT_CHARS")
RAG_MAX_DOC_CHARS = _get_int_env("RAG_MAX_DOC_CHARS")
RAG_FINAL_TOP_K = _get_int_env("RAG_FINAL_TOP_K")
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
MEDICAL_NER_MODEL_PATH = _get_env("MEDICAL_NER_MODEL_PATH")
LLM_TOOL_CALLS_ENABLED = _get_bool_env("LLM_TOOL_CALLS_ENABLED", default=True)
