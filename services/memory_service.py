"""Chat memory service."""
from __future__ import annotations

import json
import logging
from typing import Dict, List

import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, message_to_dict, messages_from_dict

from config import CHAT_MEMORY_MAX_MESSAGES, CHAT_MEMORY_TTL_SECONDS, REDIS_DB, REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


class MemoryService:
    """Store chat history in Redis."""

    def __init__(self):
        self.redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        self.memory_window = max(1, CHAT_MEMORY_MAX_MESSAGES // 2)
        self.redis_client_kwargs = {
            "socket_connect_timeout": 1,
            "socket_timeout": 1,
            "retry_on_timeout": False,
        }
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            **self.redis_client_kwargs,
        )

    def get_history(self, memory_id: str) -> List[BaseMessage]:
        try:
            raw_messages = self.client.lrange(self._key(memory_id), -self.memory_window * 2, -1)
            return messages_from_dict([json.loads(item) for item in raw_messages])
        except Exception as exc:
            logger.warning("read chat memory failed, skipped history: %s", exc)
            return []

    def get_recent_turns(self, memory_id: str, turns: int = 5) -> List[Dict[str, str]]:
        messages = self.get_history(memory_id)[-max(1, turns) * 2 :]
        history: List[Dict[str, str]] = []
        for message in messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            history.append({"role": role, "content": str(message.content or "")})
        return history

    def append_exchange(self, memory_id: str, user_message: str, assistant_message: str) -> None:
        try:
            key = self._key(memory_id)
            payloads = [
                json.dumps(message_to_dict(HumanMessage(content=user_message)), ensure_ascii=False),
                json.dumps(message_to_dict(AIMessage(content=assistant_message)), ensure_ascii=False),
            ]
            pipe = self.client.pipeline(transaction=False)
            pipe.rpush(key, *payloads)
            pipe.ltrim(key, -self.memory_window * 2, -1)
            pipe.expire(key, CHAT_MEMORY_TTL_SECONDS)
            pipe.execute()
        except Exception as exc:
            logger.warning("write chat memory failed, skipped save: %s", exc)

    def _trim(self, memory_id: str) -> None:
        try:
            self.client.ltrim(self._key(memory_id), -self.memory_window * 2, -1)
        except Exception as exc:
            logger.warning("trim chat memory failed: %s", exc)

    @staticmethod
    def _key(memory_id: str) -> str:
        return f"chat_memory:{memory_id}"
