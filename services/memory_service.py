"""Chat memory service."""
from __future__ import annotations

import json
import logging
import re
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List

import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, message_to_dict, messages_from_dict

from config import (
    CHAT_MEMORY_MAX_MESSAGES,
    CHAT_MEMORY_SUMMARY_TTL_SECONDS,
    CHAT_MEMORY_TTL_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """Store chat history in Redis."""

    EFFECTIVE_SNAPSHOT_LIST_FIELDS = (
        "conditions",
        "allergies",
    )

    def __init__(self):
        self.redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        self.memory_window = max(1, CHAT_MEMORY_MAX_MESSAGES // 2)
        self.max_messages = max(2, CHAT_MEMORY_MAX_MESSAGES)
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

    def get_all_history(self, memory_id: str) -> List[BaseMessage]:
        try:
            raw_messages = self.client.lrange(self._key(memory_id), 0, -1)
            return messages_from_dict([json.loads(item) for item in raw_messages])
        except Exception as exc:
            logger.warning("read full chat memory failed, skipped history: %s", exc)
            return []

    def get_overflow_history(self, memory_id: str, keep_turns: int) -> List[BaseMessage]:
        keep_messages = max(1, keep_turns) * 2
        messages = self.get_all_history(memory_id)
        if len(messages) <= keep_messages:
            return []
        return messages[:-keep_messages]

    def get_summary(self, memory_id: str) -> str:
        try:
            return str(self.client.get(self._summary_key(memory_id)) or "").strip()
        except Exception as exc:
            logger.warning("read chat memory summary failed: %s", exc)
            return ""

    def set_summary(self, memory_id: str, summary: str) -> None:
        try:
            text = str(summary or "").strip()
            if not text:
                return
            self.client.setex(self._summary_key(memory_id), CHAT_MEMORY_SUMMARY_TTL_SECONDS, text)
        except Exception as exc:
            logger.warning("write chat memory summary failed: %s", exc)

    def get_user_profile_cache(self, user_id: int) -> Dict[str, Any]:
        try:
            raw = self.client.get(self._user_profile_key(user_id))
            if not raw:
                return {"profile": {}, "context": ""}
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"profile": {}, "context": ""}
            profile = parsed.get("profile")
            context = parsed.get("context")
            normalized_profile = self._profile_cache_payload(profile if isinstance(profile, dict) else {})
            return {
                "profile": normalized_profile,
                "context": str(context or "").strip(),
            }
        except Exception as exc:
            logger.warning("read user profile cache failed: user_id=%s error=%s", user_id, exc)
            return {"profile": {}, "context": ""}

    def set_user_profile_cache(self, user_id: int, profile: Dict[str, Any], context: str) -> None:
        try:
            payload = {
                "profile": self._json_safe(self._profile_cache_payload(profile if isinstance(profile, dict) else {})),
                "context": str(context or "").strip(),
            }
            self.client.setex(
                self._user_profile_key(user_id),
                CHAT_MEMORY_SUMMARY_TTL_SECONDS,
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("write user profile cache failed: user_id=%s error=%s", user_id, exc)

    def invalidate_user_profile_cache(self, user_id: int) -> None:
        try:
            self.client.delete(self._user_profile_key(user_id))
        except Exception as exc:
            logger.warning("invalidate user profile cache failed: user_id=%s error=%s", user_id, exc)

    @classmethod
    def build_effective_context_snapshot(
        cls,
        personal_profile: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        profile = personal_profile or {}
        conditions_from_profile = cls._merge_unique(profile.get("conditions"))
        allergies_from_profile = cls._merge_unique(profile.get("allergies"))

        snapshot = {
            "display_name": str(profile.get("display_name") or "").strip(),
            "gender": str(profile.get("gender") or "").strip(),
            "age": profile.get("age"),
            "height_cm": profile.get("height_cm"),
            "weight_kg": profile.get("weight_kg"),
            "is_pregnant": bool(profile.get("is_pregnant")),
            "is_breastfeeding": bool(profile.get("is_breastfeeding")),
            "conditions": conditions_from_profile,
            "allergies": allergies_from_profile,
            "notes": str(profile.get("notes") or "").strip(),
            "sources": {
                "conditions": "profile" if conditions_from_profile else "none",
                "allergies": "profile" if allergies_from_profile else "none",
                "notes": "profile" if str(profile.get("notes") or "").strip() else "none",
            },
        }
        return snapshot

    @classmethod
    def format_effective_context_snapshot(cls, snapshot: Dict[str, Any]) -> str:
        if not snapshot:
            return ""
        has_content = any(snapshot.get(field) for field in cls.EFFECTIVE_SNAPSHOT_LIST_FIELDS) or any(
            snapshot.get(field) for field in ("display_name", "gender", "age", "height_cm", "weight_kg", "notes")
        )
        if not has_content:
            return ""

        lines = ["[系统整理的当前用户档案] 以下信息来自登录用户的健康档案。"]
        if snapshot.get("display_name"):
            lines.append("称呼: " + str(snapshot["display_name"]))
        if snapshot.get("gender"):
            lines.append("性别: " + str(snapshot["gender"]))
        if snapshot.get("age") is not None:
            lines.append("年龄: " + str(snapshot["age"]))
        if snapshot.get("height_cm") is not None:
            lines.append("身高(cm): " + str(snapshot["height_cm"]))
        if snapshot.get("weight_kg") is not None:
            lines.append("体重(kg): " + str(snapshot["weight_kg"]))
        if snapshot.get("is_pregnant"):
            lines.append("状态: 妊娠期")
        if snapshot.get("is_breastfeeding"):
            lines.append("状态: 哺乳期")
        if snapshot.get("conditions"):
            lines.append(f"基础病(来源: {cls._source_label(snapshot, 'conditions')}): " + "、".join(snapshot["conditions"]))
        if snapshot.get("allergies"):
            lines.append(f"过敏史(来源: {cls._source_label(snapshot, 'allergies')}): " + "、".join(snapshot["allergies"]))
        if snapshot.get("notes"):
            lines.append("用户档案备注: " + str(snapshot["notes"]))
        return "\n".join(lines)

    def get_recent_turns(self, memory_id: str, turns: int = 5) -> List[Dict[str, str]]:
        messages = self.get_history(memory_id)[-max(1, turns) * 2 :]
        history: List[Dict[str, str]] = []
        for message in messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            history.append({"role": role, "content": str(message.content or "")})
        return history

    def export_history(self, memory_id: str, turns: int = 20) -> List[Dict[str, Any]]:
        messages = self.get_all_history(memory_id)[-max(1, turns) * 2 :]
        history: List[Dict[str, Any]] = []
        for message in messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            content = str(message.content or "").strip()
            if content:
                item: Dict[str, Any] = {"role": role, "content": content}
                if isinstance(message, AIMessage):
                    meta = message.additional_kwargs.get("meta")
                    if isinstance(meta, dict) and meta:
                        item["meta"] = meta
                history.append(item)
        return history

    def update_last_assistant_meta(self, memory_id: str, meta: Dict[str, Any]) -> bool:
        if not isinstance(meta, dict) or not meta:
            return False
        try:
            messages = self.get_all_history(memory_id)
            updated = False
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if not isinstance(message, AIMessage):
                    continue
                merged_kwargs = dict(message.additional_kwargs or {})
                merged_kwargs["meta"] = meta
                messages[index] = AIMessage(
                    content=message.content,
                    additional_kwargs=merged_kwargs,
                )
                updated = True
                break
            if not updated:
                return False
            self._rewrite_history(memory_id, messages)
            return True
        except Exception as exc:
            logger.warning("update assistant meta failed: %s", exc)
            return False

    def list_sessions(self, scope_id: str) -> List[Dict[str, Any]]:
        try:
            raw = self.client.get(self._session_index_key(scope_id))
            if not raw:
                return []
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []
            sessions = [self._normalize_session_item(item) for item in parsed if isinstance(item, dict)]
            return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)
        except Exception as exc:
            logger.warning("read session index failed: %s", exc)
            return []

    def upsert_session(self, scope_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._normalize_session_item(payload)
        sessions = self.list_sessions(scope_id)
        remaining = [item for item in sessions if item["id"] != session["id"]]
        merged = [session, *remaining]
        self._write_sessions(scope_id, merged)
        return session

    def touch_session(self, scope_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._normalize_session_item(payload)
        sessions = self.list_sessions(scope_id)
        existing = next((item for item in sessions if item["id"] == session["id"]), None)
        if existing is not None:
            session["title"] = session["title"] if self._is_default_session_title(existing.get("title", "")) else existing["title"]
        remaining = [item for item in sessions if item["id"] != session["id"]]
        merged = [session, *remaining]
        self._write_sessions(scope_id, merged)
        return session

    def rename_session(self, scope_id: str, session_id: str, title: str) -> Dict[str, Any] | None:
        sessions = self.list_sessions(scope_id)
        target = None
        updated: List[Dict[str, Any]] = []
        for item in sessions:
            if item["id"] == session_id:
                target = dict(item)
                target["title"] = str(title or "").strip()[:32] or item["title"]
                target["updated_at"] = max(item["updated_at"], int(datetime.now(timezone.utc).timestamp() * 1000))
                updated.append(target)
            else:
                updated.append(item)
        if target is None:
            return None
        self._write_sessions(scope_id, updated)
        return target

    def delete_session(self, scope_id: str, session_id: str) -> bool:
        sessions = self.list_sessions(scope_id)
        remaining = [item for item in sessions if item["id"] != session_id]
        if len(remaining) == len(sessions):
            return False
        self._write_sessions(scope_id, remaining)
        self.clear_memory(session_id)
        return True

    def clear_memory(self, memory_id: str, *, clear_summary: bool = True) -> None:
        keys = [self._key(memory_id)]
        if clear_summary:
            keys.append(self._summary_key(memory_id))
        try:
            self.client.delete(*keys)
        except Exception as exc:
            logger.warning("clear chat memory failed: %s", exc)

    def append_exchange(
        self,
        memory_id: str,
        user_message: str,
        assistant_message: str,
        assistant_meta: Dict[str, Any] | None = None,
    ) -> None:
        try:
            key = self._key(memory_id)
            ai_kwargs = {}
            if isinstance(assistant_meta, dict) and assistant_meta:
                ai_kwargs["meta"] = assistant_meta
            payloads = [
                json.dumps(message_to_dict(HumanMessage(content=user_message)), ensure_ascii=False),
                json.dumps(message_to_dict(AIMessage(content=assistant_message, additional_kwargs=ai_kwargs)), ensure_ascii=False),
            ]
            pipe = self.client.pipeline(transaction=False)
            pipe.rpush(key, *payloads)
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, CHAT_MEMORY_TTL_SECONDS)
            pipe.execute()
        except Exception as exc:
            logger.warning("write chat memory failed, skipped save: %s", exc)

    def trim_to_recent_turns(self, memory_id: str, turns: int) -> None:
        try:
            self.client.ltrim(self._key(memory_id), -max(1, turns) * 2, -1)
            self.client.expire(self._key(memory_id), CHAT_MEMORY_TTL_SECONDS)
        except Exception as exc:
            logger.warning("trim chat memory to recent turns failed: %s", exc)

    def _trim(self, memory_id: str) -> None:
        try:
            self.client.ltrim(self._key(memory_id), -self.memory_window * 2, -1)
        except Exception as exc:
            logger.warning("trim chat memory failed: %s", exc)

    def _rewrite_history(self, memory_id: str, messages: List[BaseMessage]) -> None:
        key = self._key(memory_id)
        payloads = [json.dumps(message_to_dict(message), ensure_ascii=False) for message in messages]
        pipe = self.client.pipeline(transaction=False)
        pipe.delete(key)
        if payloads:
            pipe.rpush(key, *payloads)
            pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, CHAT_MEMORY_TTL_SECONDS)
        pipe.execute()

    @staticmethod
    def _key(memory_id: str) -> str:
        return f"chat_memory:{memory_id}"

    @staticmethod
    def _summary_key(memory_id: str) -> str:
        return f"chat_memory_summary:{memory_id}"

    @staticmethod
    def _session_index_key(scope_id: str) -> str:
        return f"chat_sessions:{scope_id}"

    @staticmethod
    def _user_profile_key(user_id: int) -> str:
        return f"user_profile_context:{int(user_id)}"

    @staticmethod
    def _normalize_session_item(payload: Dict[str, Any]) -> Dict[str, Any]:
        title = str(payload.get("title") or "").strip()[:32] or "最近会话"
        preview = str(payload.get("preview") or "").strip()[:80]
        updated_at = int(payload.get("updated_at") or int(datetime.now(timezone.utc).timestamp() * 1000))
        return {
            "id": str(payload.get("id") or "").strip(),
            "title": title,
            "preview": preview,
            "updated_at": updated_at,
        }

    def _write_sessions(self, scope_id: str, sessions: List[Dict[str, Any]]) -> None:
        try:
            normalized = [self._normalize_session_item(item) for item in sessions if str(item.get("id") or "").strip()]
            self.client.setex(
                self._session_index_key(scope_id),
                CHAT_MEMORY_SUMMARY_TTL_SECONDS,
                json.dumps(normalized[:100], ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("write session index failed: %s", exc)

    @staticmethod
    def _is_default_session_title(title: str) -> bool:
        text = str(title or "").strip()
        return text == "最近会话" or bool(re.fullmatch(r"新会话(?:\s+\d+)?", text))

    @staticmethod
    def _source_label(snapshot: Dict[str, Any], field: str) -> str:
        source = str((snapshot.get("sources") or {}).get(field) or "none").strip()
        mapping = {
            "profile": "用户档案",
            "memory": "对话记忆",
            "none": "未提供",
        }
        return mapping.get(source, source)

    @classmethod
    def _profile_cache_payload(cls, profile: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (
            "user_id",
            "display_name",
            "gender",
            "age",
            "height_cm",
            "weight_kg",
            "is_pregnant",
            "is_breastfeeding",
            "conditions",
            "allergies",
            "notes",
        )
        return {
            field: profile.get(field)
            for field in allowed_fields
            if field in profile
        }

    @staticmethod
    def _merge_unique(values: Any, extra_values: Any = None) -> List[str]:
        result: List[str] = []
        seen = set()
        for group in (values or [], extra_values or []):
            if isinstance(group, (str, bytes)):
                candidates = [group]
            else:
                candidates = group or []
            for value in candidates:
                text = str(value or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    result.append(text)
        return result[:12]

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Decimal):
            if value == value.to_integral():
                return int(value)
            return float(value)
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        return value
