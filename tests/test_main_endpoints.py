import asyncio
import json
import unittest

import main
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse


class FakeChatOrchestrator:
    def __init__(self):
        self.answer_with_meta_calls = []
        self.prepare_turn_calls = []
        self.answer_stream_prepared_calls = []
        self.llm_service = FakeLLMService()

    def answer_with_meta(self, memory_id, message, user_id=None, tool_calls_enabled=True):
        self.answer_with_meta_calls.append((memory_id, message, user_id, tool_calls_enabled))
        return "reply text", {"route": "continue", "intent": "interaction"}

    def prepare_turn(self, memory_id, message, user_id=None, tool_calls_enabled=True):
        self.prepare_turn_calls.append((memory_id, message, user_id, tool_calls_enabled))
        return {
            "meta": {
                "route": "continue",
                "intent": "interaction",
                "retrieval": {"backend": "graphrag", "method": "local"},
                "allowed_tool_names": ["queryUserHealthProfile", "queryUserMedicationSummary"],
            }
        }

    async def answer_stream_prepared(self, prepared, memory_id, message, user_id=None, tool_calls_enabled=True):
        self.answer_stream_prepared_calls.append((prepared, memory_id, message, user_id, tool_calls_enabled))
        yield json.dumps({"data": "hello"}, ensure_ascii=False)

    def build_context_snapshot(self, memory_id, user_id=None):
        return {
            "memory_id": memory_id,
            "user_logged_in": user_id is not None,
            "profile_available": user_id is not None,
            "memory_available": True,
            "effective_context": {
                "conditions": ["高血压"],
            },
            "effective_context_text": "基础病: 高血压",
        }


class FakeLLMService:
    def __init__(self):
        self.memory_service = FakeMemoryService()
        self.tools = [
            {"function": {"name": "queryUserHealthProfile"}},
            {"function": {"name": "queryUserMedicationSummary"}},
        ]

    def get_last_run_metadata(self):
        return {
            "used_tools": True,
            "tool_calls": [
                {"name": "queryUserHealthProfile", "ok": True},
                {"name": "queryUserMedicationSummary", "ok": True},
            ],
        }


class FakeMemoryService:
    def __init__(self):
        self.cleared = []
        self.appended = []
        self.updated_meta = []
        self.profile_cache = {}
        self.sessions = [
            {
                "id": "user_id_9_session_a",
                "title": "高血压咨询",
                "preview": "我在吃缬沙坦",
                "updated_at": 1234567890,
            }
        ]

    def export_history(self, memory_id, turns=20):
        return [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "第二句", "meta": {"route": "continue", "tool_policy": "default"}},
        ]

    def get_summary(self, memory_id):
        return "摘要"

    def clear_memory(self, memory_id):
        self.cleared.append(memory_id)

    def append_exchange(self, memory_id, user_message, assistant_message, assistant_meta=None):
        self.appended.append((memory_id, user_message, assistant_message, assistant_meta))

    def update_last_assistant_meta(self, memory_id, meta):
        self.updated_meta.append((memory_id, meta))
        return True

    def set_user_profile_cache(self, user_id, profile, context):
        self.profile_cache[user_id] = {"profile": dict(profile), "context": context}

    def list_sessions(self, scope_id):
        return list(self.sessions)

    def upsert_session(self, scope_id, payload):
        session = {
            "id": payload["id"],
            "title": payload.get("title", ""),
            "preview": payload.get("preview", ""),
            "updated_at": payload.get("updated_at", 0),
        }
        self.sessions = [session, *[item for item in self.sessions if item["id"] != session["id"]]]
        return session

    def touch_session(self, scope_id, payload):
        return self.upsert_session(scope_id, payload)

    def rename_session(self, scope_id, session_id, title):
        for item in self.sessions:
            if item["id"] == session_id:
                item["title"] = title
                return dict(item)
        return None

    def delete_session(self, scope_id, session_id):
        before = len(self.sessions)
        self.sessions = [item for item in self.sessions if item["id"] != session_id]
        return len(self.sessions) != before


class FakeAuthService:
    def __init__(self):
        self.register_calls = []
        self.login_calls = []
        self.verify_calls = []

    def register(self, username, password):
        self.register_calls.append((username, password))
        return {"user": {"id": 1, "username": username}, "token": "token-register", "expires_at": 123}

    def login(self, username, password):
        self.login_calls.append((username, password))
        return {"user": {"id": 2, "username": username}, "token": "token-login", "expires_at": 456}

    def verify_token(self, token):
        self.verify_calls.append(token)
        if token == "good-token":
            return {"uid": 9, "username": "demo"}
        raise main.AuthError("登录状态无效，请重新登录")


class FakeDatabaseTool:
    def __init__(self):
        self.get_profile_calls = []
        self.upsert_profile_calls = []
        self.capabilities = {
            "user_health_profile_table": True,
            "user_medications_table": True,
        }

    def get_capabilities(self):
        return dict(self.capabilities)

    def get_user_health_profile(self, user_id, include_medications=True):
        self.get_profile_calls.append(user_id)
        return {
            "user_id": user_id,
            "display_name": "张阿姨",
            "gender": "女",
            "age": 63,
            "conditions": ["高血压"],
            "allergies": ["青霉素"],
            "medications": [{"drug_name": "缬沙坦"}],
            "notes": "长期复查",
        }

    def upsert_user_health_profile(self, user_id, payload):
        self.upsert_profile_calls.append((user_id, payload))
        result = dict(payload)
        result["user_id"] = user_id
        return result


class MainEndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_chat_orchestrator = getattr(main, "chat_orchestrator", None)
        self.original_auth_service = getattr(main, "auth_service", None)
        self.original_database_tool = getattr(main, "database_tool", None)
        main.chat_orchestrator = FakeChatOrchestrator()
        main.auth_service = FakeAuthService()
        main.database_tool = FakeDatabaseTool()

    def tearDown(self):
        if self.original_chat_orchestrator is None:
            delattr(main, "chat_orchestrator")
        else:
            main.chat_orchestrator = self.original_chat_orchestrator
        if self.original_auth_service is None:
            delattr(main, "auth_service")
        else:
            main.auth_service = self.original_auth_service
        if self.original_database_tool is None:
            delattr(main, "database_tool")
        else:
            main.database_tool = self.original_database_tool

    def test_register_returns_auth_payload(self):
        payload = main.AuthRequest(username="demo", password="123456")

        response = asyncio.run(main.register(payload))

        self.assertEqual(response["token"], "token-register")
        self.assertEqual(main.auth_service.register_calls, [("demo", "123456")])

    def test_login_returns_auth_payload(self):
        payload = main.AuthRequest(username="demo", password="123456")

        response = asyncio.run(main.login(payload))

        self.assertEqual(response["token"], "token-login")
        self.assertEqual(main.auth_service.login_calls, [("demo", "123456")])

    def test_chat_returns_plain_text_by_default(self):
        response = asyncio.run(main.chat(memory_id="m1", message="test", include_meta=False, authorization=None))

        self.assertIsInstance(response, PlainTextResponse)
        self.assertEqual(response.body.decode("utf-8"), "reply text")
        self.assertEqual(main.chat_orchestrator.answer_with_meta_calls[0][3], bool(main.LLM_TOOL_CALLS_ENABLED))

    def test_chat_returns_payload_when_include_meta_enabled(self):
        response = asyncio.run(main.chat(memory_id="m2", message="test", include_meta=True, authorization=None))

        self.assertEqual(response.answer, "reply text")
        self.assertEqual(response.meta["route"], "continue")
        self.assertEqual(main.chat_orchestrator.llm_service.memory_service.updated_meta[-1][0], "m2")

    def test_chat_force_on_tools_for_request(self):
        response = asyncio.run(
            main.chat(memory_id="m2b", message="test", include_meta=True, tool_policy="force_on", authorization=None)
        )

        self.assertEqual(response.meta["tool_policy"], "force_on")
        self.assertTrue(main.chat_orchestrator.answer_with_meta_calls[-1][3])

    def test_chat_force_off_tools_for_request(self):
        response = asyncio.run(
            main.chat(memory_id="m2c", message="test", include_meta=True, tool_policy="force_off", authorization=None)
        )

        self.assertEqual(response.meta["tool_policy"], "force_off")
        self.assertFalse(main.chat_orchestrator.answer_with_meta_calls[-1][3])

    def test_chat_stream_emits_initial_and_final_meta(self):
        response = asyncio.run(main.chat_stream(memory_id="m3", message="test", authorization=None))
        self.assertIsInstance(response, EventSourceResponse)

        async def collect():
            items = []
            async for item in response.body_iterator:
                items.append(item)
            return items

        chunks = asyncio.run(collect())

        self.assertEqual(len(chunks), 4)
        self.assertIn('"meta"', chunks[0])
        self.assertIn('"data"', chunks[1])
        self.assertIn('"meta_update"', chunks[2])
        self.assertIn('"done"', chunks[3])
        self.assertEqual(main.chat_orchestrator.llm_service.memory_service.updated_meta[-1][0], "m3")

    def test_chat_history_returns_memory_snapshot(self):
        response = asyncio.run(main.get_chat_history(memory_id="guest_demo", turns=20, authorization=None))

        self.assertEqual(response.memory_id, "guest_demo")
        self.assertEqual(len(response.messages), 2)
        self.assertEqual(response.summary, "摘要")
        self.assertEqual(response.messages[1].meta["route"], "continue")

    def test_chat_context_returns_effective_snapshot(self):
        response = asyncio.run(main.get_chat_context(memory_id="guest_demo", authorization=None))

        self.assertEqual(response.memory_id, "guest_demo")
        self.assertTrue(response.memory_available)
        self.assertEqual(response.effective_context["conditions"], ["高血压"])

    def test_chat_history_allows_user_scoped_session_id(self):
        response = asyncio.run(
            main.get_chat_history(memory_id="user_id_9_session_demo", turns=20, authorization="Bearer good-token")
        )

        self.assertEqual(response.memory_id, "user_id_9_session_demo")

    def test_chat_history_rejects_other_user_session_id(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(
                main.get_chat_history(memory_id="user_id_10_session_demo", turns=20, authorization="Bearer good-token")
            )

        self.assertEqual(context.exception.status_code, 403)

    def test_clear_chat_history_calls_memory_service(self):
        response = asyncio.run(main.clear_chat_history(memory_id="guest_demo", authorization=None))

        self.assertTrue(response.cleared)
        self.assertEqual(main.chat_orchestrator.llm_service.memory_service.cleared, ["guest_demo"])

    def test_list_chat_sessions_returns_items(self):
        response = asyncio.run(main.list_chat_sessions(authorization="Bearer good-token"))

        self.assertEqual(len(response.sessions), 1)
        self.assertEqual(response.sessions[0].title, "高血压咨询")

    def test_create_chat_session_returns_created_item(self):
        response = asyncio.run(main.create_chat_session(main.SessionPayload(title="新建会话"), authorization="Bearer good-token"))

        self.assertTrue(response.id.startswith("user_id_9_session_"))
        self.assertEqual(response.title, "新建会话")

    def test_rename_chat_session_updates_title(self):
        response = asyncio.run(
            main.rename_chat_session(
                "user_id_9_session_a",
                main.SessionPatchPayload(title="改名后"),
                authorization="Bearer good-token",
            )
        )

        self.assertEqual(response.title, "改名后")

    def test_delete_chat_session_removes_item(self):
        response = asyncio.run(main.delete_chat_session("user_id_9_session_a", authorization="Bearer good-token"))

        self.assertTrue(response["deleted"])

    def test_runtime_status_reports_backend_capabilities(self):
        response = asyncio.run(main.get_runtime_status())

        self.assertTrue(response.database["available"])
        self.assertEqual(response.available_tools, ["queryUserHealthProfile", "queryUserMedicationSummary"])
        self.assertTrue(response.memory["available"])
        self.assertIn(response.retrieval["primary_backend"], {"graphrag", "legacy_rag", "none"})
        self.assertTrue(response.tool_calls_available)

    def test_get_health_profile_requires_valid_token(self):
        response = asyncio.run(main.get_health_profile(authorization="Bearer good-token"))

        self.assertEqual(response["display_name"], "张阿姨")
        self.assertEqual(main.database_tool.get_profile_calls, [9])

    def test_get_health_profile_rejects_invalid_token(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(main.get_health_profile(authorization="Bearer bad-token"))

        self.assertEqual(context.exception.status_code, 401)

    def test_update_health_profile_persists_payload(self):
        payload = main.HealthProfilePayload(
            display_name="李叔",
            gender="男",
            age=58,
            conditions=["高血压"],
            allergies=["阿司匹林"],
            medications=[main.UserMedicationPayload(drug_name="缬沙坦", dosage="80mg")],
            notes="需要复诊",
        )

        response = asyncio.run(main.update_health_profile(payload, authorization="Bearer good-token"))

        self.assertEqual(response["user_id"], 9)
        self.assertEqual(response["display_name"], "李叔")
        saved_user_id, saved_payload = main.database_tool.upsert_profile_calls[0]
        self.assertEqual(saved_user_id, 9)
        self.assertEqual(saved_payload["medications"][0]["drug_name"], "缬沙坦")
        self.assertEqual(saved_payload["notes"], "需要复诊")


if __name__ == "__main__":
    unittest.main()
