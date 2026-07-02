import unittest

from services.memory_service import MemoryService


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.list_store = {}

    def lrange(self, key, start, end):
        values = self.list_store.get(key, [])
        if end == -1:
            end = len(values) - 1
        if start < 0:
            start = max(len(values) + start, 0)
        if end < 0:
            end = len(values) + end
        return values[start : end + 1]

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    def expire(self, key, ttl):
        return None

    def ltrim(self, key, start, end):
        values = self.list_store.get(key, [])
        if start < 0:
            start = max(len(values) + start, 0)
        if end < 0:
            end = len(values) + end
        self.list_store[key] = values[start : end + 1]

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.list_store.pop(key, None)


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def rpush(self, key, *values):
        self.ops.append(("rpush", key, values))

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", key, start, end))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    def delete(self, key):
        self.ops.append(("delete", key))

    def execute(self):
        for op in self.ops:
            if op[0] == "rpush":
                _, key, values = op
                self.client.list_store.setdefault(key, []).extend(values)
            elif op[0] == "ltrim":
                _, key, start, end = op
                self.client.ltrim(key, start, end)
            elif op[0] == "delete":
                _, key = op
                self.client.delete(key)
        self.ops = []


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = MemoryService.__new__(MemoryService)
        self.service.redis_url = "redis://fake"
        self.service.memory_window = 10
        self.service.max_messages = 20
        self.service.redis_client_kwargs = {}
        self.service.client = FakeRedis()

    def test_export_history_and_clear_memory(self):
        self.service.append_exchange("m1", "你好", "你好，这里是助手")

        history = self.service.export_history("m1")

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，这里是助手"},
            ],
        )

        self.service.set_summary("m1", "这是摘要")
        self.service.clear_memory("m1")

        self.assertEqual(self.service.export_history("m1"), [])
        self.assertEqual(self.service.get_summary("m1"), "")

    def test_update_last_assistant_meta_persists_to_export_history(self):
        self.service.append_exchange("m2", "我能吃布洛芬吗", "可以先确认相互作用")

        updated = self.service.update_last_assistant_meta(
            "m2",
            {"route": "continue", "tooling": {"used_tools": True, "tool_rounds": 2}},
        )
        history = self.service.export_history("m2")

        self.assertTrue(updated)
        self.assertEqual(history[-1]["meta"]["route"], "continue")
        self.assertEqual(history[-1]["meta"]["tooling"]["tool_rounds"], 2)

    def test_effective_context_snapshot_uses_profile_only(self):
        snapshot = self.service.build_effective_context_snapshot(
            {
                "display_name": "张阿姨",
                "conditions": ["高血压", "糖尿病"],
                "allergies": [],
                "medications": [{"drug_name": "缬沙坦", "dosage": "80mg", "frequency": "每日一次"}],
                "notes": "长期复查",
            },
        )

        self.assertEqual(snapshot["conditions"], ["高血压", "糖尿病"])
        self.assertNotIn("current_medications", snapshot)
        self.assertEqual(snapshot["sources"]["conditions"], "profile")

    def test_format_effective_context_snapshot_labels_sources(self):
        text = self.service.format_effective_context_snapshot(
            {
                "display_name": "张阿姨",
                "conditions": ["高血压"],
                "allergies": [],
                "notes": "",
                "sources": {
                    "conditions": "profile",
                },
            }
        )

        self.assertIn("基础病(来源: 用户档案): 高血压", text)
        self.assertNotIn("当前用药", text)

    def test_session_registry_supports_upsert_rename_and_delete(self):
        session = self.service.upsert_session("user:9", {"id": "user_id_9_session_a", "title": "高血压", "preview": "缬沙坦"})

        self.assertEqual(session["title"], "高血压")
        self.assertEqual(len(self.service.list_sessions("user:9")), 1)

        renamed = self.service.rename_session("user:9", "user_id_9_session_a", "改名后")

        self.assertEqual(renamed["title"], "改名后")
        self.assertEqual(self.service.list_sessions("user:9")[0]["title"], "改名后")

        deleted = self.service.delete_session("user:9", "user_id_9_session_a")

        self.assertTrue(deleted)
        self.assertEqual(self.service.list_sessions("user:9"), [])

    def test_touch_session_keeps_manual_title(self):
        self.service.upsert_session("user:9", {"id": "user_id_9_session_b", "title": "手动标题", "preview": "旧预览"})

        touched = self.service.touch_session(
            "user:9",
            {"id": "user_id_9_session_b", "title": "自动标题", "preview": "新预览"},
        )

        self.assertEqual(touched["title"], "手动标题")
        self.assertEqual(touched["preview"], "新预览")

    def test_user_profile_cache_roundtrip(self):
        profile = {
            "display_name": "张阿姨",
            "height_cm": 160.5,
            "conditions": ["高血压"],
            "medications": [{"drug_name": "缬沙坦", "dosage": "80mg"}],
        }
        context = "[用户个人档案]\n称呼: 张阿姨"

        self.service.set_user_profile_cache(9, profile, context)
        cached = self.service.get_user_profile_cache(9)

        self.assertEqual(cached["profile"]["display_name"], "张阿姨")
        self.assertEqual(cached["profile"]["conditions"], ["高血压"])
        self.assertNotIn("medications", cached["profile"])
        self.assertEqual(cached["context"], context)

        self.service.invalidate_user_profile_cache(9)
        self.assertEqual(self.service.get_user_profile_cache(9), {"profile": {}, "context": ""})


if __name__ == "__main__":
    unittest.main()
