import unittest
import asyncio

from services.chat_orchestrator import ChatOrchestrator


class FakeMemoryService:
    def __init__(self):
        self.appended = []

    def get_recent_turns(self, memory_id, turns=5):
        return []

    def get_summary(self, memory_id):
        return ""

    def append_exchange(self, memory_id, user_message, assistant_message, assistant_meta=None):
        self.appended.append((memory_id, user_message, assistant_message, assistant_meta))

    def build_effective_context_snapshot(self, personal_profile=None):
        personal_profile = personal_profile or {}
        return {
            "display_name": str(personal_profile.get("display_name") or ""),
            "conditions": list(personal_profile.get("conditions") or []),
            "allergies": list(personal_profile.get("allergies") or []),
            "sources": {
                "conditions": "profile" if personal_profile.get("conditions") else "none",
            },
        }

    def format_effective_context_snapshot(self, snapshot):
        return "[effective]"


class FakeLLMService:
    def __init__(self):
        self.memory_service = FakeMemoryService()
        self.chat_calls = []
        self.direct_calls = []
        self.stream_calls = []
        self.last_run_metadata = {"used_tools": False, "tool_calls": []}

    def chat(self, *args):
        self.chat_calls.append(args)
        return "medical answer"

    def chat_direct(self, memory_id, message, personal_context=""):
        self.direct_calls.append((memory_id, message, personal_context))
        return "general answer"

    async def chat_stream(self, *args):
        self.stream_calls.append(args)
        yield "chunk"

    async def chat_stream_direct(self, memory_id, message, personal_context=""):
        yield "chunk"

    def get_last_run_metadata(self):
        return dict(self.last_run_metadata)


class FakeQueryUnderstanding:
    def __init__(self, analysis):
        self.analysis = analysis
        self.analyze_calls = 0
        self.analyze_payloads = []

    def analyze(self, message, **kwargs):
        self.analyze_calls += 1
        self.analyze_payloads.append({"message": message, **kwargs})
        return dict(self.analysis)

    def build_retrieval_queries(self, analysis, fallback=""):
        return [analysis.get("normalized_query") or fallback]


class FakeDatabaseTool:
    def __init__(self, capabilities, profile=None):
        self.capabilities = capabilities
        self.profile = profile or {
            "display_name": "张阿姨",
            "conditions": ["高血压"],
            "allergies": [],
            "medications": [{"drug_name": "缬沙坦", "dosage": "80mg"}],
            "notes": "",
        }

    def get_capabilities(self):
        return dict(self.capabilities)

    def get_user_health_profile(self, user_id, include_medications=True):
        return dict(self.profile)

    def query_user_medication_summary(self, user_id):
        return "1. 缬沙坦 / 80mg"


class ChatOrchestratorTests(unittest.TestCase):
    def test_prepare_turn_does_not_expose_legacy_tools_for_guest_drug_interaction(self):
        analysis = {
            "route": "continue",
            "intent": "interaction",
            "domain": "drug_related",
            "normalized_query": "布洛芬和缬沙坦能一起吃吗",
            "drug_entities": ["布洛芬", "缬沙坦"],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        prepared = orchestrator.prepare_turn("m1", "布洛芬和缬沙坦能一起吃吗", user_id=None)

        self.assertEqual(prepared["allowed_tool_names"], [])
        self.assertEqual(prepared["meta"]["retrieval"]["backend"], "graphrag")

    def test_prepare_turn_selects_profile_tool_for_logged_in_personal_question(self):
        analysis = {
            "route": "continue",
            "intent": "drug_info",
            "domain": "drug_related",
            "normalized_query": "我正在吃缬沙坦，还能吃什么止痛药",
            "drug_entities": ["缬沙坦"],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool(
                {
                    "user_health_profile_table": True,
                    "user_medications_table": True,
                }
            ),
        )

        prepared = orchestrator.prepare_turn("m2", "我正在吃缬沙坦，还能吃什么止痛药", user_id=7)

        self.assertEqual(prepared["allowed_tool_names"], ["queryUserHealthProfile", "queryUserMedicationSummary"])
        self.assertEqual(prepared["runtime_tool_kwargs"]["queryUserHealthProfile"]["user_id"], 7)
        self.assertEqual(prepared["runtime_tool_kwargs"]["queryUserMedicationSummary"]["user_id"], 7)
        self.assertEqual(prepared["effective_background_context"], "[effective]")
        self.assertTrue(prepared["meta"]["background"]["profile_available"])

    def test_prepare_turn_does_not_force_current_medication_lookup_by_rule(self):
        analysis = {
            "route": "ask_user",
            "intent": "unknown",
            "domain": "ambiguous",
            "normalized_query": "我现在在吃什么药",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
            "reason": "ambiguous_reference",
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=None,
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        prepared = orchestrator.prepare_turn("m-med", "我现在在吃什么药", user_id=7)

        self.assertEqual(prepared["route"], "ask_user")
        self.assertEqual(prepared["context"], "")

    def test_prepare_turn_passes_history_to_ambiguous_stage_llm(self):
        analysis = {
            "route": "general_answer",
            "intent": "general_query",
            "domain": "general",
            "normalized_query": "我上句话说了什么",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": True,
            "reason": "memory_question_answerable",
        }
        llm = FakeLLMService()
        llm.memory_service.get_recent_turns = lambda memory_id, turns=5: [
            {"role": "user", "content": "我刚才问了布洛芬"},
            {"role": "assistant", "content": "你刚才提到了布洛芬"},
        ]
        query = FakeQueryUnderstanding(analysis)
        orchestrator = ChatOrchestrator(
            query_understanding=query,
            rag_service=None,
            llm_service=llm,
            graphrag_service=None,
            database_tool=None,
        )

        prepared = orchestrator.prepare_turn("m-memory", "我上句话说了什么", user_id=None)

        self.assertEqual(prepared["route"], "general_answer")
        self.assertEqual(prepared["meta"]["route"], "general_answer")
        self.assertEqual(query.analyze_payloads[-1]["history"][-1]["content"], "你刚才提到了布洛芬")
        self.assertIsNone(query.analyze_payloads[-1]["user_id"])

    def test_prepare_turn_passes_user_id_into_query_understanding(self):
        analysis = {
            "route": "continue",
            "intent": "drug_info",
            "domain": "drug_related",
            "normalized_query": "我现在在吃什么药",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        query = FakeQueryUnderstanding(analysis)
        orchestrator = ChatOrchestrator(
            query_understanding=query,
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        orchestrator.prepare_turn("m-user", "我现在在吃什么药", user_id=7)

        self.assertEqual(query.analyze_payloads[-1]["user_id"], 7)

    def test_prepare_turn_honors_tool_candidates_from_ambiguous_stage(self):
        analysis = {
            "route": "continue",
            "intent": "drug_info",
            "domain": "drug_related",
            "normalized_query": "我现在在吃什么药",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
            "tool_candidates": ["queryUserMedicationSummary"],
            "need_tool": True,
            "rewrite_queries": ["我现在在吃什么药"],
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        prepared = orchestrator.prepare_turn("m-tool", "我现在在吃什么药", user_id=7)

        self.assertEqual(prepared["allowed_tool_names"], ["queryUserMedicationSummary"])

    def test_prepare_turn_uses_prefetched_tool_context_without_retrieval_or_duplicate_tools(self):
        analysis = {
            "route": "continue",
            "intent": "drug_info",
            "domain": "drug_related",
            "normalized_query": "我现在在吃什么药",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": True,
            "tool_candidates": ["queryUserMedicationSummary"],
            "need_tool": True,
            "skip_retrieval": True,
            "prefetched_context": "已读取当前用户登记的用药信息。\n\n1. 阿莫西林胶囊 / 1g / 每日 / 午餐前 / 消炎 / 每日2次",
            "prefetched_tools": ["queryUserMedicationSummary"],
            "rewrite_queries": ["用户当前正在服用哪些药物"],
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        prepared = orchestrator.prepare_turn("m-prefetch", "我现在在吃什么药", user_id=7)

        self.assertEqual(prepared["context"], analysis["prefetched_context"])
        self.assertEqual(prepared["meta"]["retrieval"]["backend"], "prefetched_context")
        self.assertEqual(prepared["meta"]["retrieval"]["prefetched_tools"], ["queryUserMedicationSummary"])
        self.assertEqual(prepared["allowed_tool_names"], [])
        self.assertEqual(prepared["runtime_tool_kwargs"], {})

    def test_answer_with_meta_returns_general_route_metadata(self):
        analysis = {
            "route": "general_answer",
            "intent": "general_query",
            "domain": "general",
            "normalized_query": "今天天气怎么样",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        llm = FakeLLMService()
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=llm,
            graphrag_service=None,
            database_tool=None,
        )

        answer, meta = orchestrator.answer_with_meta("m3", "今天天气怎么样", user_id=None)

        self.assertEqual(answer, "general answer")
        self.assertEqual(meta["response_mode"], "general_answer")
        self.assertEqual(meta["route"], "general_answer")

    def test_answer_with_meta_persists_clarification_route(self):
        analysis = {
            "route": "ask_user",
            "intent": "unknown",
            "domain": "drug_related",
            "normalized_query": "这个能吃吗",
            "drug_entities": [],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
            "clarification": "请补充药名",
        }
        llm = FakeLLMService()
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=llm,
            graphrag_service=None,
            database_tool=None,
        )

        answer, meta = orchestrator.answer_with_meta("m6", "这个能吃吗", user_id=None)

        self.assertIn("请补充药名", answer)
        self.assertEqual(meta["response_mode"], "clarification")
        self.assertEqual(llm.memory_service.appended[-1][0], "m6")
        self.assertEqual(llm.memory_service.appended[-1][3]["response_mode"], "clarification")

    def test_build_context_snapshot_uses_profile_and_session_memory_flags(self):
        analysis = {
            "route": "continue",
            "intent": "interaction",
            "domain": "drug_related",
            "normalized_query": "布洛芬和缬沙坦能一起吃吗",
            "drug_entities": ["布洛芬", "缬沙坦"],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=None,
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        orchestrator.llm_service.memory_service.get_recent_turns = lambda memory_id, turns=1: [
            {"role": "user", "content": "我在吃缬沙坦"}
        ]
        payload = orchestrator.build_context_snapshot("m7", user_id=7)

        self.assertTrue(payload["profile_available"])
        self.assertTrue(payload["memory_available"])
        self.assertEqual(payload["effective_context"]["conditions"], ["高血压"])
        self.assertNotIn("current_medications", payload["effective_context"])

    def test_prepare_turn_disables_tools_when_policy_off(self):
        analysis = {
            "route": "continue",
            "intent": "interaction",
            "domain": "drug_related",
            "normalized_query": "我正在吃缬沙坦，还能吃布洛芬吗",
            "drug_entities": ["缬沙坦", "布洛芬"],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        orchestrator = ChatOrchestrator(
            query_understanding=FakeQueryUnderstanding(analysis),
            rag_service=None,
            llm_service=FakeLLMService(),
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool(
                {
                    "user_health_profile_table": True,
                    "user_medications_table": True,
                }
            ),
        )

        prepared = orchestrator.prepare_turn("m5", "我正在吃缬沙坦，还能吃布洛芬吗", user_id=7, tool_calls_enabled=False)

        self.assertEqual(prepared["allowed_tool_names"], [])
        self.assertEqual(prepared["runtime_tool_kwargs"], {})

    def test_answer_stream_prepared_reuses_existing_prepared_turn(self):
        analysis = {
            "route": "continue",
            "intent": "interaction",
            "domain": "drug_related",
            "normalized_query": "布洛芬和缬沙坦能一起吃吗",
            "drug_entities": ["布洛芬", "缬沙坦"],
            "disease_entities": [],
            "symptom_entities": [],
            "context_resolved": False,
        }
        query = FakeQueryUnderstanding(analysis)
        llm = FakeLLMService()
        orchestrator = ChatOrchestrator(
            query_understanding=query,
            rag_service=None,
            llm_service=llm,
            graphrag_service=type("Graph", (), {"retrieve_context_for_analysis": lambda self, a, q: "ctx"})(),
            database_tool=FakeDatabaseTool({"user_health_profile_table": True, "user_medications_table": True}),
        )

        prepared = orchestrator.prepare_turn("m4", "布洛芬和缬沙坦能一起吃吗", user_id=None)

        async def collect():
            chunks = []
            async for chunk in orchestrator.answer_stream_prepared(prepared, "m4", "布洛芬和缬沙坦能一起吃吗", user_id=None):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())

        self.assertEqual(query.analyze_calls, 1)
        self.assertEqual(chunks, ["chunk"])
        self.assertEqual(len(llm.stream_calls), 1)


if __name__ == "__main__":
    unittest.main()
