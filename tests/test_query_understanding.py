import json
import unittest

from langchain_core.messages import AIMessage

from services.query_understanding import QueryUnderstandingService


class FakeBoundLLM:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, messages):
        return self.owner._invoke_with_tools(messages)


class FakeAmbiguousLLM:
    def __init__(self):
        self.tool_rounds = 0
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return FakeBoundLLM(self)

    def _invoke_with_tools(self, messages):
        self.tool_rounds += 1
        if self.tool_rounds == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "queryUserMedicationSummary",
                        "args": {},
                        "id": "call_med_1",
                    }
                ],
            )
        return AIMessage(
            content=json.dumps(
                {
                    "domain": "drug_related",
                    "route": "continue",
                    "reason": "clear_query",
                    "intent": "drug_info",
                    "resolved_query": "用户当前正在服用哪些药物",
                    "clarification": "",
                    "rewrite_queries": ["用户当前正在服用哪些药物"],
                    "need_rewrite": True,
                    "need_tool": True,
                    "tool_candidates": ["queryUserMedicationSummary"],
                    "skip_retrieval": True,
                },
                ensure_ascii=False,
            )
        )

    def invoke(self, messages):
        return AIMessage(content="{}")


class FakeMedicalNER:
    def extract(self, normalized):
        return {
            "drug_entities": [],
            "symptom_entities": [],
            "disease_entities": [],
            "population_entities": [],
            "food_entities": [],
            "dose_entities": [],
        }


class FakeDrugLexicon:
    def match_mentions(self, normalized):
        return []


class FakeDatabaseTool:
    def query_user_medication_summary_payload(self, user_id):
        return {
            "ok": True,
            "tool": "queryUserMedicationSummary",
            "reason": "success",
            "message": "已读取当前用户登记的用药信息。",
            "records": [
                {
                    "drug_name": "阿莫西林胶囊",
                    "dosage": "1g",
                    "purpose": "消炎",
                    "frequency": "每日",
                    "times_per_day": 2,
                    "administration_time": "午餐前",
                    "start_date": None,
                    "end_date": None,
                }
            ],
            "count": 1,
        }

    def query_user_health_profile_payload(self, user_id):
        return {
            "ok": True,
            "tool": "queryUserHealthProfile",
            "reason": "success",
            "message": "已读取当前用户的健康档案。",
            "records": [{"display_name": "test001", "conditions": ["风湿病"]}],
            "count": 1,
        }

    @staticmethod
    def format_tool_result(payload):
        if payload.get("tool") == "queryUserMedicationSummary":
            return "已读取当前用户登记的用药信息。\n\n1. 阿莫西林胶囊 / 1g / 每日 / 午餐前 / 消炎 / 每日2次"
        return "已读取当前用户的健康档案。\n\n基础病: 风湿病"


class QueryUnderstandingAmbiguousToolLoopTests(unittest.TestCase):
    def _build_service(self):
        service = QueryUnderstandingService.__new__(QueryUnderstandingService)
        service.light_intent_classifier = None
        service.llm = FakeAmbiguousLLM()
        service.medical_ner = FakeMedicalNER()
        service.drug_lexicon = FakeDrugLexicon()
        service.database_tool = FakeDatabaseTool()
        return service

    def test_ambiguous_stage_can_prefetch_medication_tool_context(self):
        service = self._build_service()

        result = service._resolve_ambiguous_stage(
            "我现在在吃什么药",
            {"domain": "ambiguous", "route": "stage2_review", "intent": "unknown", "reason": "clear_query"},
            history=[],
            memory_summary="",
            user_logged_in=True,
            available_tools=["queryUserMedicationSummary"],
            profile_available=True,
            user_id=7,
        )

        self.assertEqual(result["route"], "continue")
        self.assertTrue(result["need_tool"])
        self.assertTrue(result["skip_retrieval"])
        self.assertEqual(result["tool_candidates"], ["queryUserMedicationSummary"])
        self.assertEqual(result["prefetched_tools"], ["queryUserMedicationSummary"])
        self.assertIn("阿莫西林胶囊", result["prefetched_context"])


if __name__ == "__main__":
    unittest.main()
