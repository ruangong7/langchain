import asyncio
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from services.llm_service import LLMService


class FakeResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = list(tool_calls or [])


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeBoundLLM:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, messages):
        self.owner.bound_messages.append(list(messages))
        return self.owner.bound_responses.pop(0)


class FakeLLM:
    def __init__(self, *, bound_responses=None, final_response=None, stream_chunks=None):
        self.bound_responses = list(bound_responses or [])
        self.bound_messages = []
        self.bound_tools_calls = []
        self.invoke_calls = []
        self.astream_calls = []
        self.final_response = final_response or FakeResponse("final")
        self.stream_chunks = list(stream_chunks or [])

    def bind_tools(self, tools):
        self.bound_tools_calls.append(list(tools))
        return FakeBoundLLM(self)

    def invoke(self, messages):
        self.invoke_calls.append(list(messages))
        return self.final_response

    async def astream(self, messages):
        self.astream_calls.append(list(messages))
        for chunk in self.stream_chunks:
            yield FakeChunk(chunk)


class FakeMemoryService:
    def __init__(self):
        self.saved = []
        self.history = []
        self.overflow = []
        self.summary = ""
        self.trim_calls = []
        self.summary_updates = []

    def get_history(self, memory_id):
        return list(self.history)

    def get_summary(self, memory_id):
        return self.summary

    def get_overflow_history(self, memory_id, keep_turns):
        return list(self.overflow)

    def trim_to_recent_turns(self, memory_id, turns):
        self.trim_calls.append((memory_id, turns))

    def set_summary(self, memory_id, summary):
        self.summary = summary
        self.summary_updates.append((memory_id, summary))

    def append_exchange(self, memory_id, user_message, assistant_message):
        self.saved.append((memory_id, user_message, assistant_message))


class LLMServiceToolRuntimeTests(unittest.TestCase):
    def build_service(self, llm):
        service = LLMService.__new__(LLMService)
        service.llm = llm
        service.tools = [
            {"type": "function", "function": {"name": "queryUserHealthProfile"}},
            {"type": "function", "function": {"name": "queryUserMedicationSummary"}},
        ]
        service.tool_schemas_by_name = {
            item["function"]["name"]: item
            for item in service.tools
        }
        service.tool_handlers = {}
        service.memory_service = FakeMemoryService()
        service.last_run_metadata = {}
        service._runtime_tool_kwargs = {}
        service.MAX_TOOL_CALL_ROUNDS = 3
        return service

    def test_run_tool_call_merges_runtime_tool_kwargs(self):
        captured = {}
        service = self.build_service(FakeLLM())

        def handler(user_id):
            captured["user_id"] = user_id
            return "ok"

        service.tool_handlers = {"queryUserHealthProfile": handler}
        service._runtime_tool_kwargs = {"queryUserHealthProfile": {"user_id": 9}}

        tool_message, metadata = service._run_tool_call(
            {"name": "queryUserHealthProfile", "args": {}, "id": "call_1"}
        )

        self.assertEqual(captured["user_id"], 9)
        self.assertEqual(tool_message.content, "ok")
        self.assertTrue(metadata["ok"])

    def test_invoke_with_tools_supports_multiple_rounds(self):
        llm = FakeLLM(
            bound_responses=[
                FakeResponse(tool_calls=[{"name": "queryUserHealthProfile", "args": {}, "id": "call_1"}]),
                FakeResponse(
                    tool_calls=[{"name": "queryUserMedicationSummary", "args": {"focus": "止痛药"}, "id": "call_2"}]
                ),
                FakeResponse(content="最终回答"),
            ]
        )
        service = self.build_service(llm)
        captured = []

        def profile_handler(user_id):
            captured.append(("profile", user_id))
            return "读取到用户档案"

        def summary_handler(focus):
            captured.append(("summary", focus))
            return "读取到个体化用药摘要"

        service.tool_handlers = {
            "queryUserHealthProfile": profile_handler,
            "queryUserMedicationSummary": summary_handler,
        }

        response = service._invoke_with_tools(
            [HumanMessage(content="用户问题：我现在能不能吃布洛芬")],
            allowed_tool_names=["queryUserHealthProfile", "queryUserMedicationSummary"],
            runtime_tool_kwargs={"queryUserHealthProfile": {"user_id": 9}},
        )

        self.assertEqual(response.content, "最终回答")
        self.assertEqual(captured, [("profile", 9), ("summary", "止痛药")])
        self.assertEqual(service.last_run_metadata["tool_rounds"], 2)
        self.assertEqual(len(service.last_run_metadata["tool_calls"]), 2)
        self.assertTrue(service.last_run_metadata["used_tools"])
        self.assertFalse(service.last_run_metadata["tool_loop_truncated"])
        self.assertEqual(len(llm.invoke_calls), 0)

    def test_invoke_with_tools_falls_back_after_max_rounds(self):
        llm = FakeLLM(
            bound_responses=[
                FakeResponse(tool_calls=[{"name": "queryUserMedicationSummary", "args": {"focus": "止痛药"}, "id": "call_1"}]),
                FakeResponse(tool_calls=[{"name": "queryUserMedicationSummary", "args": {"focus": "退烧药"}, "id": "call_2"}]),
            ],
            final_response=FakeResponse("收口回答"),
        )
        service = self.build_service(llm)
        service.MAX_TOOL_CALL_ROUNDS = 2
        service.tool_handlers = {
            "queryUserMedicationSummary": lambda focus: f"读取到 {focus}",
        }

        response = service._invoke_with_tools(
            [HumanMessage(content="用户问题：布洛芬和缬沙坦")],
            allowed_tool_names=["queryUserMedicationSummary"],
        )

        self.assertEqual(response.content, "收口回答")
        self.assertTrue(service.last_run_metadata["tool_loop_truncated"])
        self.assertEqual(service.last_run_metadata["tool_rounds"], 2)
        self.assertEqual(len(llm.invoke_calls), 1)

    def test_chat_stream_reuses_multi_round_tool_plan(self):
        llm = FakeLLM(
            bound_responses=[
                FakeResponse(tool_calls=[{"name": "queryUserHealthProfile", "args": {}, "id": "call_1"}]),
                FakeResponse(content="无需继续调用工具"),
            ],
            stream_chunks=["分", "析"],
        )
        service = self.build_service(llm)
        service.tool_handlers = {
            "queryUserHealthProfile": lambda user_id: "读取到用户档案",
        }

        async def collect():
            items = []
            async for chunk in service.chat_stream(
                "m1",
                "我现在能不能吃布洛芬",
                allowed_tool_names=["queryUserHealthProfile"],
                runtime_tool_kwargs={"queryUserHealthProfile": {"user_id": 9}},
            ):
                items.append(chunk)
            return items

        chunks = asyncio.run(collect())

        self.assertEqual("".join(chunks), "分析")
        self.assertTrue(service.last_run_metadata["used_tools"])
        self.assertEqual(service.last_run_metadata["tool_rounds"], 1)
        self.assertEqual(service.memory_service.saved[-1], ("m1", "我现在能不能吃布洛芬", "分析"))
        self.assertTrue(any(isinstance(item, ToolMessage) for item in llm.astream_calls[0]))

    def test_memory_summary_waits_for_batch_threshold(self):
        llm = FakeLLM(final_response=FakeResponse("新摘要"))
        service = self.build_service(llm)
        service.SUMMARY_BATCH_TURNS = 3
        service.memory_service.overflow = [HumanMessage(content="u1"), HumanMessage(content="u2")]

        service._maybe_update_memory_summary("m-batch")

        self.assertEqual(llm.invoke_calls, [])
        self.assertEqual(service.memory_service.summary_updates, [])
        self.assertEqual(service.memory_service.trim_calls, [])

    def test_memory_summary_updates_once_batch_threshold_is_reached(self):
        llm = FakeLLM(final_response=FakeResponse("新摘要"))
        service = self.build_service(llm)
        service.SUMMARY_BATCH_TURNS = 3
        service.memory_service.summary = "旧摘要"
        service.memory_service.overflow = [
            HumanMessage(content="u1"),
            AIMessage(content="a1"),
            HumanMessage(content="u2"),
            AIMessage(content="a2"),
            HumanMessage(content="u3"),
            AIMessage(content="a3"),
        ]

        service._maybe_update_memory_summary("m-batch")

        self.assertEqual(len(llm.invoke_calls), 1)
        self.assertEqual(service.memory_service.summary_updates[-1], ("m-batch", "新摘要"))
        self.assertEqual(service.memory_service.trim_calls[-1], ("m-batch", service.ANSWER_HISTORY_TURNS))


if __name__ == "__main__":
    unittest.main()
