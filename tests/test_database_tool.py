import unittest

from tools.database_tool import DatabaseTool


class DatabaseToolTests(unittest.TestCase):
    def test_query_user_health_profile_payload_reports_missing_tables(self):
        tool = DatabaseTool.__new__(DatabaseTool)
        tool.capabilities = {"user_health_profile_table": False, "user_medications_table": False}
        tool.get_capabilities = lambda: dict(tool.capabilities)

        payload = tool.query_user_health_profile_payload(9)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "profile_tables_missing")

    def test_format_tool_result_renders_user_health_profile(self):
        payload = {
            "tool": "queryUserHealthProfile",
            "message": "已读取当前用户的健康档案。",
            "records": [
                {
                    "display_name": "张阿姨",
                    "gender": "女",
                    "age": 63,
                    "conditions": ["高血压"],
                    "allergies": ["青霉素"],
                }
            ],
        }

        text = DatabaseTool.format_tool_result(payload)

        self.assertIn("称呼: 张阿姨", text)
        self.assertIn("基础病: 高血压", text)
        self.assertIn("过敏史: 青霉素", text)
        self.assertNotIn("当前用药", text)

    def test_format_tool_result_renders_medication_summary(self):
        payload = {
            "tool": "queryUserMedicationSummary",
            "message": "已读取当前用户登记的用药信息。",
            "records": [
                {"drug_name": "阿莫西林胶囊", "dosage": "1g", "frequency": "每日", "times_per_day": 2, "purpose": "消炎"}
            ],
        }

        text = DatabaseTool.format_tool_result(payload)

        self.assertIn("阿莫西林胶囊 / 1g / 每日 / 消炎 / 每日2次", text)

    def test_load_drug_lexicon_returns_empty_without_legacy_tables(self):
        tool = DatabaseTool.__new__(DatabaseTool)

        self.assertEqual(tool.load_drug_lexicon(), [])


if __name__ == "__main__":
    unittest.main()
