"""Database access helpers for drug and user health data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pymysql

from config import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER

logger = logging.getLogger(__name__)


class DatabaseTool:
    """Database query helper."""

    def __init__(self):
        self.connection = None
        self._connect()
        self._ensure_health_profile_tables()
        self._migrate_legacy_health_profile_columns()

    def _connect(self):
        try:
            self.connection = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
            )
            logger.info("数据库连接成功")
        except Exception as exc:
            logger.error("数据库连接失败: %s", exc, exc_info=True)
            raise

    def _ensure_health_profile_tables(self) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_health_profile (
                        user_id BIGINT PRIMARY KEY,
                        display_name VARCHAR(64) DEFAULT '',
                        gender VARCHAR(16) DEFAULT '',
                        age INT NULL,
                        height_cm DECIMAL(6,2) NULL,
                        weight_kg DECIMAL(6,2) NULL,
                        is_pregnant TINYINT(1) NOT NULL DEFAULT 0,
                        is_breastfeeding TINYINT(1) NOT NULL DEFAULT 0,
                        conditions_text TEXT NULL,
                        allergies_text TEXT NULL,
                        notes TEXT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_medications (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        user_id BIGINT NOT NULL,
                        drug_name VARCHAR(128) NOT NULL,
                        dosage VARCHAR(64) DEFAULT '',
                        purpose VARCHAR(128) DEFAULT '',
                        frequency VARCHAR(64) DEFAULT '',
                        times_per_day INT NULL,
                        administration_time VARCHAR(64) DEFAULT '',
                        start_date DATE NULL,
                        end_date DATE NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_medications_user_id (user_id)
                    )
                    """
                )

                profile_columns = self._get_table_columns("user_health_profile", cursor)
                required_profile_columns = {
                    "display_name": "ALTER TABLE user_health_profile ADD COLUMN display_name VARCHAR(64) DEFAULT ''",
                    "gender": "ALTER TABLE user_health_profile ADD COLUMN gender VARCHAR(16) DEFAULT ''",
                    "age": "ALTER TABLE user_health_profile ADD COLUMN age INT NULL",
                    "height_cm": "ALTER TABLE user_health_profile ADD COLUMN height_cm DECIMAL(6,2) NULL",
                    "weight_kg": "ALTER TABLE user_health_profile ADD COLUMN weight_kg DECIMAL(6,2) NULL",
                    "is_pregnant": "ALTER TABLE user_health_profile ADD COLUMN is_pregnant TINYINT(1) NOT NULL DEFAULT 0",
                    "is_breastfeeding": "ALTER TABLE user_health_profile ADD COLUMN is_breastfeeding TINYINT(1) NOT NULL DEFAULT 0",
                    "conditions_text": "ALTER TABLE user_health_profile ADD COLUMN conditions_text TEXT NULL",
                    "allergies_text": "ALTER TABLE user_health_profile ADD COLUMN allergies_text TEXT NULL",
                    "notes": "ALTER TABLE user_health_profile ADD COLUMN notes TEXT NULL",
                    "created_at": "ALTER TABLE user_health_profile ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "updated_at": "ALTER TABLE user_health_profile ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
                }
                for column_name, statement in required_profile_columns.items():
                    if column_name not in profile_columns:
                        cursor.execute(statement)

                medication_columns = self._get_table_columns("user_medications", cursor)
                required_medication_columns = {
                    "dosage": "ALTER TABLE user_medications ADD COLUMN dosage VARCHAR(64) DEFAULT ''",
                    "purpose": "ALTER TABLE user_medications ADD COLUMN purpose VARCHAR(128) DEFAULT ''",
                    "frequency": "ALTER TABLE user_medications ADD COLUMN frequency VARCHAR(64) DEFAULT ''",
                    "times_per_day": "ALTER TABLE user_medications ADD COLUMN times_per_day INT NULL",
                    "administration_time": "ALTER TABLE user_medications ADD COLUMN administration_time VARCHAR(64) DEFAULT ''",
                    "start_date": "ALTER TABLE user_medications ADD COLUMN start_date DATE NULL",
                    "end_date": "ALTER TABLE user_medications ADD COLUMN end_date DATE NULL",
                }
                for column_name, statement in required_medication_columns.items():
                    if column_name not in medication_columns:
                        cursor.execute(statement)
            self.connection.commit()
            logger.info("健康档案相关数据表已检查完成")
        except Exception as exc:
            self.connection.rollback()
            logger.error("健康档案相关表初始化失败: %s", exc, exc_info=True)
            raise

    def _migrate_legacy_health_profile_columns(self) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SHOW TABLES LIKE 'user_conditions'")
                has_conditions = bool(cursor.fetchone())
                cursor.execute("SHOW TABLES LIKE 'user_allergies'")
                has_allergies = bool(cursor.fetchone())

                if has_conditions:
                    cursor.execute(
                        """
                        SELECT user_id, GROUP_CONCAT(condition_name SEPARATOR '\n') AS conditions_text
                        FROM user_conditions
                        GROUP BY user_id
                        """
                    )
                    for row in cursor.fetchall():
                        cursor.execute(
                            """
                            INSERT INTO user_health_profile (user_id, conditions_text)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE
                                conditions_text = CASE
                                    WHEN user_health_profile.conditions_text IS NULL OR user_health_profile.conditions_text = ''
                                    THEN VALUES(conditions_text)
                                    ELSE user_health_profile.conditions_text
                                END
                            """,
                            (row["user_id"], row["conditions_text"] or ""),
                        )

                if has_allergies:
                    cursor.execute(
                        """
                        SELECT user_id, GROUP_CONCAT(allergen SEPARATOR '\n') AS allergies_text
                        FROM user_allergies
                        GROUP BY user_id
                        """
                    )
                    for row in cursor.fetchall():
                        cursor.execute(
                            """
                            INSERT INTO user_health_profile (user_id, allergies_text)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE
                                allergies_text = CASE
                                    WHEN user_health_profile.allergies_text IS NULL OR user_health_profile.allergies_text = ''
                                    THEN VALUES(allergies_text)
                                    ELSE user_health_profile.allergies_text
                                END
                            """,
                            (row["user_id"], row["allergies_text"] or ""),
                        )
            self.connection.commit()
        except Exception as exc:
            self.connection.rollback()
            logger.warning("旧健康档案表迁移已跳过: %s", exc)

    def query_real_drug_database(self, drug_name: str) -> str:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM real_drug WHERE drug_name LIKE %s", (f"%{drug_name}%",))
                results = cursor.fetchall()
            if not results:
                return f"未找到药品 {drug_name} 的相关信息。"

            result_text = f"找到 {len(results)} 条关于 {drug_name} 的记录：\n\n"
            for index, row in enumerate(results, start=1):
                result_text += f"记录 {index}:\n"
                for key, value in row.items():
                    if value:
                        result_text += f"  {key}: {value}\n"
                result_text += "\n"
            return result_text
        except Exception as exc:
            logger.error("查询 real_drug 失败: %s", exc, exc_info=True)
            return f"查询 real_drug 失败: {exc}"

    def query_joint_data(self, question: str) -> str:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM yinshi WHERE question LIKE %s OR answer LIKE %s",
                    (f"%{question}%", f"%{question}%"),
                )
                yinshi_results = cursor.fetchall()

                result_text = f"联合查询结果（问题：{question}）：\n\n"
                if not yinshi_results:
                    return result_text + "未找到相关的用户用药信息。"

                result_text += "=== 用户用药信息（yinshi）===\n"
                for row in yinshi_results:
                    drug_name = row.get("drug_name", "")
                    result_text += f"药物名称: {drug_name}\n"
                    for key, value in row.items():
                        if key != "drug_name" and value:
                            result_text += f"  {key}: {value}\n"
                    result_text += "\n"

                    if not drug_name:
                        continue

                    cursor.execute("SELECT * FROM real_drug WHERE drug_name LIKE %s", (f"%{drug_name}%",))
                    real_results = cursor.fetchall()
                    if not real_results:
                        continue

                    result_text += f"=== {drug_name} 的详细信息（real_drug）===\n"
                    for real_row in real_results:
                        for key, value in real_row.items():
                            if value:
                                result_text += f"  {key}: {value}\n"
                    result_text += "\n"

            return result_text
        except Exception as exc:
            logger.error("联合查询失败: %s", exc, exc_info=True)
            return f"联合查询失败: {exc}"

    def load_drug_lexicon(self, refresh: bool = False) -> List[str]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT drug_name
                    FROM real_drug
                    WHERE drug_name IS NOT NULL AND drug_name <> ''
                    """
                )
                rows = cursor.fetchall()
            names = {
                str(row.get("drug_name", "")).strip()
                for row in rows
                if row.get("drug_name")
            }
            return sorted(names, key=len, reverse=True)
        except Exception as exc:
            logger.error("加载药品词表失败: %s", exc, exc_info=True)
            return []

    def get_user_health_profile(self, user_id: int) -> Dict[str, Any]:
        self._ensure_health_profile_tables()

        profile: Dict[str, Any] = {
            "user_id": int(user_id),
            "display_name": "",
            "gender": "",
            "age": None,
            "height_cm": None,
            "weight_kg": None,
            "is_pregnant": False,
            "is_breastfeeding": False,
            "conditions": [],
            "allergies": [],
            "medications": [],
            "notes": "",
        }

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, display_name, gender, age, height_cm, weight_kg,
                       is_pregnant, is_breastfeeding, conditions_text, allergies_text, notes
                FROM user_health_profile
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if row:
                profile.update(
                    {
                        "display_name": str(row.get("display_name") or ""),
                        "gender": str(row.get("gender") or ""),
                        "age": row.get("age"),
                        "height_cm": row.get("height_cm"),
                        "weight_kg": row.get("weight_kg"),
                        "is_pregnant": bool(row.get("is_pregnant")),
                        "is_breastfeeding": bool(row.get("is_breastfeeding")),
                        "conditions": self._split_multiline_text(row.get("conditions_text")),
                        "allergies": self._split_multiline_text(row.get("allergies_text")),
                        "notes": str(row.get("notes") or ""),
                    }
                )

            cursor.execute(
                """
                SELECT drug_name, dosage, purpose, frequency, times_per_day,
                       administration_time, start_date, end_date
                FROM user_medications
                WHERE user_id = %s
                ORDER BY id ASC
                """,
                (user_id,),
            )
            profile["medications"] = [
                {
                    "drug_name": str(item.get("drug_name") or "").strip(),
                    "dosage": str(item.get("dosage") or "").strip(),
                    "purpose": str(item.get("purpose") or "").strip(),
                    "frequency": str(item.get("frequency") or "").strip(),
                    "times_per_day": item.get("times_per_day"),
                    "administration_time": str(item.get("administration_time") or "").strip(),
                    "start_date": item.get("start_date").isoformat() if item.get("start_date") else None,
                    "end_date": item.get("end_date").isoformat() if item.get("end_date") else None,
                }
                for item in cursor.fetchall()
                if str(item.get("drug_name") or "").strip()
            ]
        return profile

    def upsert_user_health_profile(self, user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_health_profile_tables()

        profile = payload or {}
        conditions = self._clean_string_list(profile.get("conditions"))
        allergies = self._clean_string_list(profile.get("allergies"))
        medications = self._clean_medications(profile.get("medications"))

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_health_profile (
                        user_id, display_name, gender, age, height_cm, weight_kg,
                        is_pregnant, is_breastfeeding, conditions_text, allergies_text, notes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        display_name = VALUES(display_name),
                        gender = VALUES(gender),
                        age = VALUES(age),
                        height_cm = VALUES(height_cm),
                        weight_kg = VALUES(weight_kg),
                        is_pregnant = VALUES(is_pregnant),
                        is_breastfeeding = VALUES(is_breastfeeding),
                        conditions_text = VALUES(conditions_text),
                        allergies_text = VALUES(allergies_text),
                        notes = VALUES(notes),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id,
                        str(profile.get("display_name") or "").strip(),
                        str(profile.get("gender") or "").strip(),
                        profile.get("age"),
                        profile.get("height_cm"),
                        profile.get("weight_kg"),
                        1 if bool(profile.get("is_pregnant")) else 0,
                        1 if bool(profile.get("is_breastfeeding")) else 0,
                        "\n".join(conditions),
                        "\n".join(allergies),
                        str(profile.get("notes") or "").strip(),
                    ),
                )

                cursor.execute("DELETE FROM user_medications WHERE user_id = %s", (user_id,))
                if medications:
                    cursor.executemany(
                        """
                        INSERT INTO user_medications (
                            user_id, drug_name, dosage, purpose, frequency,
                            times_per_day, administration_time, start_date, end_date
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                user_id,
                                item["drug_name"],
                                item["dosage"],
                                item["purpose"],
                                item["frequency"],
                                item["times_per_day"],
                                item["administration_time"],
                                item["start_date"],
                                item["end_date"],
                            )
                            for item in medications
                        ],
                    )
            self.connection.commit()
        except Exception as exc:
            self.connection.rollback()
            logger.error("保存健康档案失败: %s", exc, exc_info=True)
            raise
        return self.get_user_health_profile(user_id)

    def build_user_personal_context(self, user_id: int) -> str:
        profile = self.get_user_health_profile(user_id)
        has_profile = any(
            [
                profile.get("display_name"),
                profile.get("gender"),
                profile.get("age") is not None,
                profile.get("conditions"),
                profile.get("allergies"),
                profile.get("medications"),
                profile.get("notes"),
            ]
        )
        if not has_profile:
            return ""

        lines = ["[用户健康档案]"]
        if profile.get("display_name"):
            lines.append(f"称呼: {profile['display_name']}")
        if profile.get("gender"):
            lines.append(f"性别: {profile['gender']}")
        if profile.get("age") is not None:
            lines.append(f"年龄: {profile['age']}")
        if profile.get("height_cm") is not None:
            lines.append(f"身高(cm): {profile['height_cm']}")
        if profile.get("weight_kg") is not None:
            lines.append(f"体重(kg): {profile['weight_kg']}")
        if profile.get("is_pregnant"):
            lines.append("状态: 妊娠期")
        if profile.get("is_breastfeeding"):
            lines.append("状态: 哺乳期")
        if profile.get("conditions"):
            lines.append("基础病: " + "、".join(profile["conditions"]))
        if profile.get("allergies"):
            lines.append("过敏史: " + "、".join(profile["allergies"]))
        if profile.get("medications"):
            med_lines = []
            for item in profile["medications"]:
                parts = [item["drug_name"]]
                if item.get("dosage"):
                    parts.append(item["dosage"])
                if item.get("frequency"):
                    parts.append(item["frequency"])
                if item.get("times_per_day"):
                    parts.append(f"每日{item['times_per_day']}次")
                if item.get("administration_time"):
                    parts.append(item["administration_time"])
                if item.get("purpose"):
                    parts.append(f"用途:{item['purpose']}")
                if item.get("start_date") or item.get("end_date"):
                    parts.append(f"{item.get('start_date') or '未设开始'} - {item.get('end_date') or '持续中'}")
                med_lines.append(" / ".join(parts))
            lines.append("当前用药: " + "；".join(med_lines))
        if profile.get("notes"):
            lines.append("备注: " + profile["notes"])
        lines.append("回答时请优先结合这些个体信息评估药物相互作用、基础病冲突、过敏风险和特殊人群风险。")
        return "\n".join(lines)

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")

    @staticmethod
    def _get_table_columns(table_name: str, cursor) -> set[str]:
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        return {str(row.get("Field") or "").strip() for row in cursor.fetchall()}

    @staticmethod
    def _clean_string_list(values: Any) -> List[str]:
        result: List[str] = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _clean_medications(values: Any) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for value in values or []:
            if not isinstance(value, dict):
                continue
            drug_name = str(value.get("drug_name") or "").strip()
            if not drug_name:
                continue
            result.append(
                {
                    "drug_name": drug_name,
                    "dosage": str(value.get("dosage") or "").strip(),
                    "purpose": str(value.get("purpose") or "").strip(),
                    "frequency": str(value.get("frequency") or "").strip(),
                    "times_per_day": DatabaseTool._clean_optional_int(value.get("times_per_day")),
                    "administration_time": str(value.get("administration_time") or "").strip(),
                    "start_date": DatabaseTool._clean_optional_date(value.get("start_date")),
                    "end_date": DatabaseTool._clean_optional_date(value.get("end_date")),
                }
            )
        return result

    @staticmethod
    def _split_multiline_text(value: Any) -> List[str]:
        raw = str(value or "")
        parts = [item.strip() for item in raw.replace("，", "\n").replace("、", "\n").splitlines()]
        return [item for item in parts if item]

    @staticmethod
    def _clean_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_optional_date(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None
