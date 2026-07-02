"""MySQL-backed user registration and login service."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict

import pymysql

from config import (
    AUTH_TOKEN_SECRET,
    AUTH_TOKEN_TTL_SECONDS,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


logger = logging.getLogger(__name__)


class AuthError(Exception):
    """User-facing authentication error."""


class UserAuthService:
    """Register and authenticate users stored in the users table."""

    def __init__(self) -> None:
        self._ensure_users_table()

    def _connect(self):
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _ensure_users_table(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            username VARCHAR(64) NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            UNIQUE KEY uq_users_username (username)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                    columns = self._get_table_columns(cursor)
                    required_columns = {
                        "password_hash": "ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) NOT NULL",
                        "created_at": "ALTER TABLE users ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
                        "updated_at": "ALTER TABLE users ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
                    }
                    for column_name, statement in required_columns.items():
                        if column_name not in columns:
                            cursor.execute(statement)

                    cursor.execute(
                        """
                        SELECT COUNT(1) AS count
                        FROM information_schema.statistics
                        WHERE table_schema = %s
                          AND table_name = 'users'
                          AND index_name = 'uq_users_username'
                        """,
                        (MYSQL_DATABASE,),
                    )
                    if int((cursor.fetchone() or {}).get("count") or 0) == 0:
                        cursor.execute("ALTER TABLE users ADD UNIQUE KEY uq_users_username (username)")
            logger.info("用户认证表 users 已检查完成")
        except Exception as exc:
            logger.error("用户认证表 users 初始化失败: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _get_table_columns(cursor) -> set[str]:
        cursor.execute("SHOW COLUMNS FROM users")
        return {str(row.get("Field") or "") for row in cursor.fetchall()}

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = str(username or "").strip().lower()
        if len(normalized) < 3:
            raise AuthError("用户名至少需要 3 个字符")
        if len(normalized) > 64:
            raise AuthError("用户名不能超过 64 个字符")
        return normalized

    @staticmethod
    def _validate_password(password: str) -> str:
        value = str(password or "")
        if len(value) < 6:
            raise AuthError("密码至少需要 6 个字符")
        if len(value) > 128:
            raise AuthError("密码不能超过 128 个字符")
        return value

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        iterations = 120_000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        try:
            algorithm, iterations, salt_hex, digest_hex = password_hash.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            )
            return hmac.compare_digest(digest.hex(), digest_hex)
        except Exception:
            return False

    @staticmethod
    def _public_user(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "created_at": str(row.get("created_at", "")),
            "updated_at": str(row.get("updated_at", "")),
        }

    def _make_token(self, user_id: int, username: str) -> Dict[str, Any]:
        expires_at = int(time.time()) + AUTH_TOKEN_TTL_SECONDS
        payload = {
            "uid": int(user_id),
            "username": username,
            "exp": expires_at,
            "nonce": secrets.token_urlsafe(8),
        }
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_json).decode("ascii").rstrip("=")
        signature = hmac.new(
            AUTH_TOKEN_SECRET.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        return {"token": f"{payload_b64}.{signature_b64}", "expires_at": expires_at}

    @staticmethod
    def _urlsafe_b64decode(data: str) -> bytes:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))

    def verify_token(self, token: str) -> Dict[str, Any]:
        raw_token = str(token or "").strip()
        if not raw_token or "." not in raw_token:
            raise AuthError("登录状态无效，请重新登录")

        payload_b64, signature_b64 = raw_token.split(".", 1)
        expected_signature = hmac.new(
            AUTH_TOKEN_SECRET.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = self._urlsafe_b64decode(signature_b64)
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise AuthError("登录状态无效，请重新登录")

        try:
            payload = json.loads(self._urlsafe_b64decode(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise AuthError("登录状态无效，请重新登录") from exc

        expires_at = int(payload.get("exp", 0) or 0)
        if expires_at <= int(time.time()):
            raise AuthError("登录状态已过期，请重新登录")

        return payload

    def register(self, username: str, password: str) -> Dict[str, Any]:
        normalized = self._normalize_username(username)
        checked_password = self._validate_password(password)
        password_hash = self._hash_password(checked_password)

        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                        (normalized, password_hash),
                    )
                    user_id = cursor.lastrowid
                    cursor.execute("SELECT id, username, created_at, updated_at FROM users WHERE id = %s", (user_id,))
                    row = cursor.fetchone()
        except pymysql.err.IntegrityError as exc:
            raise AuthError("用户名已存在") from exc

        token = self._make_token(int(row["id"]), row["username"])
        return {"user": self._public_user(row), **token}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        normalized = self._normalize_username(username)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (normalized,))
                row = cursor.fetchone()

        if row is None or not self._verify_password(str(password or ""), row["password_hash"]):
            raise AuthError("用户名或密码不正确")

        token = self._make_token(int(row["id"]), row["username"])
        return {"user": self._public_user(row), **token}
