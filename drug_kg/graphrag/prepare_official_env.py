from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official GraphRAG .env from the root project .env.")
    parser.add_argument(
        "--source-env",
        type=Path,
        default=Path(".env"),
        help="Root .env file with existing model settings.",
    )
    parser.add_argument(
        "--target-env",
        type=Path,
        default=Path("drug_kg/graphrag/official_project/.env"),
        help="Official GraphRAG project .env to write.",
    )
    args = parser.parse_args()

    values = dotenv_values(args.source_env)
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or values.get("DASHSCOPE_API_KEY") or "").strip()
    dashscope_base = (
        os.getenv("DASHSCOPE_BASE_URL")
        or values.get("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()
    completion_model = (os.getenv("MODEL_NAME") or values.get("MODEL_NAME") or "qwen3-max").strip()
    embedding_model = (os.getenv("EMBEDDING_MODEL") or values.get("EMBEDDING_MODEL") or "text-embedding-v3").strip()

    content = "\n".join(
        [
            f"GRAPHRAG_COMPLETION_MODEL={completion_model}",
            f"GRAPHRAG_COMPLETION_API_BASE={dashscope_base}",
            f"GRAPHRAG_COMPLETION_API_KEY={dashscope_key or '<YOUR_DASHSCOPE_API_KEY>'}",
            "",
            f"GRAPHRAG_EMBEDDING_MODEL={embedding_model}",
            f"GRAPHRAG_EMBEDDING_API_BASE={dashscope_base}",
            f"GRAPHRAG_EMBEDDING_API_KEY={dashscope_key or '<YOUR_DASHSCOPE_API_KEY>'}",
            "",
        ]
    )

    args.target_env.parent.mkdir(parents=True, exist_ok=True)
    args.target_env.write_text(content, encoding="utf-8")
    print(str(args.target_env))


if __name__ == "__main__":
    main()
