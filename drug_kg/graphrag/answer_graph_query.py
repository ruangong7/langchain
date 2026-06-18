from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME
from drug_kg.graphrag.search_graph_docs import search_graph_docs


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clip(text: str, max_chars: int) -> str:
    text = _norm(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _format_item(doc_type: str, item: dict[str, Any], index: int, max_chars: int) -> str:
    score = float(item.get("score", 0.0))
    doc_id = _norm(item.get("doc_id"))
    text = _clip(_norm(item.get("text")), max_chars)
    return f"[{doc_type}#{index}] score={score:.4f} doc_id={doc_id}\n{text}"


def build_context(*, results: dict[str, list[dict[str, Any]]], max_items_per_type: int, max_chars_per_item: int) -> str:
    lines: list[str] = []
    for doc_type in ("node_docs", "edge_docs", "subgraph_docs"):
        items = results.get(doc_type) or []
        if not items:
            continue
        lines.append(f"## {doc_type}")
        for idx, item in enumerate(items[:max_items_per_type], start=1):
            lines.append(_format_item(doc_type, item, idx, max_chars_per_item))
    return "\n\n".join(lines)


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def generate_answer(*, question: str, context: str, model: str, timeout_s: int) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable")

    system_prompt = (
        "你是一个医疗知识图谱问答助手。"
        "你必须仅基于给定的图谱检索上下文回答，不要编造。"
        "如果证据不足，要明确说“当前检索证据不足”。"
        "优先输出结构化、简洁、可核查的答案。"
        "回答时尽量归纳出涉及的药物、关系类型、证据要点。"
    )
    user_prompt = (
        f"用户问题：{question}\n\n"
        "下面是从知识图谱 GraphRAG 检索到的上下文，请仅依据这些内容作答。\n\n"
        f"{context}\n\n"
        "请按如下格式回答：\n"
        "1. 直接答案\n"
        "2. 证据摘要\n"
        "3. 涉及的关键实体与关系\n"
        "4. 不确定性说明（如果有）"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        _chat_completions_url(base_url),
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout_s,
    )
    resp.raise_for_status()
    obj = resp.json()
    answer = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"answer": answer, "raw_response": obj}


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Answer a question with GraphRAG retrieval + LLM generation")
    parser.add_argument("query", help="Natural language medical question")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("drug_kg/graphrag/index"),
        help="Directory containing graph indexes",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K results per doc type")
    parser.add_argument(
        "--max-items-per-type",
        type=int,
        default=3,
        help="Max retrieved items per doc type to include in prompt context",
    )
    parser.add_argument(
        "--max-chars-per-item",
        type=int,
        default=500,
        help="Max characters per retrieved item included in prompt context",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL_NAME", MODEL_NAME),
        help="LLM model name for answer generation",
    )
    parser.add_argument("--timeout-s", type=int, default=120, help="Generation timeout in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional UTF-8 JSON file path to write retrieval and answer results",
    )
    args = parser.parse_args()

    results = search_graph_docs(query=args.query, index_dir=args.index_dir, top_k=args.top_k)
    context = build_context(
        results=results,
        max_items_per_type=max(1, args.max_items_per_type),
        max_chars_per_item=max(100, args.max_chars_per_item),
    )
    generated = generate_answer(
        question=args.query,
        context=context,
        model=args.model,
        timeout_s=args.timeout_s,
    )

    output = {
        "query": args.query,
        "retrieval": results,
        "context": context,
        "answer": generated["answer"],
    }
    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")

    print("=== Answer ===")
    print(generated["answer"])
    print("\n=== Retrieval Summary ===")
    for doc_type in ("node_docs", "edge_docs", "subgraph_docs"):
        items = results.get(doc_type) or []
        print(f"{doc_type}: {len(items)}")
        for idx, item in enumerate(items[: max(1, args.max_items_per_type)], start=1):
            print(_format_item(doc_type, item, idx, 180))
            print()


if __name__ == "__main__":
    main()
