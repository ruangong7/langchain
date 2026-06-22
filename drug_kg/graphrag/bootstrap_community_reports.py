from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from graphrag.index.operations.finalize_community_reports import finalize_community_reports


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


def _clip(text: str, max_chars: int) -> str:
    text = _norm(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_bootstrap_reports(
    *,
    communities: pd.DataFrame,
    entities: pd.DataFrame,
    text_units: pd.DataFrame,
) -> pd.DataFrame:
    entity_title_by_id = {
        _norm(row["id"]): _norm(row["title"])
        for _, row in entities.iterrows()
    }
    text_by_id = {
        _norm(row["id"]): _norm(row["text"])
        for _, row in text_units.iterrows()
    }

    report_rows: list[dict[str, Any]] = []
    for _, row in communities.iterrows():
        community_id = int(row["community"])
        level = int(row["level"])
        entity_ids = [_norm(item) for item in _to_list(row.get("entity_ids")) if _norm(item)]
        relationship_ids = [_norm(item) for item in _to_list(row.get("relationship_ids")) if _norm(item)]
        text_unit_ids = [_norm(item) for item in _to_list(row.get("text_unit_ids")) if _norm(item)]

        top_entities = [entity_title_by_id.get(entity_id, entity_id) for entity_id in entity_ids[:8]]
        snippets = [_clip(text_by_id.get(text_unit_id, ""), 280) for text_unit_id in text_unit_ids[:3] if _norm(text_by_id.get(text_unit_id, ""))]
        findings = []
        if top_entities:
            findings.append(f"核心实体包括：{', '.join(top_entities[:5])}")
        findings.append(f"社区包含 {len(entity_ids)} 个实体、{len(relationship_ids)} 条关系、{len(text_unit_ids)} 个文本单元")
        if snippets:
            findings.append(f"代表证据摘要：{snippets[0]}")

        summary_parts = [
            f"社区 {community_id} 位于层级 {level}",
            f"包含 {len(entity_ids)} 个实体",
            f"高频核心实体：{', '.join(top_entities[:5])}" if top_entities else "",
        ]
        summary = "；".join([part for part in summary_parts if part])

        full_content_lines = [
            f"# Community {community_id}",
            f"层级：{level}",
            f"实体数：{len(entity_ids)}",
            f"关系数：{len(relationship_ids)}",
            f"文本单元数：{len(text_unit_ids)}",
        ]
        if top_entities:
            full_content_lines.append(f"核心实体：{', '.join(top_entities[:10])}")
        if snippets:
            full_content_lines.append("代表证据：")
            full_content_lines.extend([f"- {snippet}" for snippet in snippets])
        full_content = "\n".join(full_content_lines)

        full_content_json = json.dumps(
            {
                "community": community_id,
                "level": level,
                "entity_count": len(entity_ids),
                "relationship_count": len(relationship_ids),
                "text_unit_count": len(text_unit_ids),
                "top_entities": top_entities[:10],
                "snippets": snippets,
            },
            ensure_ascii=False,
        )

        report_rows.append(
            {
                "community": community_id,
                "level": level,
                "title": f"Community {community_id}",
                "summary": summary,
                "full_content": full_content,
                "rank": float(max(1, int(row.get("size", len(entity_ids)) or len(entity_ids) or 1))),
                "rating_explanation": "Bootstrap community report generated from existing KG structure and text units.",
                "findings": findings,
                "full_content_json": full_content_json,
            }
        )

    reports_df = pd.DataFrame(report_rows)
    return finalize_community_reports(reports_df, communities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap official GraphRAG community_reports without LLM.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("drug_kg/graphrag/official_byog/output"),
        help="Directory containing official GraphRAG parquet outputs.",
    )
    args = parser.parse_args()

    communities_path = args.output_dir / "communities.parquet"
    entities_path = args.output_dir / "entities.parquet"
    text_units_path = args.output_dir / "text_units.parquet"
    if not communities_path.exists():
        raise FileNotFoundError(f"Missing {communities_path}")

    communities = pd.read_parquet(communities_path)
    entities = pd.read_parquet(entities_path)
    text_units = pd.read_parquet(text_units_path)

    community_reports = build_bootstrap_reports(
        communities=communities,
        entities=entities,
        text_units=text_units,
    )
    out_path = args.output_dir / "community_reports.parquet"
    community_reports.to_parquet(out_path, index=False)
    print(json.dumps({"community_reports": len(community_reports), "path": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
