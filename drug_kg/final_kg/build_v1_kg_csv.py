from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = FINAL_KG_DIR / "neo4j_import"
DEFAULT_OUTPUT_DIR = FINAL_KG_DIR / "neo4j_import_v1"

CLASS_PATTERNS = [
    re.compile(r"(抑制剂|拮抗剂|激动剂|阻滞剂)$"),
    re.compile(r"^ace.*抑制剂$", re.I),
    re.compile(r"^cdk\d+/\d+抑制剂$", re.I),
    re.compile(r".*疗法药物$"),
    re.compile(r".*阻滞剂.*药物$"),
]
DROP_PATTERNS = [
    re.compile(r"(方案|疗法|联合用药)$"),
    re.compile(r"^cyp\d+[a-z0-9\-]*$", re.I),
    re.compile(r"^p-?gp$", re.I),
]
TRIGGER_BY_REL = {
    "INTERACTS_WITH": ["合用", "联用", "同用", "相互作用", "增加", "降低", "影响", "增强", "减弱"],
    "AFFECTS_INDICATOR": ["监测", "升高", "降低", "影响", "血糖", "INR", "血药浓度", "肝功能", "肾功能"],
    "HAS_ADVERSE_REACTION": ["不良反应", "副作用", "导致", "引起", "发生"],
    "INDICATED_FOR": ["用于", "治疗", "适用于", "可用于"],
    "CONTRAINDICATED_FOR": ["禁用", "慎用", "避免用于", "不宜用于"],
    "HAS_SYMPTOM": ["出现", "表现为", "症状"],
    "APPLIES_TO": ["患者", "人群", "老年", "儿童", "孕妇"],
    "IN_CLASS": ["属于", "为", "分类"],
}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;])")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _node_action(row: dict[str, str]) -> tuple[str, str | None]:
    if _norm(row.get("entity_type")) != "Drug":
        return "keep", None
    name = _norm(row.get("canonical_name"))
    if any(pattern.search(name) for pattern in DROP_PATTERNS):
        return "drop", "pseudo_non_drug"
    if any(pattern.search(name) for pattern in CLASS_PATTERNS):
        return "relabel", "drugclass_like"
    return "keep", None


def _new_entity_id(row: dict[str, str], new_label: str) -> str:
    old_id = _norm(row.get(":ID"))
    if "::" not in old_id:
        return old_id
    _, suffix = old_id.split("::", 1)
    return f"{new_label}::{suffix}"


def _pick_sentences(text: str, relation_type: str, limit_chars: int = 500) -> str:
    text = _norm(text)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return text[:limit_chars]

    triggers = TRIGGER_BY_REL.get(relation_type, [])
    selected = [s for s in sentences if any(token in s for token in triggers)]
    if not selected:
        selected = sentences[:2]

    merged = ""
    for sent in selected:
        candidate = merged + sent
        if len(candidate) > limit_chars:
            break
        merged = candidate

    return merged[:limit_chars] if merged else text[:limit_chars]


def _trim_evidence_json(raw: str, relation_type: str) -> tuple[str, int]:
    evidence: list[str]
    try:
        parsed = json.loads(raw) if raw else []
        evidence = parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        evidence = [raw] if raw else []

    trimmed: list[str] = []
    for item in evidence:
        snippet = _pick_sentences(_norm(item), relation_type=relation_type, limit_chars=500)
        if snippet and snippet not in trimmed:
            trimmed.append(snippet)
        if len(trimmed) >= 3:
            break

    return json.dumps(trimmed, ensure_ascii=False), len(trimmed)


def _merge_pipe_values(values: list[str]) -> str:
    parts: list[str] = []
    seen = set()
    for value in values:
        for part in _norm(value).split("|"):
            part = _norm(part)
            if part and part not in seen:
                seen.add(part)
                parts.append(part)
    return "|".join(parts)


def build_v1(nodes: list[dict[str, str]], relations: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    id_map: dict[str, str | None] = {}
    cleaned_nodes: list[dict[str, str]] = []
    relabeled_count = 0
    dropped_nodes = 0

    for row in nodes:
        action, reason = _node_action(row)
        old_id = _norm(row.get(":ID"))
        if action == "drop":
            id_map[old_id] = None
            dropped_nodes += 1
            continue
        new_row = dict(row)
        if action == "relabel":
            relabeled_count += 1
            new_row[":LABEL"] = "DrugClass"
            new_row["entity_type"] = "DrugClass"
            new_id = _new_entity_id(row, "DrugClass")
            new_row[":ID"] = new_id
            id_map[old_id] = new_id
        else:
            id_map[old_id] = old_id
        cleaned_nodes.append(new_row)

    relation_acc: dict[tuple[str, str, str], dict[str, str]] = {}
    dropped_relations = 0
    deduped_interacts = 0

    for row in relations:
        start_id = id_map.get(_norm(row.get(":START_ID")))
        end_id = id_map.get(_norm(row.get(":END_ID")))
        if not start_id or not end_id:
            dropped_relations += 1
            continue

        relation_type = _norm(row.get(":TYPE"))
        new_row = dict(row)
        new_row[":START_ID"] = start_id
        new_row[":END_ID"] = end_id

        trimmed_evidence, trimmed_count = _trim_evidence_json(_norm(row.get("evidence_json")), relation_type)
        new_row["evidence_json"] = trimmed_evidence
        new_row["evidence_count:int"] = str(trimmed_count)

        if relation_type == "INTERACTS_WITH":
            ordered = sorted([start_id, end_id])
            key = (ordered[0], relation_type, ordered[1])
            if key in relation_acc:
                deduped_interacts += 1
                existing = relation_acc[key]
                existing["chunk_ids"] = _merge_pipe_values([existing.get("chunk_ids", ""), new_row.get("chunk_ids", "")])
                existing["source_types"] = _merge_pipe_values([existing.get("source_types", ""), new_row.get("source_types", "")])
                ev_existing = _norm(existing.get("evidence_json"))
                ev_new = _norm(new_row.get("evidence_json"))
                try:
                    merged_ev = json.loads(ev_existing) if ev_existing else []
                except Exception:
                    merged_ev = [ev_existing] if ev_existing else []
                try:
                    new_ev = json.loads(ev_new) if ev_new else []
                except Exception:
                    new_ev = [ev_new] if ev_new else []
                combined = []
                for item in merged_ev + new_ev:
                    item = _norm(item)
                    if item and item not in combined:
                        combined.append(item)
                    if len(combined) >= 3:
                        break
                existing["evidence_json"] = json.dumps(combined, ensure_ascii=False)
                existing["evidence_count:int"] = str(len(combined))
                continue

            new_row[":START_ID"] = ordered[0]
            new_row[":END_ID"] = ordered[1]
            relation_acc[key] = new_row
            continue

        key = (start_id, relation_type, end_id)
        relation_acc[key] = new_row

    cleaned_relations = list(relation_acc.values())

    summary = {
        "input_nodes": len(nodes),
        "input_relations": len(relations),
        "kept_nodes": len(cleaned_nodes),
        "kept_relations": len(cleaned_relations),
        "dropped_nodes": dropped_nodes,
        "dropped_relations": dropped_relations,
        "relabeled_drug_to_drugclass": relabeled_count,
        "deduped_interacts_with": deduped_interacts,
    }
    return cleaned_nodes, cleaned_relations, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fastest usable v1 KG CSVs from current neo4j_import files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    nodes = _load_csv(args.input_dir / "nodes.csv")
    relations = _load_csv(args.input_dir / "relations.csv")
    cleaned_nodes, cleaned_relations, summary = build_v1(nodes, relations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output_dir / "nodes.csv",
        cleaned_nodes,
        [":ID", ":LABEL", "entity_type", "canonical_name", "normalized_name", "aliases", "source_types", "source_records:int", "properties_json"],
    )
    _write_csv(
        args.output_dir / "relations.csv",
        cleaned_relations,
        [":START_ID", ":END_ID", ":TYPE", "relation_id", "source_type", "source_name", "target_type", "target_name", "chunk_ids", "source_types", "evidence_count:int", "evidence_json"],
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
