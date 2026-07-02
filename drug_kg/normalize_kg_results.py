"""Normalize extracted KG entities and relations from all result.json files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DRUG_KG_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    ("structured", DRUG_KG_DIR / "structured" / "result.json"),
    ("unstructured", DRUG_KG_DIR / "unstructured" / "result.json"),
    ("pdf", DRUG_KG_DIR / "PDF" / "pdf_kg_result.json"),
)
DEFAULT_OUTPUT_DIR = DRUG_KG_DIR / "normalized"
DEFAULT_ALIAS_FILE = DRUG_KG_DIR / "entity_aliases.json"

ALLOWED_ENTITY_TYPES = {
    "Drug",
    "DrugClass",
    "Food",
    "Disease",
    "SideEffect",
    "Symptom",
    "Indicator",
    "Population",
}
ALLOWED_RELATION_TYPES = {
    "IN_CLASS",
    "INDICATED_FOR",
    "HAS_ADVERSE_REACTION",
    "AFFECTS_INDICATOR",
    "HAS_SYMPTOM",
    "CONTRAINDICATED_FOR",
    "APPLIES_TO",
    "INTERACTS_WITH",
}
SYMMETRIC_RELATIONS = {"INTERACTS_WITH"}

GENERIC_ENTITY_NAMES = {
    "患者",
    "医生",
    "医院",
    "研究",
    "结果",
    "风险",
    "作用",
    "说明",
    "资料",
    "情况",
    "建议",
    "禁忌",
    "慎用",
    "不宜",
    "尚不明确",
    "药物",
    "药品",
    "治疗",
    "用药",
    "临床",
    "实验",
    "观察",
    "相互作用",
    "抑制剂",
    "诱导剂",
}
BAD_ENTITY_TOKENS = {
    "编辑",
    "作者",
    "通讯作者",
    "收稿",
    "基金",
    "参考文献",
    "关键词",
    "摘要",
}
DOSAGE_FORM_SUFFIXES = (
    "缓释片",
    "控释片",
    "肠溶片",
    "分散片",
    "咀嚼片",
    "口服液",
    "注射液",
    "注射剂",
    "颗粒剂",
    "胶囊剂",
    "滴眼液",
    "滴丸",
    "片剂",
    "胶囊",
    "颗粒",
    "软膏",
    "乳膏",
    "喷雾剂",
    "溶液",
    "片",
)
BUILTIN_ALIAS_MAP = {
    "fk506": "他克莫司",
    "tacrolimus": "他克莫司",
    "paxlovid": "奈玛特韦/利托那韦",
    "paxlovid(奈玛特韦片/利托那韦片)": "奈玛特韦/利托那韦",
    "奈玛特韦片/利托那韦片": "奈玛特韦/利托那韦",
    "奈玛特韦/利托那韦片": "奈玛特韦/利托那韦",
    "乙胺碘呋酮": "胺碘酮",
    "胺碘达隆": "胺碘酮",
    "西米替丁": "西咪替丁",
    "甲氰咪胍": "西咪替丁",
    "异搏定": "维拉帕米",
    "维生素b6": "维生素B6",
    "维生素b_6": "维生素B6",
    "cyp3a4": "CYP3A4",
    "cyp2c9": "CYP2C9",
    "cyp2c19": "CYP2C19",
    "p-gp": "P-糖蛋白",
    "p糖蛋白": "P-糖蛋白",
}
NEGATION_PATTERNS = (
    "未见",
    "无明显",
    "无显著",
    "未产生",
    "未发生",
    "不会",
    "不增加",
    "不降低",
    "不影响",
    "不支持",
    "不是增毒",
    "不是相加",
    "无差别",
    "无关",
)


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_alias(alias_map: dict[str, str], alias: Any, canonical: Any) -> None:
    alias_text = norm_text(alias)
    canonical_text = norm_text(canonical)
    if not alias_text or not canonical_text:
        return
    alias_map[alias_text] = canonical_text
    alias_map[alias_text.lower()] = canonical_text


def load_alias_map(path: Path | None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for alias, canonical in BUILTIN_ALIAS_MAP.items():
        add_alias(alias_map, alias, canonical)

    if path is None or not path.is_file():
        return alias_map

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Alias file must be a JSON object: {path}")

    for key, value in raw.items():
        if isinstance(value, list):
            canonical = key
            add_alias(alias_map, canonical, canonical)
            for alias in value:
                add_alias(alias_map, alias, canonical)
        elif isinstance(value, str):
            add_alias(alias_map, key, value)
        else:
            raise ValueError(f"Unsupported alias value for {key!r}: {type(value).__name__}")
    return alias_map


def canonical_entity_name(name: str, alias_map: dict[str, str]) -> str:
    text = norm_text(name)
    text = text.strip(" \t\r\n,，。;；:：、()（）[]【】{}<>《》\"'")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("（", "(").replace("）", ")")
    lowered = text.lower()
    if lowered in alias_map:
        return alias_map[lowered]
    if text in alias_map:
        return alias_map[text]

    for suffix in DOSAGE_FORM_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            text = text[: -len(suffix)]
            break

    lowered = text.lower()
    return alias_map.get(lowered, text)


def entity_key(entity_type: str, canonical_name: str) -> str:
    digest = hashlib.sha1(f"{entity_type}\t{canonical_name}".encode("utf-8")).hexdigest()[:16]
    return f"ent_{digest}"


def relation_key(
    source_entity: dict[str, Any],
    rel_type: str,
    target_entity: dict[str, Any],
    polarity: str,
) -> tuple[str, str, str, str]:
    source_key = source_entity["entity_key"]
    target_key = target_entity["entity_key"]
    if rel_type in SYMMETRIC_RELATIONS and source_key > target_key:
        source_key, target_key = target_key, source_key
    return source_key, rel_type, target_key, polarity


def relation_id(key: tuple[str, str, str, str]) -> str:
    digest = hashlib.sha1("\t".join(key).encode("utf-8")).hexdigest()[:16]
    return f"rel_{digest}"


def is_bad_entity(name: str, entity_type: str) -> bool:
    if not name or len(name) > 40:
        return True
    if name in GENERIC_ENTITY_NAMES:
        return True
    if name.startswith(("-", "_", "/", "\\")) or name.endswith(("-", "_", "/", "\\")):
        return True
    if any(token in name for token in BAD_ENTITY_TOKENS):
        return True
    if re.fullmatch(r"[\d年月日.%+\-_/]+", name):
        return True
    if re.search(r"[。！？!?；;]", name):
        return True
    if entity_type not in ALLOWED_ENTITY_TYPES:
        return True
    return False


def detect_polarity(text: str) -> str:
    evidence = norm_text(text)
    if any(pattern in evidence for pattern in NEGATION_PATTERNS):
        return "negated_or_no_effect"
    return "affirmed"


def load_result(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def normalize_record(
    *,
    dataset: str,
    record: dict[str, Any],
    entity_store: dict[str, dict[str, Any]],
    triple_store: dict[tuple[str, str, str, str], dict[str, Any]],
    stats: Counter,
    alias_map: dict[str, str],
    max_evidence_per_triple: int,
) -> None:
    entities = record.get("entities") if isinstance(record.get("entities"), list) else []
    relations = record.get("relations") if isinstance(record.get("relations"), list) else []
    source_text = norm_text(record.get("source"))
    chunk_id = norm_text(record.get("chunk_index"))
    record_text = norm_text(record.get("text"))

    local_entities: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            stats["bad_entity_shape"] += 1
            continue
        old_id = norm_text(entity.get("id"))
        entity_type = norm_text(entity.get("type") or "Entity")
        name = canonical_entity_name(
            entity.get("canonical_name") or entity.get("name") or "",
            alias_map,
        )
        if not old_id or is_bad_entity(name, entity_type):
            stats["dropped_entities"] += 1
            continue

        key = entity_key(entity_type, name)
        normalized_entity = entity_store.setdefault(
            key,
            {
                "entity_id": key,
                "type": entity_type,
                "canonical_name": name,
                "aliases": [],
                "source_datasets": [],
                "mention_count": 0,
            },
        )
        mention_name = norm_text(entity.get("name") or name)
        if mention_name and mention_name not in normalized_entity["aliases"]:
            normalized_entity["aliases"].append(mention_name)
        if dataset not in normalized_entity["source_datasets"]:
            normalized_entity["source_datasets"].append(dataset)
        normalized_entity["mention_count"] += 1
        local_entities[old_id] = {
            "entity_key": key,
            "type": entity_type,
            "canonical_name": name,
        }

    for relation in relations:
        if not isinstance(relation, dict):
            stats["bad_relation_shape"] += 1
            continue
        rel_type = norm_text(relation.get("type"))
        if rel_type not in ALLOWED_RELATION_TYPES:
            stats["dropped_relation_type"] += 1
            continue
        source_entity = local_entities.get(norm_text(relation.get("source_id")))
        target_entity = local_entities.get(norm_text(relation.get("target_id")))
        if not source_entity or not target_entity:
            stats["dropped_relation_missing_entity"] += 1
            continue
        if source_entity["entity_key"] == target_entity["entity_key"]:
            stats["dropped_relation_self_loop"] += 1
            continue

        props = relation.get("properties") if isinstance(relation.get("properties"), dict) else {}
        evidence = norm_text(
            props.get("raw_text")
            or props.get("evidence")
            or props.get("data")
            or record_text
        )
        polarity = detect_polarity(evidence)
        key = relation_key(source_entity, rel_type, target_entity, polarity)
        rel_id = relation_id(key)
        normalized = triple_store.setdefault(
            key,
            {
                "triple_id": rel_id,
                "head_id": key[0],
                "head_name": entity_store[key[0]]["canonical_name"],
                "head_type": entity_store[key[0]]["type"],
                "relation": rel_type,
                "tail_id": key[2],
                "tail_name": entity_store[key[2]]["canonical_name"],
                "tail_type": entity_store[key[2]]["type"],
                "polarity": polarity,
                "evidence": [],
                "source_datasets": [],
                "source_count": 0,
            },
        )
        if dataset not in normalized["source_datasets"]:
            normalized["source_datasets"].append(dataset)
        normalized["source_count"] += 1
        if evidence and len(normalized["evidence"]) < max_evidence_per_triple:
            normalized["evidence"].append(
                {
                    "dataset": dataset,
                    "source": source_text,
                    "chunk_id": chunk_id,
                    "text": evidence[:1000],
                }
            )
        stats["kept_relations"] += 1


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize KG extraction result files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alias-file",
        type=Path,
        default=DEFAULT_ALIAS_FILE,
        help="Optional entity alias JSON file. Supports alias->canonical or canonical->[aliases].",
    )
    parser.add_argument("--max-evidence-per-triple", type=int, default=20)
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("DATASET", "PATH"),
        help="Add an input result file. Can be passed multiple times.",
    )
    args = parser.parse_args()

    inputs = (
        [(name, Path(path)) for name, path in args.input]
        if args.input
        else list(DEFAULT_INPUTS)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    entity_store: dict[str, dict[str, Any]] = {}
    triple_store: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    alias_map = load_alias_map(args.alias_file)
    dataset_record_counts: dict[str, int] = {}
    relation_type_counts: Counter = Counter()
    entity_type_counts: Counter = Counter()
    polarity_counts: Counter = Counter()
    dataset_relation_counts: Counter = Counter()

    for dataset, path in inputs:
        if not path.is_file():
            stats[f"missing_input:{dataset}"] += 1
            continue
        records = load_result(path)
        dataset_record_counts[dataset] = len(records)
        for record in records:
            normalize_record(
                dataset=dataset,
                record=record,
                entity_store=entity_store,
                triple_store=triple_store,
                stats=stats,
                alias_map=alias_map,
                max_evidence_per_triple=max(1, args.max_evidence_per_triple),
            )

    entities = sorted(entity_store.values(), key=lambda item: (item["type"], item["canonical_name"]))
    triples = sorted(
        triple_store.values(),
        key=lambda item: (item["relation"], item["head_name"], item["tail_name"], item["polarity"]),
    )

    for entity in entities:
        entity_type_counts[entity["type"]] += 1
    for triple in triples:
        relation_type_counts[triple["relation"]] += 1
        polarity_counts[triple["polarity"]] += 1
        for dataset in triple["source_datasets"]:
            dataset_relation_counts[dataset] += 1

    write_jsonl(args.output_dir / "normalized_entities.jsonl", entities)
    write_jsonl(args.output_dir / "normalized_triples.jsonl", triples)

    summary = {
        "inputs": {dataset: str(path) for dataset, path in inputs},
        "alias_file": str(args.alias_file) if args.alias_file else None,
        "alias_entries": len(alias_map),
        "dataset_record_counts": dataset_record_counts,
        "normalized_entities": len(entities),
        "normalized_triples": len(triples),
        "entity_type_counts": dict(entity_type_counts),
        "relation_type_counts": dict(relation_type_counts),
        "polarity_counts": dict(polarity_counts),
        "dataset_relation_counts": dict(dataset_relation_counts),
        "stats": dict(stats),
    }
    (args.output_dir / "normalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
