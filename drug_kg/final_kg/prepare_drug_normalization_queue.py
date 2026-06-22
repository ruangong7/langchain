from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_NODE_CSV = FINAL_KG_DIR / "neo4j_import" / "nodes.csv"
DEFAULT_DRUG_DICT_CSV = FINAL_KG_DIR / "drug_dict" / "drug_dictionary.csv"
DEFAULT_OUTPUT_DIR = FINAL_KG_DIR / "drug_normalization"

SPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[\(\)（）\[\]【】]")
DOSE_SUFFIX_RE = re.compile(
    r"(肠溶片|分散片|缓释片|控释片|咀嚼片|泡腾片|片|胶囊|颗粒|滴丸|丸|散|口服液|糖浆|注射液|注射剂|粉针剂|针|滴眼液|乳膏|软膏|贴片|喷雾剂|吸入剂)$"
)
SPEC_RE = re.compile(r"(\d+(\.\d+)?\s*(mg|g|ml|iu|μg|ug|%)($|/))", re.I)
PSEUDO_DRUG_PATTERNS = [
    re.compile(r"(抑制剂|激动剂|拮抗剂|阻滞剂|制剂)$"),
    re.compile(r"(方案|疗法|联合用药|治疗方案)$"),
    re.compile(r"^cyp\d+[a-z0-9]*", re.I),
    re.compile(r"^p-?gp$", re.I),
]


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_text(value: str) -> str:
    value = _norm(value)
    value = SPACE_RE.sub("", value)
    value = BRACKET_RE.sub("", value)
    return value


def _strip_suffixes(name: str) -> str:
    name = _clean_text(name)
    name = SPEC_RE.sub("", name)
    name = DOSE_SUFFIX_RE.sub("", name)
    return name


def _is_pseudo_drug(name: str) -> bool:
    text = _clean_text(name)
    return any(pattern.search(text) for pattern in PSEUDO_DRUG_PATTERNS)


def _load_drug_dict(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    alias_map: dict[str, str] = {}
    canonical_rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = _norm(row.get("canonical_name"))
            if not canonical:
                continue
            canonical_rows[canonical] = row
            aliases = [_norm(x) for x in _norm(row.get("aliases")).split("|") if _norm(x)]
            aliases.append(canonical)
            for alias in aliases:
                alias_map[_clean_text(alias).lower()] = canonical
                stripped = _strip_suffixes(alias)
                if stripped:
                    alias_map[_clean_text(stripped).lower()] = canonical
    return alias_map, canonical_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Drug normalization queues from nodes.csv and internal drug dictionary.")
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--drug-dict-csv", type=Path, default=DEFAULT_DRUG_DICT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    alias_map, _ = _load_drug_dict(args.drug_dict_csv)

    auto_rule: list[dict[str, Any]] = []
    auto_dict: list[dict[str, Any]] = []
    pseudo_drug: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []

    with args.node_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _norm(row.get("entity_type")) != "Drug":
                continue

            canonical_name = _norm(row.get("canonical_name"))
            aliases = [_norm(x) for x in _norm(row.get("aliases")).split("|") if _norm(x)]
            source_records = _norm(row.get("source_records:int"))

            record = {
                "entity_id": _norm(row.get(":ID")),
                "canonical_name": canonical_name,
                "aliases": "|".join(aliases),
                "source_records": source_records,
                "suggested_canonical": "",
                "reason": "",
            }

            if _is_pseudo_drug(canonical_name):
                record["reason"] = "pseudo_drug_like"
                pseudo_drug.append(record)
                continue

            dict_hit = alias_map.get(_clean_text(canonical_name).lower())
            if dict_hit and dict_hit != canonical_name:
                record["suggested_canonical"] = dict_hit
                record["reason"] = "dictionary_match"
                auto_dict.append(record)
                continue

            stripped = _strip_suffixes(canonical_name)
            if stripped and stripped != canonical_name:
                dict_hit = alias_map.get(_clean_text(stripped).lower())
                record["suggested_canonical"] = dict_hit or stripped
                record["reason"] = "suffix_stripped"
                auto_rule.append(record)
                continue

            if len(canonical_name) > 40 or SPEC_RE.search(canonical_name):
                record["reason"] = "long_or_spec_like"
                manual_review.append(record)
                continue

            for alias in aliases:
                dict_hit = alias_map.get(_clean_text(alias).lower())
                if dict_hit and dict_hit != canonical_name:
                    record["suggested_canonical"] = dict_hit
                    record["reason"] = "alias_dictionary_match"
                    auto_dict.append(record)
                    break
            else:
                record["reason"] = "manual_review"
                manual_review.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = {
        "drug_auto_rule.csv": auto_rule,
        "drug_auto_dict.csv": auto_dict,
        "drug_pseudo_drug.csv": pseudo_drug,
        "drug_manual_review.csv": manual_review,
    }
    fieldnames = ["entity_id", "canonical_name", "aliases", "source_records", "suggested_canonical", "reason"]
    for name, rows in categories.items():
        _write_csv(args.output_dir / name, rows, fieldnames)

    summary = {
        "auto_rule": len(auto_rule),
        "auto_dict": len(auto_dict),
        "pseudo_drug": len(pseudo_drug),
        "manual_review": len(manual_review),
        "total_drug_nodes": len(auto_rule) + len(auto_dict) + len(pseudo_drug) + len(manual_review),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
