from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FINAL_KG_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = FINAL_KG_DIR.parent / "structured" / "result.json"
DEFAULT_OUTPUT_DIR = FINAL_KG_DIR / "drug_dict"

DOSE_SUFFIXES = [
    "肠溶片",
    "分散片",
    "缓释片",
    "控释片",
    "咀嚼片",
    "泡腾片",
    "片",
    "胶囊",
    "颗粒",
    "滴丸",
    "丸",
    "散",
    "口服液",
    "糖浆",
    "注射液",
    "注射剂",
    "粉针剂",
    "针",
    "滴眼液",
    "乳膏",
    "软膏",
    "贴片",
    "喷雾剂",
    "吸入剂",
]
SPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[\(\)（）\[\]【】]")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_name(value: str) -> str:
    value = _norm(value)
    value = SPACE_RE.sub("", value)
    value = BRACKET_RE.sub("", value)
    return value


def _strip_dose_suffix(name: str) -> str:
    name = _clean_name(name)
    for suffix in sorted(DOSE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _choose_canonical(name: str, canonical_name: str, ingredient: str) -> str:
    canonical_name = _clean_name(canonical_name)
    name = _clean_name(name)

    stripped = _strip_dose_suffix(canonical_name or name)
    return stripped or canonical_name or name


def build_dict(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    drug_map: dict[str, dict[str, Any]] = {}
    alias_to_canonical: dict[str, str] = {}
    collisions: dict[str, set[str]] = defaultdict(set)

    for record in records:
        for entity in record.get("entities") or []:
            if not isinstance(entity, dict) or _norm(entity.get("type")) != "Drug":
                continue

            props = entity.get("properties") or {}
            if not isinstance(props, dict):
                props = {}

            raw_name = _clean_name(entity.get("name"))
            raw_canonical = _clean_name(entity.get("canonical_name") or raw_name)
            ingredient = _clean_name(props.get("ingredient"))
            canonical = _choose_canonical(raw_name, raw_canonical, ingredient)
            if not canonical:
                continue

            item = drug_map.setdefault(
                canonical,
                {
                    "canonical_name": canonical,
                    "aliases": set(),
                    "ingredients": set(),
                    "approval_numbers": set(),
                    "example_instructions": set(),
                    "record_count": 0,
                },
            )
            item["record_count"] += 1

            for alias in (raw_name, raw_canonical, ingredient, _strip_dose_suffix(raw_name), _strip_dose_suffix(raw_canonical)):
                alias = _clean_name(alias)
                if alias:
                    item["aliases"].add(alias)
                    collisions[alias].add(canonical)

            if ingredient and len(ingredient) <= 40 and "成份" not in ingredient and "辅料" not in ingredient:
                item["ingredients"].add(ingredient)
            approval_number = _norm(props.get("approval_number"))
            if approval_number:
                item["approval_numbers"].add(approval_number)
            instruction = _norm(props.get("instruction"))
            if instruction:
                item["example_instructions"].add(instruction[:200])

    for alias, canonicals in collisions.items():
        if len(canonicals) == 1:
            alias_to_canonical[alias] = next(iter(canonicals))

    dict_rows: list[dict[str, Any]] = []
    for canonical, item in sorted(drug_map.items()):
        dict_rows.append(
            {
                "canonical_name": canonical,
                "aliases": sorted(item["aliases"]),
                "ingredients": sorted(item["ingredients"]),
                "approval_numbers": sorted(item["approval_numbers"]),
                "example_instructions": sorted(item["example_instructions"])[:3],
                "record_count": item["record_count"],
            }
        )

    summary = {
        "canonical_drugs": len(dict_rows),
        "alias_entries": len(alias_to_canonical),
        "ambiguous_aliases": sum(1 for v in collisions.values() if len(v) > 1),
    }
    return dict_rows, alias_to_canonical, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_name",
        "aliases",
        "ingredients",
        "approval_numbers",
        "example_instructions",
        "record_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "canonical_name": row["canonical_name"],
                    "aliases": "|".join(row["aliases"]),
                    "ingredients": "|".join(row["ingredients"]),
                    "approval_numbers": "|".join(row["approval_numbers"]),
                    "example_instructions": " || ".join(row["example_instructions"]),
                    "record_count": row["record_count"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an internal drug alias dictionary from structured KG results.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {args.input_path}")

    rows, alias_map, summary = build_dict(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "drug_dictionary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "drug_alias_map.json").write_text(
        json.dumps(alias_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "drug_dictionary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "drug_dictionary.csv", rows)


if __name__ == "__main__":
    main()
