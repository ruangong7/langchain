from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}
TARGET_FIELDS = ("interaction", "precaution", "indication")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_picture_urls(raw: str) -> list[str]:
    text = normalize_text(raw)
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        value = None
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    return [text]


def read_row(row: ET.Element, shared_strings: list[str]) -> list[str]:
    indexed: dict[int, str] = {}
    max_col = -1
    for cell in row.findall("a:c", XML_NS):
        ref = cell.attrib.get("r", "")
        col_idx = column_ref_to_index(ref)
        indexed[col_idx] = read_cell(cell, shared_strings)
        if col_idx > max_col:
            max_col = col_idx

    if max_col < 0:
        return []
    return [indexed.get(i, "") for i in range(max_col + 1)]


def column_ref_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def read_cell(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", XML_NS)
    if value_node is None:
        inline = cell.find("a:is", XML_NS)
        if inline is not None:
            return "".join(t.text or "" for t in inline.findall(".//a:t", XML_NS))
        return ""

    raw = value_node.text or ""
    if cell_type == "s":
        idx = int(raw)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else raw
    return raw


def load_rows(workbook_path: Path, sheet_name: str | None = None) -> tuple[str, list[dict[str, str]]]:
    with zipfile.ZipFile(workbook_path) as zf:
        workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_xml.findall("p:Relationship", XML_NS)
        }

        sheets: list[tuple[str, str]] = []
        for sheet in workbook_xml.find("a:sheets", XML_NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            sheets.append((sheet.attrib["name"], "xl/" + rel_map[rid]))

        if not sheets:
            raise ValueError("No worksheets found in workbook")

        selected_name, selected_path = sheets[0]
        if sheet_name:
            matched = [item for item in sheets if item[0] == sheet_name]
            if not matched:
                raise ValueError(f"Sheet not found: {sheet_name}")
            selected_name, selected_path = matched[0]

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("a:si", XML_NS):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", XML_NS)))

        sheet_root = ET.fromstring(zf.read(selected_path))
        sheet_rows = sheet_root.findall(".//a:sheetData/a:row", XML_NS)
        rows = [read_row(row, shared_strings) for row in sheet_rows]

    if not rows:
        return selected_name, []

    headers = [normalize_text(value) for value in rows[0]]
    records: list[dict[str, str]] = []
    for row_idx, row in enumerate(rows[1:], start=2):
        record = {
            header: normalize_text(row[col_idx] if col_idx < len(row) else "")
            for col_idx, header in enumerate(headers)
            if header
        }
        record["_sheet"] = selected_name
        record["_row_index"] = str(row_idx)
        records.append(record)
    return selected_name, records


def build_cleaned_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        drug_name = normalize_text(row.get("drug_name"))
        if not drug_name:
            continue
        cleaned.append(
            {
                "drug_name": drug_name,
                "approval_number": normalize_text(row.get("approval_number")),
                "instruction": normalize_text(row.get("instruction")),
                "ingredient": normalize_text(row.get("ingredient")),
                "picture": parse_picture_urls(row.get("picture", "")),
                "interaction": normalize_text(row.get("interaction")),
                "precaution": normalize_text(row.get("precaution")),
                "indication": normalize_text(row.get("indication")),
                "sheet": row.get("_sheet", ""),
                "row_index": int(row.get("_row_index", "0") or 0),
            }
        )
    return cleaned


def build_extraction_records(cleaned_rows: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in cleaned_rows:
        for field in TARGET_FIELDS:
            text = normalize_text(row.get(field))
            if not text:
                continue
            records.append(
                {
                    "record_id": f"{row['sheet']}:{row['row_index']}:{field}",
                    "record_index": len(records),
                    "chunk_index": f"{row['row_index']}:{field}",
                    "source": source_name,
                    "sheet": row["sheet"],
                    "row_index": row["row_index"],
                    "field": field,
                    "text": text,
                    "main_entity": {
                        "type": "Drug",
                        "name": row["drug_name"],
                        "properties": {
                            "approval_number": row["approval_number"],
                            "instruction": row["instruction"],
                            "ingredient": row["ingredient"],
                            "picture": row["picture"],
                        },
                    },
                }
            )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(value: Any) -> str:
    return normalize_text(value)


def _entity_name(entity: dict[str, Any]) -> str:
    return _norm(entity.get("canonical_name") or entity.get("canonical") or entity.get("name"))


def build_triples_from_extractions(
    extracted_dir: Path,
    cleaned_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    drug_rows = {f"{row['sheet']}:{row['row_index']}": row for row in cleaned_rows}
    triple_rows: list[dict[str, Any]] = []
    node_map: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    edge_map: OrderedDict[tuple[str, str, str, str], dict[str, Any]] = OrderedDict()

    if not extracted_dir.exists():
        logger.warning("Extraction output dir not found: %s", extracted_dir)
        return [], [], []

    for path in sorted(extracted_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            entities = record.get("entities") or []
            relations = record.get("relations") or []
            if not isinstance(entities, list) or not isinstance(relations, list):
                continue

            local_entities = {
                _norm(entity.get("id")): entity
                for entity in entities
                if isinstance(entity, dict) and _norm(entity.get("id"))
            }

            chunk_index = _norm(record.get("chunk_index"))
            row_key = ""
            if ":" in chunk_index:
                row_key = chunk_index.split(":", 1)[0]
            field = ""
            if ":" in chunk_index:
                field = chunk_index.split(":", 1)[1]
            row_meta = None
            if row_key and record.get("sheet"):
                row_meta = drug_rows.get(f"{record.get('sheet')}:{row_key}")
            if row_meta is None and record.get("record_id"):
                parts = str(record["record_id"]).split(":")
                if len(parts) >= 3:
                    row_meta = drug_rows.get(f"{parts[0]}:{parts[1]}")
                    field = parts[2]

            main_drug = None
            if row_meta:
                main_drug = {
                    "type": "Drug",
                    "name": row_meta["drug_name"],
                    "properties": {
                        "approval_number": row_meta["approval_number"],
                        "instruction": row_meta["instruction"],
                        "ingredient": row_meta["ingredient"],
                        "picture": row_meta["picture"],
                    },
                }
                node_map.setdefault(
                    ("Drug", row_meta["drug_name"]),
                    {
                        "type": "Drug",
                        "name": row_meta["drug_name"],
                        "properties": main_drug["properties"],
                    },
                )

            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                entity_type = _norm(entity.get("type") or "Entity")
                entity_name = _entity_name(entity)
                if not entity_name:
                    continue
                node_map.setdefault(
                    (entity_type, entity_name),
                    {
                        "type": entity_type,
                        "name": entity_name,
                        "properties": entity.get("properties") or {},
                    },
                )

            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                source_entity = local_entities.get(_norm(relation.get("source_id")))
                target_entity = local_entities.get(_norm(relation.get("target_id")))
                if not isinstance(source_entity, dict) or not isinstance(target_entity, dict):
                    continue

                source_type = _norm(source_entity.get("type") or "Entity")
                source_name = _entity_name(source_entity)
                target_type = _norm(target_entity.get("type") or "Entity")
                target_name = _entity_name(target_entity)
                predicate = _norm(relation.get("type") or field.upper() or "RELATED_TO")
                if not source_name or not target_name or not predicate:
                    continue

                evidence = {
                    "source": record.get("source", ""),
                    "sheet": record.get("sheet", ""),
                    "row_index": record.get("row_index", ""),
                    "field": field,
                    "chunk_index": record.get("chunk_index", ""),
                    "text": record.get("text", ""),
                }
                triple_rows.append(
                    {
                        "head_type": source_type,
                        "head": source_name,
                        "relation": predicate,
                        "tail_type": target_type,
                        "tail": target_name,
                        "field": field,
                        "source": record.get("source", ""),
                        "sheet": record.get("sheet", ""),
                        "row_index": record.get("row_index", ""),
                        "record_id": record.get("record_id", ""),
                        "properties": relation.get("properties") or {},
                        "evidence": evidence,
                        "drug_name": row_meta["drug_name"] if row_meta else "",
                    }
                )
                edge_map.setdefault(
                    (source_type, source_name, predicate, target_type + "::" + target_name),
                    {
                        "head_type": source_type,
                        "head": source_name,
                        "relation": predicate,
                        "tail_type": target_type,
                        "tail": target_name,
                        "properties": relation.get("properties") or {},
                    },
                )

                if main_drug and row_meta and source_name != row_meta["drug_name"] and target_name != row_meta["drug_name"]:
                    triple_rows.append(
                        {
                            "head_type": "Drug",
                            "head": row_meta["drug_name"],
                            "relation": predicate,
                            "tail_type": target_type,
                            "tail": target_name,
                            "field": field,
                            "source": record.get("source", ""),
                            "sheet": record.get("sheet", ""),
                            "row_index": record.get("row_index", ""),
                            "record_id": record.get("record_id", ""),
                            "properties": relation.get("properties") or {},
                            "evidence": evidence,
                            "drug_name": row_meta["drug_name"],
                        }
                    )
                    edge_map.setdefault(
                        ("Drug", row_meta["drug_name"], predicate, target_type + "::" + target_name),
                        {
                            "head_type": "Drug",
                            "head": row_meta["drug_name"],
                            "relation": predicate,
                            "tail_type": target_type,
                            "tail": target_name,
                            "properties": relation.get("properties") or {},
                        },
                    )

    node_rows = [
        {
            "type": item["type"],
            "name": item["name"],
            "properties": json.dumps(item.get("properties") or {}, ensure_ascii=False, separators=(",", ":")),
        }
        for item in node_map.values()
    ]
    edge_rows = [
        {
            "head_type": item["head_type"],
            "head": item["head"],
            "relation": item["relation"],
            "tail_type": item["tail_type"],
            "tail": item["tail"],
            "properties": json.dumps(item.get("properties") or {}, ensure_ascii=False, separators=(",", ":")),
        }
        for item in edge_map.values()
    ]
    return triple_rows, node_rows, edge_rows


def write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare structured real_drug records for KG extraction and export triples")
    parser.add_argument("--input", type=Path, default=Path("data/real_drug.xlsx"), help="Path to the Excel file")
    parser.add_argument("--sheet", default="", help="Optional sheet name. Defaults to the first sheet.")
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=Path("data/real_drug_cleaned.jsonl"),
        help="Path to cleaned structured JSONL",
    )
    parser.add_argument(
        "--extract-input",
        type=Path,
        default=Path("data/real_drug_structured_extract_input.json"),
        help="Path to extraction input JSON for the three target fields",
    )
    parser.add_argument(
        "--extract-output-dir",
        type=Path,
        default=Path("drug_kg/structured/output"),
        help="Directory containing entity/relation extraction outputs",
    )
    parser.add_argument(
        "--triples-output",
        type=Path,
        default=Path("drug_kg/structured/triples.jsonl"),
        help="Path to output extracted triples JSONL",
    )
    parser.add_argument(
        "--nodes-output",
        type=Path,
        default=Path("drug_kg/structured/nodes.csv"),
        help="Path to output graph nodes CSV",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=Path("drug_kg/structured/edges.csv"),
        help="Path to output graph edges CSV",
    )
    args = parser.parse_args()

    sheet_name, rows = load_rows(args.input, args.sheet or None)
    cleaned_rows = build_cleaned_rows(rows)
    write_jsonl(cleaned_rows, args.clean_output)

    extraction_records = build_extraction_records(cleaned_rows, args.input.name)
    write_json(extraction_records, args.extract_input)

    triples, nodes, edges = build_triples_from_extractions(args.extract_output_dir, cleaned_rows)
    write_jsonl(triples, args.triples_output)
    write_csv(args.nodes_output, ["type", "name", "properties"], nodes)
    write_csv(args.edges_output, ["head_type", "head", "relation", "tail_type", "tail", "properties"], edges)

    logger.info("Processed sheet %s from %s", sheet_name, args.input)
    logger.info("Wrote %d cleaned rows to %s", len(cleaned_rows), args.clean_output)
    logger.info("Wrote %d extraction input records to %s", len(extraction_records), args.extract_input)
    logger.info("Wrote %d triples to %s", len(triples), args.triples_output)
    logger.info("Wrote %d nodes to %s", len(nodes), args.nodes_output)
    logger.info("Wrote %d edges to %s", len(edges), args.edges_output)


if __name__ == "__main__":
    main()
