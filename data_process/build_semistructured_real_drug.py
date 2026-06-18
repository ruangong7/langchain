from __future__ import annotations

import argparse
import json
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TARGET_FIELDS = ("interaction", "precaution", "indication")
XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_semistructured_records(
    workbook_path: Path,
    sheet_name: str | None = None,
) -> list[dict[str, Any]]:
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
        if not sheet_rows:
            return []

        rows = [read_row(row, shared_strings) for row in sheet_rows]

    headers = [normalize_text(v) for v in rows[0]]
    header_index = {name: idx for idx, name in enumerate(headers)}

    required = {"drug_name", *TARGET_FIELDS}
    missing = sorted(required - set(header_index))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows[1:], start=2):
        drug_name = normalize_text(row[header_index["drug_name"]])
        if not drug_name:
            continue

        for field in TARGET_FIELDS:
            text = normalize_text(row[header_index[field]])
            if not text:
                continue

            records.append(
                {
                    "record_id": f"{selected_name}:{row_idx}:{field}",
                    "source": workbook_path.name,
                    "sheet": selected_name,
                    "row_index": row_idx,
                    "field": field,
                    "main_entity": {
                        "type": "Drug",
                        "name": drug_name,
                    },
                    "text": text,
                }
            )

    return records


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


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semi-structured KG extraction records from real_drug.xlsx")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/real_drug.xlsx"),
        help="Path to the source Excel file",
    )
    parser.add_argument(
        "--sheet",
        default="",
        help="Optional sheet name. Defaults to the first sheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_drug_semistructured.jsonl"),
        help="Path to the output JSONL file",
    )
    args = parser.parse_args()

    records = iter_semistructured_records(args.input, args.sheet or None)
    write_jsonl(records, args.output)
    logger.info("Wrote %d semi-structured records to %s", len(records), args.output)


if __name__ == "__main__":
    main()
