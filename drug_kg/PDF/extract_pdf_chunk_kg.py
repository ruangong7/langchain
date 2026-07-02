"""Run KG triple extraction over cleaned PDF chunk JSON files."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any


PDF_DIR = Path(__file__).resolve().parent
DRUG_KG_DIR = PDF_DIR.parent
ROOT_DIR = DRUG_KG_DIR.parent
UNSTRUCTURED_DIR = DRUG_KG_DIR / "unstructured"

if str(UNSTRUCTURED_DIR) not in sys.path:
    sys.path.insert(0, str(UNSTRUCTURED_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extract_unstructured_chunk_kg import (  # noqa: E402
    DEEPSEEK_MODEL_NAME,
    MODEL_NAME,
    build_system_prompt,
    load_modeling_rules,
    process_file,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SIGNAL_TERMS = (
    "相互作用",
    "联用",
    "合用",
    "禁忌",
    "慎用",
    "血药浓度",
    "不良反应",
    "CYP",
    "P-糖蛋白",
    "INR",
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    return [record for record in records if isinstance(record, dict)]


def _has_signal(path: Path, *, min_chars: int, require_signal: bool) -> bool:
    try:
        text = "\n".join(str(record.get("text", "") or "") for record in _read_records(path)).strip()
    except (OSError, json.JSONDecodeError):
        logger.warning("Skip invalid chunk JSON: %s", path)
        return False

    if min_chars > 0 and len(text) < min_chars:
        return False
    if require_signal and not any(term in text for term in SIGNAL_TERMS):
        return False
    return bool(text)


def collect_input_paths(
    *,
    input_dir: Path,
    pattern: str,
    recursive: bool,
    input_file: Path | None,
    min_chars: int,
    require_signal: bool,
) -> list[Path]:
    if input_file is not None:
        paths = [input_file]
    else:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        paths = sorted(path for path in iterator if path.is_file())

    return [
        path
        for path in paths
        if _has_signal(path, min_chars=min_chars, require_signal=require_signal)
    ]


def output_path_for(*, input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        rel = input_path.relative_to(input_dir)
    except ValueError:
        rel = Path(input_path.parent.name) / input_path.name
    return output_dir / rel


def remove_previous_outputs(path: Path) -> None:
    for candidate in (
        path,
        path.with_suffix(".success.jsonl"),
        path.with_suffix(".failed.jsonl"),
    ):
        if candidate.exists():
            candidate.unlink()


def write_aggregate_result(*, output_dir: Path, result_path: Path) -> None:
    aggregated: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("chunk_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skip invalid output JSON: %s", path)
            continue
        if isinstance(payload, list):
            aggregated.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            aggregated.append(payload)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote aggregate PDF KG result: %s records=%s", result_path, len(aggregated))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM triple extraction for cleaned PDF chunk JSON files")
    parser.add_argument("--input-dir", type=Path, default=PDF_DIR / "chunk_json")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PDF_DIR / "pdf_kg_output")
    parser.add_argument("--result-path", type=Path, default=PDF_DIR / "pdf_kg_result.json")
    parser.add_argument("--rules-file", type=Path, default=ROOT_DIR / "knowledge_graph_model.txt")
    parser.add_argument("--pattern", default="chunk_*.json")
    parser.add_argument("--model", default=DEEPSEEK_MODEL_NAME or MODEL_NAME)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N chunk files when > 0")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count at chunk-file level")
    parser.add_argument("--min-chars", type=int, default=0, help="Skip chunk files whose text is shorter than this")
    parser.add_argument("--require-signal", action="store_true", help="Only keep chunks with obvious drug KG signals")
    parser.add_argument("--no-recursive", action="store_true", help="Only read one directory level")
    parser.add_argument("--force", action="store_true", help="Remove existing per-chunk outputs before processing")
    parser.add_argument("--dry-run", action="store_true", help="Only list how many chunk files would be processed")
    args = parser.parse_args()

    if not args.rules_file.is_file():
        raise FileNotFoundError(f"Rules file not found: {args.rules_file}")

    input_paths = collect_input_paths(
        input_dir=args.input_dir,
        input_file=args.input_file,
        pattern=args.pattern,
        recursive=not args.no_recursive,
        min_chars=max(0, args.min_chars),
        require_signal=args.require_signal,
    )
    if args.limit > 0:
        input_paths = input_paths[: args.limit]

    logger.info("Found %s PDF chunk files to process", len(input_paths))
    if not input_paths:
        return
    if args.dry_run:
        for path in input_paths[:10]:
            logger.info("Dry-run sample: %s", path)
        return

    modeling_rules = load_modeling_rules(args.rules_file)
    system_prompt = build_system_prompt(modeling_rules)

    def _run_file(input_path: Path) -> tuple[Path, Path]:
        output_path = output_path_for(
            input_path=input_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        if args.force:
            remove_previous_outputs(output_path)
        process_file(
            system_prompt=system_prompt,
            model=args.model,
            input_path=input_path,
            output_path=output_path,
            limit=0,
            max_workers=1,
        )
        return input_path, output_path

    if max(1, args.max_workers) <= 1:
        for input_path in input_paths:
            try:
                _, output_path = _run_file(input_path)
                logger.info("Processed %s -> %s", input_path, output_path)
            except Exception as exc:
                logger.exception("File failed: %s err=%s", input_path, exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {executor.submit(_run_file, path): path for path in input_paths}
            for future in concurrent.futures.as_completed(future_map):
                input_path = future_map[future]
                try:
                    _, output_path = future.result()
                    logger.info("Processed %s -> %s", input_path, output_path)
                except Exception as exc:
                    logger.exception("File failed: %s err=%s", input_path, exc)

    write_aggregate_result(output_dir=args.output_dir, result_path=args.result_path)


if __name__ == "__main__":
    main()
