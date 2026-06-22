from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from graphrag.api.index import build_index
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.load_config import load_config


def _replace_workflows(settings_text: str, workflows: list[str]) -> str:
    lines = settings_text.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip() == "workflows:":
            start = idx
            break
    if start is None:
        raise RuntimeError("settings.yaml missing workflows section")

    end = start + 1
    while end < len(lines):
        if lines[end].startswith("  - "):
            end += 1
            continue
        break

    workflow_lines = ["workflows:", "  - create_community_reports"]
    return "\n".join(lines[:start] + workflow_lines + lines[end:]) + "\n"


def _read_report_marker(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    first = df.iloc[0].to_dict()
    return {
        "rating_explanation": str(first.get("rating_explanation", "")),
        "summary": str(first.get("summary", ""))[:120],
    }


async def _run(project_root: Path, verbose: bool) -> int:
    settings_path = project_root / "settings.yaml"
    original_text = settings_path.read_text(encoding="utf-8")
    patched_text = _replace_workflows(original_text, ["create_community_reports"])

    output_dir = (project_root / ".." / "official_byog" / "output").resolve()
    report_path = output_dir / "community_reports.parquet"
    before = _read_report_marker(report_path)

    settings_path.write_text(patched_text, encoding="utf-8")
    try:
        config = load_config(root_dir=project_root)
        outputs = await build_index(
            config=config,
            callbacks=[NoopWorkflowCallbacks()],
            verbose=verbose,
        )
        encountered_errors = any(output.error is not None for output in outputs)
    finally:
        settings_path.write_text(original_text, encoding="utf-8")

    after = _read_report_marker(report_path)
    print(
        json.dumps(
            {
                "encountered_errors": encountered_errors,
                "before": before,
                "after": after,
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 1 if encountered_errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume official GraphRAG community_reports using the Python API and existing cache.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("drug_kg/graphrag/official_project"),
        help="Official GraphRAG project root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose official logging.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args.project_root, args.verbose)))


if __name__ == "__main__":
    main()
