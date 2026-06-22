from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from graphrag.api.index import build_index
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.load_config import load_config

import asyncio

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
        line = lines[end]
        if line.startswith("  - "):
            end += 1
            continue
        break

    workflow_lines = ["workflows:"] + [f"  - {name}" for name in workflows]
    return "\n".join(lines[:start] + workflow_lines + lines[end:]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official GraphRAG BYOG indexing in phases.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("drug_kg/graphrag/official_project"),
        help="Official GraphRAG project root.",
    )
    parser.add_argument(
        "--phase",
        choices=["communities", "reports", "embeddings", "full"],
        default="full",
        help="Which official workflow phase to run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose mode to graphrag index.",
    )
    parser.add_argument(
        "--use-cli",
        action="store_true",
        help="Use the official CLI instead of the Python API runner.",
    )
    args = parser.parse_args()

    workflow_map = {
        "communities": ["create_communities"],
        "reports": ["create_community_reports"],
        "embeddings": ["generate_text_embeddings"],
        "full": ["create_communities", "create_community_reports", "generate_text_embeddings"],
    }

    settings_path = args.project_root / "settings.yaml"
    backup_path = args.project_root / "settings.original.yaml"
    original_text = settings_path.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")
    patched_text = _replace_workflows(original_text, workflow_map[args.phase])
    settings_path.write_text(patched_text, encoding="utf-8")

    try:
        if args.use_cli:
            cmd = ["graphrag", "index", "-r", str(args.project_root), "--skip-validation"]
            if args.verbose:
                cmd.append("--verbose")
            completed = subprocess.run(cmd, check=False)
            sys.exit(completed.returncode)

        config = load_config(root_dir=args.project_root)
        outputs = asyncio.run(
            build_index(
                config=config,
                callbacks=[NoopWorkflowCallbacks()],
                verbose=args.verbose,
            )
        )
        encountered_errors = any(output.error is not None for output in outputs)
        sys.exit(1 if encountered_errors else 0)
    finally:
        settings_path.write_text(original_text, encoding="utf-8")
        if backup_path.exists():
            backup_path.unlink()


if __name__ == "__main__":
    main()
