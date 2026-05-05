import argparse
import json
from pathlib import Path

def is_empty_json_file(p: Path) -> bool:
    try:
        if p.stat().st_size == 0:
            return True
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return True
        data = json.loads(text)
        return data is None or data == {} or data == []
    except Exception:
        # 解析失败的不当作“空json”（避免误删坏文件/非json内容）
        return False
def main():
    ap = argparse.ArgumentParser(description="Delete empty JSON files under a directory.")
    default_output_dir = (Path(__file__).resolve().parent / "output")
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=str(default_output_dir),
        help="Path to output directory (will scan recursively).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only print files to delete, do not delete.")
    args = ap.parse_args()
    root = Path(args.output_dir)
    if not root.exists():
        raise SystemExit(f"Not found: {root}")
    targets = []
    for p in root.rglob("*.json"):
        if p.is_file() and is_empty_json_file(p):
            targets.append(p)
    for p in targets:
        print(("DRY-RUN " if args.dry_run else "DELETE  ") + str(p))
    if not args.dry_run:
        for p in targets:
            try:
                p.unlink()
            except Exception as e:
                print(f"FAILED  {p}  ({e})")
    print(f"Done. matched={len(targets)} dry_run={args.dry_run}")
if __name__ == "__main__":
    main()