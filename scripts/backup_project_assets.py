from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a timestamped backup archive of the local project assets.")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("backups"),
        help="Directory where backup archives will be created (default: backups).",
    )
    parser.add_argument(
        "--name",
        default="medical_ai_research_backup",
        help="Base name for the backup archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be archived without creating files.",
    )
    return parser


def collect_sources(project_root: Path) -> list[Path]:
    include = [
        project_root / "data",
        project_root / "checkpoints",
        project_root / "outputs",
        project_root / "runs",
        project_root / "report_source",
        project_root / "notebooks",
        project_root / "requirements.txt",
        project_root / "README.md",
        project_root / "CONTRIBUTING.md",
        project_root / "src",
    ]
    return [path for path in include if path.exists()]


def create_backup(project_root: Path, target_dir: Path, base_name: str, dry_run: bool) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = target_dir / f"{base_name}_{timestamp}.zip"

    sources = collect_sources(project_root)
    if not sources:
        raise RuntimeError("No backup sources were found.")

    print(f"Project root: {project_root}")
    print(f"Backup target: {target_dir}")
    print("Sources:")
    for source in sources:
        print(f" - {source.relative_to(project_root)}")

    if dry_run:
        print("Dry run only; no archive was created.")
        return archive_path

    archive_root = project_root.parent
    archive_name = archive_path.with_suffix("")
    shutil.make_archive(
        base_name=str(archive_name),
        format="zip",
        root_dir=str(archive_root),
        base_dir=str(project_root.name),
    )

    if archive_path.exists():
        archive_path.unlink()
    shutil.move(str(archive_name) + ".zip", str(archive_path))
    print(f"Created backup archive: {archive_path}")
    return archive_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    create_backup(project_root, args.target.resolve(), args.name, args.dry_run)


if __name__ == "__main__":
    main()
