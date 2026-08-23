"""Synchronize evidence hashes and reviewer-facing case catalog exports."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from pisteflow.showcase import (
    ShowcaseCatalog,
    bind_evidence_hashes,
    export_catalog,
)


def _paths(repository: Path) -> tuple[Path, Path, Path]:
    return (
        repository / "src/pisteflow/resources/showcase_cases.json",
        repository / "docs/showcase/cases.json",
        repository / "docs/CASE_STUDIES.md",
    )


def _expected_catalog(repository: Path) -> ShowcaseCatalog:
    source, _, _ = _paths(repository)
    catalog = ShowcaseCatalog.model_validate_json(source.read_text(encoding="utf-8"))
    return bind_evidence_hashes(catalog, evidence_root=repository)


def synchronize(repository: Path) -> None:
    """Write all generated catalog surfaces from one evidence-bound source."""

    source, demo, markdown = _paths(repository)
    catalog = _expected_catalog(repository)
    export_catalog(catalog, source, "json")
    export_catalog(catalog, demo, "json")
    export_catalog(catalog, markdown, "markdown")


def check(repository: Path) -> list[str]:
    """Return drifted generated paths without modifying the repository."""

    source, demo, markdown = _paths(repository)
    catalog = _expected_catalog(repository)
    drifted: list[str] = []

    # Export through temporary sibling-independent files to exercise the same
    # canonical writers without changing repository state.
    with tempfile.TemporaryDirectory(prefix="pisteflow-showcase-") as directory:
        temporary = Path(directory)
        expected_source = temporary / "cases.json"
        expected_markdown = temporary / "cases.md"
        export_catalog(catalog, expected_source, "json")
        export_catalog(catalog, expected_markdown, "markdown")
        expected_json_bytes = expected_source.read_bytes()
        expected_markdown_bytes = expected_markdown.read_bytes()

    for path in (source, demo):
        if not path.is_file() or path.read_bytes() != expected_json_bytes:
            drifted.append(path.relative_to(repository).as_posix())
    if not markdown.is_file() or markdown.read_bytes() != expected_markdown_bytes:
        drifted.append(markdown.relative_to(repository).as_posix())
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report generated-file drift without writing",
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if arguments.check:
        drifted = check(repository)
        if drifted:
            print("showcase drift: " + ", ".join(drifted), file=sys.stderr)
            return 1
        print("showcase catalog is synchronized")
        return 0
    synchronize(repository)
    print("synchronized showcase catalog, demo, and case-study guide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
