from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import yaml

from pisteflow import __version__
from pisteflow.showcase import load_catalog

REPOSITORY = Path(__file__).resolve().parents[1]


def test_readme_local_links_resolve() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    unresolved: list[str] = []
    for target in targets:
        path_text = unquote(target.split("#", maxsplit=1)[0])
        if not path_text or "://" in path_text or path_text.startswith("mailto:"):
            continue
        if not (REPOSITORY / path_text).exists():
            unresolved.append(target)
    assert unresolved == []


def test_public_versions_and_generated_catalogs_are_synchronized() -> None:
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((REPOSITORY / "CITATION.cff").read_text(encoding="utf-8"))
    catalog = load_catalog()
    demo_catalog = json.loads(
        (REPOSITORY / "docs/showcase/cases.json").read_text(encoding="utf-8")
    )
    html = (REPOSITORY / "docs/showcase/index.html").read_text(encoding="utf-8")

    assert project["project"]["version"] == __version__ == catalog.catalog_version
    assert citation["version"] == __version__
    assert f"catalog {catalog.catalog_version}" in html
    assert demo_catalog == catalog.model_dump(mode="json")


def test_readme_spatial_metrics_match_machine_readable_evidence() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    card = json.loads(
        (REPOSITORY / "reports/model-cards/france-spatial-access.json").read_text(
            encoding="utf-8"
        )
    )
    association = card["associations"][
        "membership_rate_vs_active_establishment_density"
    ]
    interval = association["percentile_95_interval"]

    assert f"{card['departments']} in-scope departments" in readme
    assert f"{card['national_context']['licence_memberships']:,} memberships" in readme
    assert f"\u03c1 {association['spearman_rho']:.3f}" in readme
    assert f"{interval[0]:.3f}\u2013{interval[1]:.3f}" in readme


def test_static_demo_has_no_external_runtime_dependencies_or_tracking() -> None:
    html = (REPOSITORY / "docs/showcase/index.html").read_text(encoding="utf-8")
    script = (REPOSITORY / "docs/showcase/app.js").read_text(encoding="utf-8")

    assert not re.search(r'<script[^>]+src=["\']https?://', html)
    assert not re.search(r'<link[^>]+href=["\']https?://[^>]+stylesheet', html)
    assert "google-analytics" not in html.casefold()
    assert "localStorage" not in script
    assert "document.cookie" not in script


def test_static_demo_local_assets_stay_inside_pages_artifact() -> None:
    showcase = REPOSITORY / "docs/showcase"
    html = (showcase / "index.html").read_text(encoding="utf-8")
    local_targets = re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html)
    escaped: list[str] = []
    missing: list[str] = []

    for target in local_targets:
        path_text = unquote(target.split("#", maxsplit=1)[0])
        if not path_text or "://" in path_text:
            continue
        candidate = (showcase / path_text).resolve()
        if candidate != showcase and showcase not in candidate.parents:
            escaped.append(target)
        elif not candidate.exists():
            missing.append(target)

    assert escaped == []
    assert missing == []


def test_public_evidence_contains_no_machine_specific_home_paths() -> None:
    public_roots = [
        REPOSITORY / "data/manifests",
        REPOSITORY / "data/registry",
        REPOSITORY / "reports",
        REPOSITORY / "docs",
    ]
    private_home = re.compile(r"(?:/Users/|/home/)[^/\s]+/")
    offenders: list[str] = []

    for root in public_roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if private_home.search(line):
                            offenders.append(f"{path.relative_to(REPOSITORY)}:{line_number}")
                            break
            except UnicodeDecodeError:
                continue

    assert offenders == []
