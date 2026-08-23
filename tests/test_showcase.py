from __future__ import annotations

import json
from pathlib import Path

from pisteflow.showcase import (
    CapabilityLens,
    ResultStatus,
    audit_catalog,
    export_catalog,
    filter_cases,
    find_case,
    load_catalog,
)


def test_bundled_catalog_covers_every_capability_and_keeps_negative_results() -> None:
    catalog = load_catalog()

    assert len(catalog.cases) == 15
    assert {lens for case in catalog.cases for lens in case.capabilities} == set(
        CapabilityLens
    )
    assert len(filter_cases(catalog, status=ResultStatus.NO_LIFT)) >= 4
    spatial = find_case(catalog, "france-spatial-access")
    assert CapabilityLens.SPATIAL_POLICY in spatial.capabilities
    assert spatial.result_status is ResultStatus.DESCRIPTIVE


def test_catalog_filters_compose_without_changing_case_order() -> None:
    catalog = load_catalog()
    cases = filter_cases(
        catalog,
        lens=CapabilityLens.STATISTICAL_INFERENCE,
        status=ResultStatus.LIFT,
        weapon="foil",
    )

    assert [case.id for case in cases] == ["three-weapon-elo", "foil-pose-outcomes"]


def test_repository_evidence_and_static_demo_are_synchronized() -> None:
    repository = Path(__file__).resolve().parents[1]
    audit = audit_catalog(load_catalog(), evidence_root=repository)

    assert audit.valid, [finding.model_dump() for finding in audit.findings]
    assert audit.evidence_root_checked == str(repository)


def test_repository_audit_rejects_evidence_hash_drift() -> None:
    repository = Path(__file__).resolve().parents[1]
    catalog = load_catalog()
    evidence_path = next(iter(catalog.evidence_sha256))
    hashes = {**catalog.evidence_sha256, evidence_path: "0" * 64}

    audit = audit_catalog(
        catalog.model_copy(update={"evidence_sha256": hashes}),
        evidence_root=repository,
    )

    assert not audit.valid
    assert any(
        finding.code == "evidence-hash-mismatch" and finding.path == evidence_path
        for finding in audit.findings
    )


def test_catalog_export_round_trip(tmp_path: Path) -> None:
    catalog = load_catalog()
    json_path = tmp_path / "cases.json"
    markdown_path = tmp_path / "cases.md"

    export_catalog(catalog, json_path, "json")
    export_catalog(catalog, markdown_path, "markdown")

    assert json.loads(json_path.read_text(encoding="utf-8")) == catalog.model_dump(mode="json")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# PisteFlow case studies" in markdown
    assert "Ranking trajectory negative result" in markdown
