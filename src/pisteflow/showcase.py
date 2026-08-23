"""Zero-data case catalog with evidence and claim-boundary auditing."""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json


class CapabilityLens(StrEnum):
    """School-neutral capability lenses used to organize case studies."""

    DECISION_ANALYTICS = "decision-analytics"
    REPRODUCIBLE_ML = "reproducible-ml"
    STATISTICAL_INFERENCE = "statistical-inference"
    DATA_SYSTEMS_GOVERNANCE = "data-systems-governance"
    RISK_FINANCE = "risk-finance"
    BEHAVIORAL_SCIENCE = "behavioral-science"
    POPULATION_HEALTH = "population-health"
    SPATIAL_POLICY = "spatial-policy"


class ResultStatus(StrEnum):
    """Honest outcome labels that keep negative evidence visible."""

    LIFT = "lift"
    NO_LIFT = "no-lift"
    MIXED = "mixed"
    REPLICATION = "replication"
    DIAGNOSTIC = "diagnostic"
    DESCRIPTIVE = "descriptive"


class CaseStudy(BaseModel):
    """One auditable, reviewer-readable PisteFlow case study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=8)
    question: str = Field(min_length=20)
    summary: str = Field(min_length=40)
    capabilities: list[CapabilityLens] = Field(min_length=1)
    weapons: list[Literal["epee", "foil", "sabre"]]
    evidence_kind: Literal[
        "predictive-model",
        "behavioral-analysis",
        "descriptive-analysis",
        "data-system",
        "data-quality-audit",
    ]
    unit: str = Field(min_length=3)
    sample: str = Field(min_length=3)
    method: str = Field(min_length=15)
    validation: str = Field(min_length=15)
    result_status: ResultStatus
    headline: str = Field(min_length=8)
    baseline: str | None = None
    decision_use: str = Field(min_length=15)
    evidence_paths: list[str] = Field(min_length=1)
    claim_boundary: str = Field(min_length=80)

    @field_validator("capabilities", "weapons")
    @classmethod
    def reject_duplicate_tags(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("case tags must be unique")
        return value

    @field_validator("evidence_paths")
    @classmethod
    def validate_evidence_paths(cls, paths: list[str]) -> list[str]:
        if len(paths) != len(set(paths)):
            raise ValueError("evidence paths must be unique")
        for value in paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("evidence paths must be repository-relative")
        return paths

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        if (
            self.result_status in {ResultStatus.LIFT, ResultStatus.NO_LIFT, ResultStatus.MIXED}
            and self.baseline is None
        ):
            raise ValueError("comparative results require an explicit baseline")
        return self


class ShowcaseCatalog(BaseModel):
    """Versioned case-study collection distributed with the Python package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    catalog_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    evidence_sha256: dict[str, str] = Field(min_length=1)
    cases: list[CaseStudy] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog_identity(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        evidence_paths = {path for case in self.cases for path in case.evidence_paths}
        if set(self.evidence_sha256) != evidence_paths:
            raise ValueError("evidence hash keys must exactly match referenced evidence paths")
        invalid_hashes = [
            path
            for path, value in self.evidence_sha256.items()
            if re.fullmatch(r"[0-9a-f]{64}", value) is None
        ]
        if invalid_hashes:
            raise ValueError(f"invalid SHA-256 values for: {sorted(invalid_hashes)}")
        return self


class AuditFinding(BaseModel):
    """One deterministic catalog or repository evidence finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["error", "warning"]
    code: str
    message: str
    case_id: str | None = None
    path: str | None = None


class CatalogAudit(BaseModel):
    """Machine-readable showcase audit result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    schema_version: int
    catalog_version: str
    cases: int
    evidence_files: int
    capability_coverage: dict[str, int]
    result_status_counts: dict[str, int]
    evidence_root_checked: str | None
    findings: list[AuditFinding]


def load_catalog() -> ShowcaseCatalog:
    """Load the package-bundled catalog without repository or data access."""

    payload = resources.files("pisteflow.resources").joinpath("showcase_cases.json").read_text(
        encoding="utf-8"
    )
    return ShowcaseCatalog.model_validate_json(payload)


def filter_cases(
    catalog: ShowcaseCatalog,
    *,
    lens: CapabilityLens | None = None,
    status: ResultStatus | None = None,
    weapon: Literal["epee", "foil", "sabre"] | None = None,
) -> list[CaseStudy]:
    """Filter case studies while preserving their curated order."""

    return [
        case
        for case in catalog.cases
        if (lens is None or lens in case.capabilities)
        and (status is None or status == case.result_status)
        and (weapon is None or weapon in case.weapons)
    ]


def find_case(catalog: ShowcaseCatalog, case_id: str) -> CaseStudy:
    """Return a case by stable ID or raise a reviewer-readable error."""

    for case in catalog.cases:
        if case.id == case_id:
            return case
    available = ", ".join(case.id for case in catalog.cases)
    raise ValueError(f"Unknown case ID {case_id!r}. Available: {available}")


def catalog_summary(catalog: ShowcaseCatalog) -> dict[str, Any]:
    """Return compact counts used by the CLI and static showcase."""

    capabilities = Counter(lens.value for case in catalog.cases for lens in case.capabilities)
    statuses = Counter(case.result_status.value for case in catalog.cases)
    return {
        "schema_version": catalog.schema_version,
        "catalog_version": catalog.catalog_version,
        "cases": len(catalog.cases),
        "capability_coverage": dict(sorted(capabilities.items())),
        "result_status_counts": dict(sorted(statuses.items())),
    }


def bind_evidence_hashes(
    catalog: ShowcaseCatalog,
    *,
    evidence_root: Path,
) -> ShowcaseCatalog:
    """Return a catalog bound to the current bytes of every evidence file."""

    resolved_root = evidence_root.expanduser().resolve()
    hashes: dict[str, str] = {}
    for relative in sorted(catalog.evidence_sha256):
        candidate = (resolved_root / relative).resolve()
        if candidate != resolved_root and resolved_root not in candidate.parents:
            raise ValueError(f"evidence path resolves outside repository: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"evidence file does not exist: {relative}")
        hashes[relative] = sha256_file(candidate)
    return catalog.model_copy(update={"evidence_sha256": hashes})


def audit_catalog(
    catalog: ShowcaseCatalog,
    *,
    evidence_root: Path | None = None,
) -> CatalogAudit:
    """Audit coverage, evidence existence, JSON boundaries, and demo synchronization."""

    findings: list[AuditFinding] = []
    covered = {lens for case in catalog.cases for lens in case.capabilities}
    for lens in CapabilityLens:
        if lens not in covered:
            findings.append(
                AuditFinding(
                    severity="error",
                    code="missing-capability-lens",
                    message=f"No case covers {lens.value}",
                )
            )

    resolved_root: Path | None = None
    if evidence_root is not None:
        resolved_root = evidence_root.expanduser().resolve()
        for case in catalog.cases:
            for relative in case.evidence_paths:
                candidate = (resolved_root / relative).resolve()
                if candidate != resolved_root and resolved_root not in candidate.parents:
                    findings.append(
                        AuditFinding(
                            severity="error",
                            code="evidence-path-escape",
                            message="Evidence path resolves outside the repository",
                            case_id=case.id,
                            path=relative,
                        )
                    )
                    continue
                if not candidate.is_file():
                    findings.append(
                        AuditFinding(
                            severity="error",
                            code="missing-evidence",
                            message="Referenced evidence file does not exist",
                            case_id=case.id,
                            path=relative,
                        )
                    )
                    continue
                actual_sha256 = sha256_file(candidate)
                expected_sha256 = catalog.evidence_sha256[relative]
                if actual_sha256 != expected_sha256:
                    findings.append(
                        AuditFinding(
                            severity="error",
                            code="evidence-hash-mismatch",
                            message=(
                                f"Evidence SHA-256 changed: expected {expected_sha256}, "
                                f"observed {actual_sha256}"
                            ),
                            case_id=case.id,
                            path=relative,
                        )
                    )
                if relative.startswith("reports/model-cards/") and candidate.suffix == ".json":
                    try:
                        evidence = json.loads(candidate.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as error:
                        findings.append(
                            AuditFinding(
                                severity="error",
                                code="invalid-evidence-json",
                                message=str(error),
                                case_id=case.id,
                                path=relative,
                            )
                        )
                        continue
                    if not isinstance(evidence, dict) or not evidence.get("claim_boundary"):
                        findings.append(
                            AuditFinding(
                                severity="error",
                                code="missing-evidence-boundary",
                                message="Analysis evidence lacks a claim boundary",
                                case_id=case.id,
                                path=relative,
                            )
                        )

        demo_path = resolved_root / "docs/showcase/cases.json"
        if not demo_path.is_file():
            findings.append(
                AuditFinding(
                    severity="error",
                    code="missing-demo-catalog",
                    message="Static showcase catalog is missing",
                    path="docs/showcase/cases.json",
                )
            )
        else:
            try:
                demo_payload = json.loads(demo_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                findings.append(
                    AuditFinding(
                        severity="error",
                        code="invalid-demo-catalog",
                        message=str(error),
                        path="docs/showcase/cases.json",
                    )
                )
            else:
                if demo_payload != catalog.model_dump(mode="json"):
                    findings.append(
                        AuditFinding(
                            severity="error",
                            code="demo-catalog-drift",
                            message="Static demo and package catalogs differ",
                            path="docs/showcase/cases.json",
                        )
                    )

    summary = catalog_summary(catalog)
    return CatalogAudit(
        valid=not any(finding.severity == "error" for finding in findings),
        schema_version=catalog.schema_version,
        catalog_version=catalog.catalog_version,
        cases=len(catalog.cases),
        evidence_files=len(catalog.evidence_sha256),
        capability_coverage=summary["capability_coverage"],
        result_status_counts=summary["result_status_counts"],
        evidence_root_checked=str(resolved_root) if resolved_root is not None else None,
        findings=findings,
    )


def render_markdown(catalog: ShowcaseCatalog) -> str:
    """Render a concise catalog that stays synchronized with machine-readable cases."""

    lines = [
        "# PisteFlow case studies",
        "",
        (
            f"Catalog {catalog.catalog_version} contains {len(catalog.cases)} bounded cases. "
            "Negative and mixed results remain first-class evidence."
        ),
        "",
        "| Case | Capability lenses | Evidence status | Headline |",
        "|---|---|---|---|",
    ]
    for case in catalog.cases:
        title = case.title.replace("|", "\\|")
        lenses = ", ".join(lens.value for lens in case.capabilities)
        headline = case.headline.replace("|", "\\|")
        lines.append(
            f"| [{title}](#{case.id}) | {lenses} | {case.result_status.value} | "
            f"{headline} |"
        )
    for case in catalog.cases:
        lines.extend(
            [
                "",
                f"## {case.title}",
                "",
                f"<a id=\"{case.id}\"></a>",
                "",
                case.summary,
                "",
                f"- **Question:** {case.question}",
                f"- **Unit / sample:** {case.unit}; {case.sample}",
                f"- **Method:** {case.method}",
                f"- **Validation:** {case.validation}",
                f"- **Result:** {case.headline}",
                f"- **Baseline:** {case.baseline or 'Not applicable; descriptive case'}",
                f"- **Decision use:** {case.decision_use}",
                f"- **Boundary:** {case.claim_boundary}",
                "- **Evidence:** " + ", ".join(f"`{path}`" for path in case.evidence_paths),
            ]
        )
    return "\n".join(lines) + "\n"


def export_catalog(catalog: ShowcaseCatalog, output_path: Path, format_name: str) -> None:
    """Write a deterministic JSON or Markdown catalog export."""

    normalized = format_name.casefold()
    if normalized == "json":
        write_json(output_path, catalog.model_dump(mode="json"))
        return
    if normalized == "markdown":
        atomic_write_bytes(output_path, render_markdown(catalog).encode("utf-8"))
        return
    raise ValueError("format must be 'json' or 'markdown'")
