"""Zero-data showcase command group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from pisteflow.config import PATHS
from pisteflow.showcase import (
    CapabilityLens,
    ResultStatus,
    audit_catalog,
    export_catalog,
    filter_cases,
    find_case,
    load_catalog,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Browse and audit evidence-bounded cases without downloading raw data.",
)
console = Console()


def _parse_lens(value: str | None) -> CapabilityLens | None:
    if value is None:
        return None
    try:
        return CapabilityLens(value)
    except ValueError as error:
        allowed = ", ".join(lens.value for lens in CapabilityLens)
        raise typer.BadParameter(f"lens must be one of: {allowed}") from error


def _parse_status(value: str | None) -> ResultStatus | None:
    if value is None:
        return None
    try:
        return ResultStatus(value)
    except ValueError as error:
        allowed = ", ".join(status.value for status in ResultStatus)
        raise typer.BadParameter(f"status must be one of: {allowed}") from error


@app.command("list")
def list_cases(
    lens: Annotated[str | None, typer.Option(help="Filter by capability lens.")] = None,
    status: Annotated[str | None, typer.Option(help="Filter by evidence result status.")] = None,
    weapon: Annotated[
        str | None,
        typer.Option(help="Filter by epee, foil, or sabre."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List curated cases from the package-bundled zero-data catalog."""

    normalized_weapon = weapon.casefold() if weapon is not None else None
    if normalized_weapon not in {None, "epee", "foil", "sabre"}:
        raise typer.BadParameter("weapon must be epee, foil, or sabre")
    typed_weapon = cast(Literal["epee", "foil", "sabre"] | None, normalized_weapon)
    catalog = load_catalog()
    cases = filter_cases(
        catalog,
        lens=_parse_lens(lens),
        status=_parse_status(status),
        weapon=typed_weapon,
    )
    if json_output:
        console.print_json(
            json.dumps([case.model_dump(mode="json") for case in cases], ensure_ascii=False)
        )
        return
    table = Table(title=f"PisteFlow cases · catalog {catalog.catalog_version}")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Case")
    table.add_column("Status", style="green")
    table.add_column("Headline")
    for case in cases:
        table.add_row(case.id, case.title, case.result_status.value, case.headline)
    console.print(table)
    console.print(f"[dim]{len(cases)} of {len(catalog.cases)} cases[/dim]")


@app.command("show")
def show_case(case_id: str) -> None:
    """Show one full evidence contract by stable case ID."""

    try:
        case = find_case(load_catalog(), case_id)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print_json(json.dumps(case.model_dump(mode="json"), ensure_ascii=False))


@app.command("audit")
def audit_cases(
    evidence_root: Annotated[
        Path | None,
        typer.Option(file_okay=False, help="Repository root for evidence and demo checks."),
    ] = None,
) -> None:
    """Validate catalog schema and, when available, every repository evidence path."""

    root = evidence_root
    if root is None and (PATHS.repository / "reports/model-cards").is_dir():
        root = PATHS.repository
    audit = audit_catalog(load_catalog(), evidence_root=root)
    console.print_json(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.valid:
        raise typer.Exit(code=1)


@app.command("export")
def export_cases(
    output: Annotated[Path, typer.Option(help="Destination .json or .md path")],
    format_name: Annotated[
        str,
        typer.Option("--format", help="json or markdown"),
    ] = "markdown",
) -> None:
    """Export the bundled catalog for review, documentation, or a static demo."""

    try:
        export_catalog(load_catalog(), output, format_name)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(str(output.resolve()))
