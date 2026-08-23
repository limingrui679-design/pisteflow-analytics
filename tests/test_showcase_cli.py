from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pisteflow.cli import app

runner = CliRunner()


def test_showcase_cli_combines_filters_and_emits_json() -> None:
    result = runner.invoke(
        app,
        [
            "showcase",
            "list",
            "--lens",
            "behavioral-science",
            "--status",
            "no-lift",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    cases = json.loads(result.output)
    assert [case["id"] for case in cases] == ["sabre-eye-outcomes", "hot-hand-validation"]


def test_showcase_cli_rejects_unknown_capability() -> None:
    result = runner.invoke(app, ["showcase", "list", "--lens", "magic"])

    assert result.exit_code != 0
    assert "lens must be one of" in result.output


def test_showcase_cli_audits_repository_evidence() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        ["showcase", "audit", "--evidence-root", str(repository)],
    )

    assert result.exit_code == 0, result.output
    assert '"valid": true' in result.output
    assert '"findings": []' in result.output


def test_showcase_cli_exports_markdown(tmp_path: Path) -> None:
    output = tmp_path / "catalog.md"
    result = runner.invoke(
        app,
        ["showcase", "export", "--output", str(output), "--format", "markdown"],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "Ranking trajectory negative result" in output.read_text(encoding="utf-8")
