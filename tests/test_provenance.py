from pathlib import Path

from pisteflow.config import PATHS
from pisteflow.provenance import (
    portable_provenance,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def test_jsonl_is_canonical_and_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    first = write_jsonl(path, [{"b": 2, "a": 1}])
    second = write_jsonl(path, [{"a": 1, "b": 2}])
    assert first == second == sha256_file(path)
    assert read_jsonl(path) == [{"a": 1, "b": 2}]


def test_portable_provenance_strips_managed_absolute_roots(tmp_path: Path) -> None:
    repository_file = PATHS.repository / "reports" / "example.json"
    data_file = PATHS.data / "processed" / "example.parquet"

    portable = portable_provenance(
        {
            "output": str(repository_file),
            "nested": [str(data_file)],
            "source_url": "https://example.org/reference/absolute-paths",
        }
    )

    assert portable["output"] == "reports/example.json"
    assert portable["nested"] == ["data/processed/example.parquet"]
    assert portable["source_url"] == "https://example.org/reference/absolute-paths"

    output = tmp_path / "portable.jsonl"
    write_jsonl(output, [{"output": str(repository_file)}])
    assert read_jsonl(output) == [{"output": "reports/example.json"}]
