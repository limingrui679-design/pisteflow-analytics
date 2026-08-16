from pathlib import Path

from pisteflow.provenance import read_jsonl, sha256_file, write_jsonl


def test_jsonl_is_canonical_and_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    first = write_jsonl(path, [{"b": 2, "a": 1}])
    second = write_jsonl(path, [{"a": 1, "b": 2}])
    assert first == second == sha256_file(path)
    assert read_jsonl(path) == [{"a": 1, "b": 2}]
