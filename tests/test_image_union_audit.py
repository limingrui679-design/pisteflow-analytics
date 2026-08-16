import json
from pathlib import Path

from PIL import Image

from pisteflow.sources import image_union_audit


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_unions_same_upstream_identity_across_resized_files(tmp_path: Path) -> None:
    first = tmp_path / "data/raw/a.png"
    second = tmp_path / "data/raw/b.png"
    first.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "red").save(first)
    Image.new("RGB", (30, 30), "red").save(second)
    first_hash = image_union_audit.sha256_file(first)
    second_hash = image_union_audit.sha256_file(second)
    _write_jsonl(
        tmp_path / "data/manifests/a.jsonl",
        [
            {
                "id": "a",
                "hash": first_hash,
                "path": str(first),
                "commons_sha1": "a" * 40,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/manifests/b.jsonl",
        [
            {
                "id": "b",
                "hash": second_hash,
                "path": str(second),
                "commons_sha1": "a" * 40,
            }
        ],
    )

    def resolve(row: dict[str, object], _repository: Path) -> Path:
        return Path(str(row["path"]))

    sources = (
        image_union_audit.SourceSpec("a", "a.jsonl", "id", "hash", resolve),
        image_union_audit.SourceSpec("b", "b.jsonl", "id", "hash", resolve),
    )
    summary = image_union_audit.audit(
        tmp_path,
        tmp_path / "union.jsonl",
        tmp_path / "summary.json",
        sources=sources,
    )

    assert summary["rows_with_sha256"] == 2
    assert summary["unique_sha256"] == 2
    assert summary["conservative_union_groups"] == 1
    assert summary["cross_source_union_groups"] == 1


def test_flickr_identity_normalization() -> None:
    identities = image_union_audit.upstream_identities(
        {"original_landing_url": "https://www.flickr.com/photos/example/5387415067/"}
    )
    assert "flickr-photo:5387415067" in identities
