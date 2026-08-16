"""Audit a compact COCO body-part/strip/scoreboard fencing pre-dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_URL = "https://github.com/Ph1n-Pham/fencing-vision"
SNAPSHOT = "da7425afca48be3d3402eae108afaf3c06eacd3d"
LICENSE = "COCO export says MIT; repository and source-footage rights are undocumented"


def _source_family(filename: str) -> str:
    return filename.split("_mp4", 1)[0]


def _group_id(filename: str) -> str:
    family = _source_family(filename)
    digest = hashlib.sha256(f"pisteflow-fencing-vision:{family}".encode()).hexdigest()[:20]
    return f"fencing-vision:{digest}"


def _segmentation_points(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for polygon in value:
        if isinstance(polygon, list):
            count += len(polygon) // 2
    return count


def _image_audit(image_directory: Path, images: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional image dependency
        raise RuntimeError("Install the video extra to validate fencing images") from exc

    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(images, key=lambda item: str(item["file_name"])):
        path = image_directory / str(row["file_name"])
        if not path.is_file():
            raise ValueError(f"FencingVision annotated image is missing: {path.name}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width != int(row["width"]) or image.height != int(row["height"]):
                raise ValueError(f"FencingVision image dimensions changed: {path.name}")
        total_bytes += path.stat().st_size
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return {
        "images_verified": len(images),
        "image_bytes": total_bytes,
        "image_tree_sha256": digest.hexdigest(),
    }


def prepare(
    annotations_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    image_directory: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = payload.get("images", []) if isinstance(payload, dict) else []
    annotations = payload.get("annotations", []) if isinstance(payload, dict) else []
    categories = payload.get("categories", []) if isinstance(payload, dict) else []
    if not all(isinstance(rows, list) for rows in (images, annotations, categories)):
        raise ValueError("FencingVision COCO sections are malformed")
    image_rows = [row for row in images if isinstance(row, dict)]
    annotation_rows = [row for row in annotations if isinstance(row, dict)]
    category_rows = [row for row in categories if isinstance(row, dict)]
    if len(image_rows) != len(images) or len(annotation_rows) != len(annotations):
        raise ValueError("FencingVision COCO rows are malformed")

    image_by_id = {int(row["id"]): row for row in image_rows}
    category_by_id = {int(row["id"]): str(row["name"]) for row in category_rows}
    if len(image_by_id) != len(image_rows) or len(category_by_id) != len(category_rows):
        raise ValueError("FencingVision COCO IDs are not unique")

    output_rows: list[dict[str, Any]] = []
    for row in annotation_rows:
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        if image_id not in image_by_id or category_id not in category_by_id:
            raise ValueError("FencingVision annotation has a broken foreign key")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("FencingVision annotation has an invalid bounding box")
        filename = str(image_by_id[image_id]["file_name"])
        output_rows.append(
            {
                "annotation_id": f"fencing-vision:{int(row['id'])}",
                "image_group_id": _group_id(filename),
                "category": category_by_id[category_id],
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_width": float(bbox[2]),
                "bbox_height": float(bbox[3]),
                "area": float(row.get("area", 0.0)),
                "segmentation_points": _segmentation_points(row.get("segmentation")),
                "iscrowd": int(row.get("iscrowd", 0)),
            }
        )
    output = pd.DataFrame(output_rows)
    if output["annotation_id"].duplicated().any():
        raise ValueError("FencingVision annotation IDs are duplicated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    license_names = [
        str(row.get("name"))
        for row in payload.get("licenses", [])
        if isinstance(row, dict) and row.get("name")
    ]
    image_summary = (
        _image_audit(image_directory, image_rows)
        if image_directory is not None
        else {"images_verified": 0, "metadata_only": True}
    )
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "rights_status": LICENSE,
        "embedded_coco_license_names": license_names,
        "images": len(image_rows),
        "source_image_families": len({_source_family(str(row["file_name"])) for row in image_rows}),
        "annotations": len(output),
        "category_counts": {
            str(key): int(value)
            for key, value in output["category"].value_counts().sort_index().items()
        },
        "categories_defined": dict(sorted(category_by_id.items())),
        "annotations_sha256": sha256_file(annotations_path),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
        "image_audit": image_summary,
        "claim_boundary": (
            "A very small COCO pre-dataset with body-part, point, scoreboard, strip, and weapon "
            "annotations. Multiple frames sharing a filename family are one source group. The COCO "
            "export embeds an MIT label, but the repository has no top-level license and does not "
            "document the footage provenance, so images and row-level derivatives remain local. "
            "Eight images cannot establish a general segmentation benchmark."
        ),
    }
    write_json(report_path, summary)
    return summary
