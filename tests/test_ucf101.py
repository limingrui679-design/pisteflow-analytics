import struct
from zipfile import ZIP_DEFLATED, ZipFile

from pisteflow.sources.ucf101 import parse_central_directory


def test_parse_central_directory_member(tmp_path) -> None:
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("UCF-101/Fencing/example.avi", b"RIFF0000AVI payload")
    payload = archive_path.read_bytes()
    eocd = payload.rfind(b"PK\x05\x06")
    central_bytes = struct.unpack_from("<I", payload, eocd + 12)[0]
    central_offset = struct.unpack_from("<I", payload, eocd + 16)[0]

    members = parse_central_directory(
        payload[central_offset : central_offset + central_bytes]
    )

    member = members["UCF-101/Fencing/example.avi"]
    assert member["bytes"] == 19
    assert member["compression"] == ZIP_DEFLATED
