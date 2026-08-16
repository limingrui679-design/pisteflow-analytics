"""Download public PMC supplementary assets protected by the documented hash puzzle."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx


def download_asset(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    user_agent: str,
    force: bool = False,
    expected_magic: bytes | None = None,
) -> Path:
    """Download one CloudPMC asset after completing its published SHA-256 puzzle."""

    existing_asset_is_valid = (
        not force
        and destination.exists()
        and destination.stat().st_size == expected_bytes
        and (
            expected_magic is None
            or destination.read_bytes()[: len(expected_magic)] == expected_magic
        )
    )
    if existing_asset_is_valid:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=httpx.Timeout(60, read=300),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        challenge_match = re.search(rb'const POW_CHALLENGE = "([^"]+)"', response.content)
        difficulty_match = re.search(rb'const POW_DIFFICULTY = "(\d+)"', response.content)
        if challenge_match is not None or difficulty_match is not None:
            if challenge_match is None or difficulty_match is None:
                raise ValueError("CloudPMC response contains incomplete PoW metadata")
            challenge = challenge_match.group(1).decode("ascii")
            difficulty = int(difficulty_match.group(1))
            prefix = "0" * difficulty
            nonce = 0
            while not hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest().startswith(
                prefix
            ):
                nonce += 1
            client.cookies.set(
                "cloudpmc-viewer-pow",
                f"{challenge},{nonce}",
                domain="pmc.ncbi.nlm.nih.gov",
                path="/",
            )
            response = client.get(url)
            response.raise_for_status()
    if len(response.content) != expected_bytes:
        raise ValueError(
            f"CloudPMC attachment changed: expected {expected_bytes} bytes, "
            f"received {len(response.content)}"
        )
    if expected_magic is not None and not response.content.startswith(expected_magic):
        raise ValueError("CloudPMC attachment has an unexpected file signature")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination
