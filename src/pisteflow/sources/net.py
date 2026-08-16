"""Small, resumable HTTP downloads for public research inputs."""

from __future__ import annotations

import ssl
import time
from pathlib import Path

import httpx
import truststore


def download_resumable(
    url: str,
    destination: Path,
    *,
    user_agent: str,
    expected_bytes: int | None = None,
    attempts: int = 7,
) -> Path:
    """Download to a sidecar part file and atomically publish on success."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        expected_bytes is None or destination.stat().st_size == expected_bytes
    ):
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(60, read=300),
        headers={"User-Agent": user_agent},
    ) as client:
        for attempt in range(attempts):
            existing = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            try:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else float(2**attempt)
                        )
                        time.sleep(min(delay, 45.0))
                        continue
                    response.raise_for_status()
                    append = existing > 0 and response.status_code == 206
                    with temporary.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            handle.write(chunk)
                size = temporary.stat().st_size
                if expected_bytes is not None and size != expected_bytes:
                    raise RuntimeError(
                        f"size mismatch for {destination.name}: "
                        f"expected {expected_bytes}, got {size}"
                    )
                temporary.replace(destination)
                return destination
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error
