"""Runtime paths that keep large or restricted media outside version control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Resolved project paths.

    ``PISTEFLOW_DATA_DIR`` may point to a separate disk. When absent, local data
    is placed below the repository's ignored ``data`` directory.
    """

    repository: Path
    data: Path
    raw: Path
    cache: Path
    processed: Path
    models: Path
    artifacts: Path

    @classmethod
    def discover(cls) -> Paths:
        repository = Path(__file__).resolve().parents[2]
        data_override = os.environ.get("PISTEFLOW_DATA_DIR")
        data = Path(data_override).expanduser().resolve() if data_override else repository / "data"
        return cls(
            repository=repository,
            data=data,
            raw=data / "raw",
            cache=data / "cache",
            processed=data / "processed",
            models=data / "models",
            artifacts=repository / "artifacts",
        )

    def create_local_directories(self) -> None:
        for path in (self.raw, self.cache, self.processed, self.models, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths.discover()
