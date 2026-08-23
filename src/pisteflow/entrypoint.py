"""Fast console dispatcher for lightweight and full command groups."""

from __future__ import annotations

import sys


def main() -> None:
    """Load the zero-data showcase without importing every optional source adapter."""

    arguments = sys.argv[1:]
    if arguments and arguments[0] == "showcase":
        from pisteflow.commands.showcase import app as showcase_app

        showcase_app(prog_name="pisteflow showcase", args=arguments[1:])
        return

    from pisteflow.cli import app

    app(prog_name="pisteflow", args=arguments)
