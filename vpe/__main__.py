"""Allow ``python -m vpe ...``."""

from vpe.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
