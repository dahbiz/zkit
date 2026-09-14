"""Allow `python -m zkit <command>`."""

from zkit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
