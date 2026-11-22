"""python -m roboarc <command>: the `roboarc` command line (roboarc/cli/), for where the console script is not on PATH,
e.g. inside an image with the repository on PYTHONPATH. `python -m roboarc --help` lists the commands."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
