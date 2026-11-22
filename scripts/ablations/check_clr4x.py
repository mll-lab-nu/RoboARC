#!/usr/bin/env python3
"""The E4 check of one clearance-x4 variant (CPU, standard library only; bash-runners/ablations/check_variants.sh runs
it next to the Harbor oracle / nop runs): the clearance measured in the variant's own frozen scene must be 4 times the
original, and the relaxed source and instance must be the ones every runtime_binding.json of the variant names
(scripts/ablations/cond_clearance.py check). Prints one line, `ok {json}` or `FAIL ...`; exit 0 on pass.

    python3 scripts/ablations/check_clr4x.py tasks/<base>-clr4x-<mode>"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cond_clearance  # noqa: E402

if __name__ == "__main__":
    sys.exit(cond_clearance.check_main(sys.argv[1:], 4))
