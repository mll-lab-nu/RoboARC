"""CPU confirmation of GPU stress candidates for a redesign tier: the stress harness's replay.run (production Session,
privileged mode, the stress observer) on each candidate stream, with the redesign overlay registered and the Robotiq
springref (BUZZ_SPRINGREF, default 2.62) set right after each Session build. One result JSON per candidate, in the
format verdict.py reads. usage: confirm.py <instance.json> <cand dir> <out dir> [--workers N] [--hold 40]"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRESS = Path(os.environ.get("BZ_STRESS", HERE.parent / "stress"))


def _one(args):
    inst, cand, out, hold = args
    sys.path.insert(0, str(STRESS))
    import _env  # noqa: F401
    import rdreg
    rdreg.register()
    import rcb_topogym
    from topogym.planning.buzz_wire_3d.redesign.io import set_springref
    sr = float(os.environ.get("BUZZ_SPRINGREF", "2.62"))
    if not getattr(rcb_topogym.Session, "_bz_sr", False):
        init = rcb_topogym.Session.__init__

        def patched(self, *a, **k):
            init(self, *a, **k)
            set_springref(self.env._world.model, sr)
        rcb_topogym.Session.__init__ = patched
        rcb_topogym.Session._bz_sr = True
    import replay
    try:
        rec = replay.run(inst, cand, hold=hold, case_id=Path(cand).stem)
        rec["robotiq_springref"] = sr
    except Exception as exc:  # recorded, never hidden
        rec = dict(id=Path(cand).stem, error=f"{type(exc).__name__}: {exc}")
    Path(out).write_text(json.dumps(rec, default=str))
    return Path(cand).stem, rec.get("verdict") or rec.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("cands"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--hold", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cands = [Path(a.cands)] if a.cands.endswith(".npz") else sorted(Path(a.cands).glob("*.npz"))
    jobs = [(a.instance, str(c), str(out / (c.stem + ".json")), a.hold) for c in cands
            if not (out / (c.stem + ".json")).exists()]
    if len(jobs) == 1 and a.workers == 1:
        print(*_one(jobs[0]), flush=True)
        return
    with ProcessPoolExecutor(a.workers) as ex:
        for stem, v in ex.map(_one, jobs):
            print(stem, v, flush=True)


if __name__ == "__main__":
    main()
