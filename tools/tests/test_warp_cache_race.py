"""warp_cpu.link_image_cache under a start-up race (2026-10-11, the validator's round: 6 of 23 workers died with
FileExistsError at the cache symlink when they started together). Eight processes link the same 200 module directories
into the same per-user cache at once, released by a barrier; every process must finish without an error and every link
must point at its module. Without the atomic rename (a plain `link.symlink_to` after an exists check) this fails within
a few runs. Needs no Warp:  python tools/tests/test_warp_cache_race.py"""
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from topogym.world.mujoco.warp_cpu import link_image_cache  # noqa: E402

N_PROC, N_MODULES, ROUNDS = 8, 200, 5


def worker(barrier, image, own, errors):
    barrier.wait()
    try:
        link_image_cache(Path(image), Path(own))
    except Exception as exc:          # reported to the parent
        errors.put(repr(exc))


def main():
    ctx = mp.get_context("fork")
    for r in range(ROUNDS):
        with tempfile.TemporaryDirectory() as root:
            image, own = Path(root) / "image" / "1.18.0", Path(root) / "own" / "1.18.0"
            for i in range(N_MODULES):
                (image / f"wp_module_{i:04d}").mkdir(parents=True)
            barrier, errors = ctx.Barrier(N_PROC), ctx.Queue()
            procs = [ctx.Process(target=worker, args=(barrier, str(image), str(own), errors)) for _ in range(N_PROC)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            failed = []
            while not errors.empty():
                failed.append(errors.get())
            assert not failed and all(p.exitcode == 0 for p in procs), (r, failed[:3])
            links = sorted(own.iterdir())
            assert [l.name for l in links] == sorted(m.name for m in image.iterdir()), (r, "missing or stray entries")
            assert all(l.is_symlink() and l.resolve() == (image / l.name).resolve() for l in links), (r, "bad link")
    print(f"ok: {ROUNDS} rounds x {N_PROC} processes x {N_MODULES} modules, no error, every link correct")


if __name__ == "__main__":
    main()
