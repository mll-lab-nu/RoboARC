"""scripts/harbor/environments/roboarc_docker.py on a stand-in Harbor (standard library only; no docker, no Harbor):
  1. the copy's gost.yaml sets the handler's readTimeout to 900s, once, and is otherwise Harbor's text; every other file
     of the sidecar folder is copied byte for byte with its mode;
  2. the copy is made once and reused (the same folder for the same source), and a changed source gets another folder;
  3. a gost.yaml that already sets readTimeout is used as it is (no copy), and one without the expected handler block
     stops the import with an error;
  4. RoboARCDocker is a subclass of Harbor's DockerEnvironment whose sidecar build folder is the patched copy.
The end-to-end check (Harbor 0.23.0, oracle agent, a host server that sends its headers after 20 s, allowlisted) is in
the CHANGELOG entry: cut at 15 s with Harbor's environment, delivered with this one, other hosts still blocked.
usage: python3 scripts/harbor/tests/test_roboarc_docker.py"""
import importlib, os, stat, sys, tempfile, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "environments"))
sys.dont_write_bytecode = True

GOST = """# addr, so_mark, and file.path must stay in sync with the constants in
# bin/network-policy (GOST_PORT, GOST_MARK, ALLOWLIST); the `reload` interval
# must stay below RELOAD_WAIT there.
services:
  - name: transparent-egress
    addr: ":12345"
    bypass: allowlist
    metadata:
      so_mark: 114514
    handler:
      type: red
      metadata:
        sniffing: true
        sniffing.timeout: 5s
        sniffing.fallback: true
    listener:
      type: red

bypasses:
  - name: allowlist
    whitelist: true
    reload: 1s
    file:
      path: /opt/egress-sidecar/allowlist.txt
"""
errs, n = [], 0


def check(ok, msg):
    global n
    n += 1
    if not ok:
        errs.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="rd_"))
src = tmp / "harbor-docker-egress-control-sidecar"
(src / "bin").mkdir(parents=True)
(src / "gost.yaml").write_text(GOST)
(src / "Dockerfile").write_text("FROM gogost/gost\n")
(src / "entrypoint.sh").write_text("#!/bin/sh\n"); os.chmod(src / "entrypoint.sh", 0o755)
(src / "bin" / "network-policy").write_text("#!/bin/sh\n"); os.chmod(src / "bin" / "network-policy", 0o755)
(src / "allowlist.txt").write_text("")

# a stand-in for harbor.environments.docker.docker.DockerEnvironment
for name in ("harbor", "harbor.environments", "harbor.environments.docker", "harbor.environments.docker.docker"):
    sys.modules[name] = types.ModuleType(name)
DockerEnvironment = type("DockerEnvironment", (), {"_EGRESS_CONTROL_SIDECAR_CONTEXT_PATH": src})
sys.modules["harbor.environments.docker.docker"].DockerEnvironment = DockerEnvironment
os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
rd = importlib.import_module("roboarc_docker")

# 1, 4
dst = rd.RoboARCDocker._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH
check(issubclass(rd.RoboARCDocker, DockerEnvironment), "RoboARCDocker is not a DockerEnvironment")
check(dst != src and dst.parent == tmp / "cache" / "roboarc", f"the build folder is {dst}, not a copy in the cache")
text = (dst / "gost.yaml").read_text()
check(text.count("readTimeout") == 1 and "      metadata:\n        readTimeout: 900s\n        sniffing: true\n" in text,
      f"gost.yaml of the copy:\n{text}")
check(text.replace("        readTimeout: 900s\n", "") == GOST, "the copy's gost.yaml differs from Harbor's beyond the readTimeout line")
for rel in ("Dockerfile", "entrypoint.sh", "bin/network-policy", "allowlist.txt"):
    a, b = src / rel, dst / rel
    check(b.is_file() and a.read_bytes() == b.read_bytes() and stat.S_IMODE(a.stat().st_mode) == stat.S_IMODE(b.stat().st_mode),
          f"{rel} is not copied byte for byte with its mode")

# 2
again = rd.patched_context(src, tmp / "cache" / "roboarc")
check(again == dst, f"a second call made another folder: {again}")
check(len([p for p in (tmp / "cache" / "roboarc").iterdir()]) == 1, "leftovers in the cache")
(src / "Dockerfile").write_text("FROM gogost/gost:other\n")
other = rd.patched_context(src, tmp / "cache" / "roboarc")
check(other != dst and (other / "Dockerfile").read_text() == "FROM gogost/gost:other\n", "a changed source reused the old copy")

# 3
done = tmp / "already"; done.mkdir()
(done / "gost.yaml").write_text(GOST.replace("        sniffing: true\n", "        readTimeout: 60s\n        sniffing: true\n"))
check(rd.patched_context(done, tmp / "cache" / "roboarc") == done, "a gost.yaml with its own readTimeout was copied")
odd = tmp / "odd"; odd.mkdir()
(odd / "gost.yaml").write_text(GOST.replace("type: red\n      metadata:", "type: tcp\n      metadata:", 1))
try:
    rd.patched_context(odd, tmp / "cache" / "roboarc")
    check(False, "a gost.yaml without the handler block was accepted")
except RuntimeError:
    check(True, "")

for e in errs:
    print("FAIL", e)
print(f"{'FAIL' if errs else 'ok'} roboarc_docker (egress sidecar readTimeout): {n} checks, {len(errs)} failed")
sys.exit(1 if errs else 0)
