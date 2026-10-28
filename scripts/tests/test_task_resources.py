"""Tests for the family classes of scripts/task_resources.toml, read by scripts/task_conventions.py (the declared
resources) and scripts/harbor/gpu_slots.py (a GPU of its own for the own_gpu classes). Standard library only (3.11+):

    python3 scripts/tests/test_task_resources.py

- every RoboARC task that `classes_required` finds has its family in exactly one class, and every class's families are
  tasks of tasks/;
- a task whose family is in no class, a family in two classes and a rule naming an undeclared class are refused;
- a class's rules apply to its families only: the families released after the classes (bed_bolt, wire_splicing, ...)
  get their class's values, IKEA none;
- gpu_slots.py's own reader (no TOML parser) gives the same classes as tomllib.
"""
import pathlib
import re
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import task_conventions as TC  # noqa: E402
sys.path.insert(0, str(REPO / "scripts" / "harbor"))
import gpu_slots  # noqa: E402

RULES = tomllib.loads(TC.RESOURCES_FILE.read_text())


def fake_task(root: pathlib.Path, name: str, family: str) -> pathlib.Path:
    t = root / name
    t.mkdir()
    (t / "task.toml").write_text(f'[metadata]\nfamily = "{family}"\n')
    return t


class Classes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_task_in_one_class(self):
        tasks = sorted(p.parent for p in (REPO / "tasks").glob("roboarc-*/task.toml"))
        if not tasks:
            self.skipTest("no tasks/")
        seen = set()
        for t in tasks:
            with self.subTest(task=t.name):
                cls = TC.family_class(t, RULES)
                if re.search(RULES["classes_required"], t.name):
                    self.assertIsNotNone(cls)
                    seen.add(tomllib.loads((t / "task.toml").read_text())["metadata"]["family"])
                else:
                    self.assertIsNone(cls)                               # IKEA: its own rules, no class
        declared = {f for c in RULES["class"].values() for f in c["families"]}
        self.assertEqual(declared, seen)                                 # no class lists a family tasks/ lacks

    def test_family_without_class(self):
        t = fake_task(self.root, "roboarc-new-family-01-i01-privileged", "new_family")
        with self.assertRaisesRegex(RuntimeError, "'new_family' is in no class"):
            TC.family_class(t, RULES)
        with self.assertRaisesRegex(RuntimeError, "no class"):
            TC.resources_for(t)
        ikea = fake_task(self.root, "roboarc-ikea-new-i00-privileged", "ikea_new")
        self.assertIsNone(TC.family_class(ikea, RULES))

    def test_family_in_two_classes(self):
        rules = {"classes_required": "^roboarc-", "class": {"a": {"families": ["f"]}, "b": {"families": ["f"]}}}
        t = fake_task(self.root, "roboarc-f-01-i01-privileged", "f")
        with self.assertRaisesRegex(RuntimeError, "'f' is in the classes 'a' and 'b'"):
            TC.family_class(t, rules)

    def test_rule_with_undeclared_class(self):
        f = self.root / "task_resources.toml"
        f.write_text('classes_required = "^roboarc-"\n\n[class.a]\nfamilies = ["f"]\n\n'
                     '[[rule]]\nmatch = "."\nclass = "nope"\nmin_cpus = 1.0\n')
        t = fake_task(self.root, "roboarc-f-01-i01-privileged", "f")
        with mock.patch.object(TC, "RESOURCES_FILE", f):
            with self.assertRaisesRegex(RuntimeError, "class 'nope'"):
                TC.resources_for(t)

    def test_class_rules_apply_to_their_families(self):
        def res(base: str, mode: str):
            t = REPO / "tasks" / f"roboarc-{base}-{mode}"
            if not t.is_dir():
                self.skipTest(f"{t.name} is not in tasks/")
            return TC.resources_for(t)

        for base in ("cast-coil-04-i01", "bed-bolt-05-i01", "clock-assembling-01-i01"):   # rigid
            r = res(base, "privileged")
            self.assertEqual((r["min_cpus"], r["caps"]["main"]["cpus"], r["caps"]["main"]["memory_gb"]),
                             (4.0, 16.0, 30.0), base)
            self.assertEqual(res(base, "standard")["caps"]["sim"]["memory_gb"], 8.0, base)
        wire = res("wire-splicing-01-i01", "privileged")                                   # rope / chain
        self.assertEqual((wire["min_memory_gb"], wire["caps"]["main"]["memory_gb"]), (8.0, 40.0))
        cloth = res("blanket-fold-01-i01", "standard")
        self.assertEqual(cloth["min_cpus"], 6.0)
        ikea = res("ikea-lack-i00", "privileged")                                          # no class rule
        self.assertEqual(ikea["min_cpus"], 2.0)

    def test_gpu_slots_reads_the_same_classes(self):
        classes, required = gpu_slots.family_classes()
        self.assertEqual(required, RULES["classes_required"])
        want = {f: (name, bool(c.get("own_gpu", False))) for name, c in RULES["class"].items() for f in c["families"]}
        self.assertEqual(classes, want)


if __name__ == "__main__":
    unittest.main()
