"""Regression tests for canonical source/output paths, independent of /tmp layout.
All filesystem writes are isolated in TemporaryDirectory fixtures.
The original tests/test_parallel.py is retained without alteration.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_interfaces as ci


class PathGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qcml_path_guard_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "source"
        self.root.mkdir()
        self.alias = self.base / "source_alias"
        self.alias.symlink_to(self.root, target_is_directory=True)

    def reject(self, out, roots):
        with self.assertRaises(ValueError):
            ci.safe_parent(out, roots)
        # A rejected nonexistent output must not be created in the source.
        self.assertFalse(Path(out).exists())

    def test_canonical_child_rejected_without_creation(self):
        self.reject(self.root / "new" / "output", [self.root])

    def test_alias_child_with_alias_root_rejected(self):
        self.reject(self.alias / "out", [self.alias])

    def test_real_output_with_alias_root_rejected(self):
        self.reject(self.root / "out", [self.alias])

    def test_alias_output_with_real_root_rejected(self):
        self.reject(self.alias / "out", [self.root])

    def test_same_directory_alias_rejected(self):
        with self.assertRaises(ValueError):
            ci.safe_parent(self.alias, [self.root])
        with self.assertRaises(ValueError):
            ci.safe_parent(self.root, [self.alias])

    def test_output_symlink_into_source_rejected(self):
        hidden = self.root / "inner"
        hidden.mkdir()
        link = self.base / "apparently_outside"
        link.symlink_to(hidden, target_is_directory=True)
        self.reject(link / "out", [self.alias])

    def test_relative_root_and_parent_rejected(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.base)
            self.reject(Path("source_alias") / "out", [Path("source_alias")])
        finally:
            os.chdir(cwd)

    def test_dotdot_and_string_root_rejected(self):
        self.reject(self.alias / "sub" / ".." / "out", [str(self.alias)])

    def test_home_expansion_rejected(self):
        with patch.dict(os.environ, {"HOME": str(self.base)}):
            with self.assertRaises(ValueError):
                ci.safe_parent("~/source_alias/out", ["~/source_alias"])
        self.assertFalse((self.root / "out").exists())

    def test_multiple_roots_checked_before_mkdir(self):
        other = self.base / "other_source"
        other.mkdir()
        self.reject(self.root / "out", [other, self.alias])

    def test_same_prefix_sibling_allowed(self):
        out = self.base / "source_extra" / "out"
        self.assertEqual(ci.safe_parent(out, [self.alias]), out.resolve())
        self.assertTrue(out.is_dir())

    def test_external_output_alias_allowed_and_canonical(self):
        external = self.base / "external"
        external.mkdir()
        external_alias = self.base / "external_alias"
        external_alias.symlink_to(external, target_is_directory=True)
        out = external_alias / "out"
        self.assertEqual(ci.safe_parent(out, [self.alias]), external / "out")
        self.assertTrue((external / "out").is_dir())

    def test_collect_rejects_alias_output_without_artifacts(self):
        self.reject(self.alias / "out", [self.root])
        with self.assertRaises(ValueError):
            ci.collect([self.alias], self.alias / "out")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob("qcml_physics_interfaces_*")))


if __name__ == "__main__":
    unittest.main()
