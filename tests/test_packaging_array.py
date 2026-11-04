"""Packaging contract tests independent of the original checkout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_license_payloads_have_canonical_line_endings(self):
        root = TOOLS.parent
        for base in (root, root / "rust/crates/behave-fire"):
            for name in ("LICENSE", "NOTICE", "LICENSES/NumPy-BSD-3-Clause.txt"):
                with self.subTest(path=base / name):
                    self.assertNotIn(b"\r", (base / name).read_bytes())

    def test_compiled_sources_are_rejected_before_staging(self):
        builder = _load("build_pure_wheel")
        for suffix in (".so", ".pyd", ".dylib", ".pdb"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "src/behave_fire").mkdir(parents=True)
                (root / ("src/behave_fire/stale" + suffix)).touch()
                with self.assertRaisesRegex(ValueError, "compiled"):
                    builder.stage_project(root, root / "stage")

    def test_numpy_markers_cover_exact_conditional_floor(self):
        import tomllib
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        project = tomllib.loads((TOOLS.parent / "pyproject.toml").read_text())["project"]
        for python, machine, system, expected in (
            ("3.11", "AMD64", "Windows", [">=2.1.3"]),
            ("3.13", "x86_64", "Linux", [">=2.1.3"]),
            ("3.13", "ARM64", "Windows", [">=2.3.3"]),
            ("3.14", "ARM64", "Windows", [">=2.3.3"]),
            ("3.14", "x86_64", "Linux", [">=2.3.3"]),
        ):
            environment = dict(default_environment(), python_version=python, python_full_version=python+".0", platform_machine=machine, platform_system=system)
            actual = [str(r.specifier) for text in project["dependencies"] if (r := Requirement(text)).marker.evaluate(environment)]
            self.assertEqual(actual, expected)

    def test_pure_configuration_preserves_metadata_and_notices(self):
        import tomllib
        builder = _load("build_pure_wheel")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src/behave_fire").mkdir(parents=True)
            (root / "src/behave_fire/__init__.py").write_text("")
            (root / "rust/crates/behave-py").mkdir(parents=True)
            (root / "rust/crates/behave-py/Cargo.toml").write_text('[package]\nversion="1.0.0-beta.2"\n')
            (root / "pyproject.toml").write_bytes((TOOLS.parent / "pyproject.toml").read_bytes())
            for file in ("README.md", "LICENSE", "NOTICE", "LICENSES/NumPy-BSD-3-Clause.txt"):
                (root / file).parent.mkdir(parents=True, exist_ok=True)
                (root / file).write_text("test notice")
            stage = root / "stage"
            builder.stage_project(root, stage)
            original = tomllib.loads((root / "pyproject.toml").read_text())["project"]
            actual = tomllib.loads((stage / "pyproject.toml").read_text())["project"]
            original.pop("dynamic")
            original["version"] = "1.0.0b2"
            self.assertEqual(actual, original)
            self.assertTrue((stage / "LICENSES/NumPy-BSD-3-Clause.txt").is_file())
            self.assertFalse((stage / "rust").exists())

    def test_version_mapping_rejects_unsupported_cargo_prerelease(self):
        builder = _load("build_pure_wheel")
        self.assertEqual(builder.python_version("1.0.0-beta.2"), "1.0.0b2")
        self.assertEqual(builder.python_version("2.1.0-rc.3"), "2.1.0rc3")
        with self.assertRaises(ValueError):
            builder.python_version("1.0.0-preview.1")


if __name__ == "__main__":
    unittest.main()
