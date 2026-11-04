"""Artifact inventory and version behavior exercised through real file formats."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).parents[1]


def _load(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ArchiveTests(unittest.TestCase):
    def test_archive_rejects_missing_core_manifest(self):
        checker = _load("check_sdist")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            with self.assertRaisesRegex(ValueError, "missing source distribution"):
                checker.check_archive(archive)

    def test_archive_rejects_unsafe_members(self):
        checker = _load("check_sdist")
        for name in ("../outside", "/absolute", "package/rust/target/stale"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "source.tar.gz"
                with tarfile.open(archive, "w:gz") as handle:
                    handle.addfile(tarfile.TarInfo(name))
                with self.assertRaises(ValueError):
                    checker.check_archive(archive)

    def test_archive_requires_svml_notice(self):
        checker = _load("check_sdist")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in checker.REQUIRED - {"LICENSES/Intel-SVML-BSD-3-Clause.txt"}:
                    handle.addfile(tarfile.TarInfo("package/" + name))
            with self.assertRaisesRegex(ValueError, "Intel-SVML-BSD-3-Clause"):
                checker.check_archive(archive)

    def test_expected_inventory_rejects_missing_embedded_table(self):
        checker = _load("check_sdist")
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in checker.REQUIRED:
                    handle.addfile(tarfile.TarInfo("package/" + name))
            try:
                checker.check_archive(archive, {"rust/crates/behave-fire/src/generated/fuels.rs"})
            except TypeError:
                self.fail("check_archive lacks expected-inventory validation")
            except ValueError as error:
                self.assertIn("fuels.rs", str(error))
            else:
                self.fail("archive without an expected embedded table passed")

    def test_source_version_fallback_reads_binding_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src/example").mkdir(parents=True)
            (root / "rust/crates/behave-py").mkdir(parents=True)
            (root / "rust/crates/behave-py/Cargo.toml").write_text('[package]\nversion="1.0.0-beta.2"\n')
            (root / "src/example/_version.py").write_bytes((ROOT / "src/behave_fire/_version.py").read_bytes())
            (root / "src/example/__init__.py").write_text('from ._version import get_version\n__version__=get_version()\n')
            source = f'import sys; sys.path.insert(0,{str(root / "src")!r}); import example; assert example.__version__ == "1.0.0b2"'
            # Conflicting installed metadata must not mask this source checkout.
            installed = root / "unrelated/behave_fire-9.9.9.dist-info"
            installed.mkdir(parents=True)
            (installed / "METADATA").write_text("Metadata-Version: 2.4\nName: behave-fire\nVersion: 9.9.9\n")
            source = f"import sys; sys.path.insert(0, {str(installed.parent)!r}); " + source
            import venv
            venv.EnvBuilder().create(root / "venv")
            python = root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            subprocess.run([str(python), "-I", "-c", source], check=True)


if __name__ == "__main__":
    unittest.main()
