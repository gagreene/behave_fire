"""Check wheel metadata using real ZIP archives at the tool boundary."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).parents[1]
NATIVE_TAGS = (
    "cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64",
    "cp311-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64",
    "cp311-abi3-musllinux_1_2_x86_64",
    "cp311-abi3-macosx_10_13_x86_64",
    "cp311-abi3-macosx_11_0_arm64",
    "cp311-abi3-win_amd64",
    "cp311-abi3-win_arm64",
)


def _distributions(directory):
    paths = {tag: _wheel(directory, tag, "original attribution") for tag in (*NATIVE_TAGS, "py3-none-any")}
    # Source contents are checked by check_sdist.py; this boundary checks inventory.
    paths["sdist"] = directory / "behave_fire-1.0.0b2.tar.gz"
    paths["sdist"].write_bytes(b"source inventory fixture")
    return paths


def _wheel(directory, tag, notice):
    path = directory / ("behave_fire-1.0.0b2-" + tag + ".whl")
    prefix = "behave_fire-1.0.0b2.dist-info/"
    metadata = "Metadata-Version: 2.4\nName: behave-fire\nVersion: 1.0.0b2\nLicense-Expression: MIT AND BSD-3-Clause\n\nDescription\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(prefix + "METADATA", metadata)
        for name, content in (("LICENSE", "MIT"), ("NOTICE", notice), ("LICENSES/NumPy-BSD-3-Clause.txt", "BSD-3-Clause"), ("LICENSES/Intel-SVML-BSD-3-Clause.txt", "Intel BSD-3-Clause")):
            archive.writestr(prefix + "licenses/" + name, content)
    return path


class MetadataTests(unittest.TestCase):
    def test_changed_native_notice_blocks_artifact_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _distributions(directory)
            _wheel(directory, "cp311-abi3-win_amd64", "missing attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("license mismatch", result.stderr)

    def test_duplicate_native_platform_blocks_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _distributions(directory)
            _wheel(directory, "cp311-abi3-macosx_12_0_arm64", "original attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native wheel", result.stderr)

    def test_identical_metadata_and_notices_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _distributions(directory)
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


    def test_missing_native_platform_blocks_aggregation(self):
        for tag in NATIVE_TAGS:
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                paths = _distributions(directory)
                paths[tag].unlink()
                result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("native wheel", result.stderr)

    def test_missing_pure_or_source_archive_blocks_aggregation(self):
        for missing in ("py3-none-any", "sdist"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                paths = _distributions(directory)
                paths[missing].unlink()
                result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_pure_only_directory_cannot_claim_native_metadata_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _wheel(directory, "py3-none-any", "original attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native wheel", result.stderr)

    def test_wrong_native_platform_or_interpreter_tag_blocks_aggregation(self):
        for tag in ("cp311-abi3-win32", "cp311-abi3-linux_x86_64", "cp310-cp310-win_arm64"):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                paths = _distributions(directory)
                paths["cp311-abi3-win_arm64"].unlink()
                _wheel(directory, tag, "original attribution")
                result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

if __name__ == "__main__":
    unittest.main()
