"""Check wheel metadata using real ZIP archives at the tool boundary."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).parents[1]


def _wheel(directory, tag, notice):
    path = directory / ("behave_fire-1.0.0b2-" + tag + ".whl")
    prefix = "behave_fire-1.0.0b2.dist-info/"
    metadata = "Metadata-Version: 2.4\nName: behave-fire\nVersion: 1.0.0b2\nLicense-Expression: MIT AND BSD-3-Clause\n\nDescription\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(prefix + "METADATA", metadata)
        for name, content in (("LICENSE", "MIT"), ("NOTICE", notice), ("LICENSES/NumPy-BSD-3-Clause.txt", "BSD-3-Clause")):
            archive.writestr(prefix + "licenses/" + name, content)
    return path


class MetadataTests(unittest.TestCase):
    def test_changed_native_notice_blocks_artifact_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _wheel(directory, "py3-none-any", "original attribution")
            _wheel(directory, "cp311-abi3-win_amd64", "missing attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("license mismatch", result.stderr)

    def test_identical_metadata_and_notices_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _wheel(directory, "py3-none-any", "original attribution")
            _wheel(directory, "cp311-abi3-win_amd64", "original attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


    def test_pure_only_directory_cannot_claim_native_metadata_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _wheel(directory, "py3-none-any", "original attribution")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_wheel_metadata.py"), str(directory)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native wheel", result.stderr)

if __name__ == "__main__":
    unittest.main()
