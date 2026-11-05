"""Build a universal fallback wheel from an isolated setuptools source tree."""

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

_COMPILED = {".so", ".pyd", ".dylib", ".pdb"}


def _toml_value(value):
    """Encode the PEP 621 data types as TOML values.

    :param value: String, list, dictionary, or Boolean metadata value.
    :return: TOML value preserving the metadata structure.
    """
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(key) + " = " + _toml_value(item) for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def build(root, output):
    """Build and inspect the fallback wheel without touching source metadata.

    :param root: Repository containing the binding manifest and Python source.
    :param output: Destination directory for the completed wheel.
    :return: Path of the validated wheel written to output.
    """
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="behave-pure-") as temporary:
        stage = Path(temporary) / "source"
        stage_project(root, stage)
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(output), str(stage)], check=True)
    version = python_version(tomllib.loads((root / "rust/crates/behave-py/Cargo.toml").read_text())["package"]["version"])
    wheel = output / f"behave_fire-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if Path(name).suffix.lower() in _COMPILED or name.startswith("rust/") or "/_rust" in name:
                raise ValueError(f"compiled or Rust payload in pure wheel: {name}")
    return wheel


def main():
    """Run the isolated pure wheel builder.

    :return: None; writes one validated wheel to the requested directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    arguments = parser.parse_args()
    print(build(arguments.root.resolve(), arguments.out_dir))


def python_version(cargo_version):
    """Map supported Cargo semantic versions to Maturin's Python spelling.

    :param cargo_version: Binding package semantic version.
    :return: PEP 440 version; unknown prerelease conventions raise ValueError.
    """
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?(?:\+([0-9A-Za-z.-]+))?", cargo_version)
    if not match:
        raise ValueError(f"unsupported Cargo version: {cargo_version}")
    release, label, number, local = match.groups()
    return release + ({"alpha": "a", "beta": "b", "rc": "rc"}[label] + number if label else "") + ("+" + local if local else "")


def stage_project(root, stage):
    """Stage Python sources and unchanged project metadata for setuptools.

    :param root: Original repository root, which remains unchanged.
    :param stage: New staging directory outside the source tree.
    :return: None; copies Python sources, readme, notices, and build metadata.
    """
    source = root / "src/behave_fire"
    compiled = [path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in _COMPILED]
    if compiled:
        raise ValueError(f"compiled files in source tree: {compiled}")
    metadata = copy.deepcopy(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"])
    metadata.pop("dynamic", None)
    cargo = tomllib.loads((root / "rust/crates/behave-py/Cargo.toml").read_text(encoding="utf-8"))
    metadata["version"] = python_version(cargo["package"]["version"])
    shutil.copytree(source, stage / "src/behave_fire", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build", "dist"))
    readme = metadata["readme"]
    files = [readme if isinstance(readme, str) else readme["file"], *metadata["license-files"]]
    for name in files:
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    text = '[build-system]\nrequires = ["setuptools>=77.0.3", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\n'
    text += "\n".join(json.dumps(key) + " = " + _toml_value(value) for key, value in metadata.items())
    text += '\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'
    (stage / "pyproject.toml").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
