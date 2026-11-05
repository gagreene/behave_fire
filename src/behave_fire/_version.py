"""Expose installed metadata, with a binding-manifest fallback for source checkouts."""

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
import tomllib


def get_version():
    """Return this installed distribution's version or its source binding version.

    :return: PEP 440 version; unrelated installed metadata cannot mask a checkout.
    """
    package_root = Path(__file__).resolve().parent
    try:
        installed = distribution("behave-fire")
    except PackageNotFoundError:
        pass
    else:
        if Path(installed.locate_file("behave_fire")).resolve() == package_root:
            return installed.version
    manifest = package_root.parents[1] / "rust/crates/behave-py/Cargo.toml"
    cargo = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["version"]
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?(?:\+([0-9A-Za-z.-]+))?", cargo)
    if not match:
        raise ValueError(f"unsupported Cargo version: {cargo}")
    release, label, number, local = match.groups()
    return release + ({"alpha": "a", "beta": "b", "rc": "rc"}[label] + number if label else "") + ("+" + local if local else "")
