"""Smoke-test an installed distribution with isolated Python outside the checkout."""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig
import warnings


def _compare(actual, expected):
    """Compare nested calculation fields including masks and nonfinite values.

    :param actual: Installed selected-backend output.
    :param expected: Installed explicit Python reference output.
    :return: None; raises AssertionError on numerical or mask differences.
    """
    import numpy as np
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
        np.testing.assert_allclose(np.ma.getdata(actual), np.ma.getdata(expected), rtol=1e-9, atol=1e-12, equal_nan=True)


def _direction(selected, reference, normalize_half_loop):
    """Check widened native angles against Python on the same float64 inputs.

    :param selected: Installed selected surface backend.
    :param reference: Installed explicit Python surface backend.
    :param normalize_half_loop: Whether to use the boolean fixture instead of uint16.
    :return: None; raises AssertionError on structure or strict live parity differences.
    """
    import numpy as np
    x = np.array([-1], dtype=np.int8)
    y = np.array([False], dtype=np.bool_) if normalize_half_loop else np.array([65534], dtype=np.uint16)
    reference_x, reference_y = x.astype(np.float64), y.astype(np.float64)
    expected = reference.calculate_direction_of_max_spread(reference_x, reference_y, np.ones(1), "RelativeToNorth", 95.0)
    actual = selected.calculate_direction_of_max_spread(x, y, np.ones(1), "RelativeToNorth", 95.0)
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape == (1,)
    assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(expected)
    _compare(actual, expected)


def _direction_reversed(selected, reference):
    """Check direction parity after widening reversed float32 operands.

    :param selected: Installed native surface backend.
    :param reference: Installed explicit Python surface backend.
    :return: None; raises on dtype, shape, or strict North-direction differences.
    """
    import numpy as np
    x = np.full(7, -1., dtype=np.float32)[::-1]
    y = np.full(7, np.finfo(np.float32).max, dtype=np.float32)[::-1]
    rate = np.ones(7)
    expected = reference.calculate_direction_of_max_spread(x.astype(np.float64), y.astype(np.float64), rate, "RelativeToNorth", 95.)
    actual = selected.calculate_direction_of_max_spread(x, y, rate, "RelativeToNorth", 95.)
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape == (7,)
    _compare(actual, expected)


def _native_contract(selected):
    """Verify installed numeric storage, independent outputs, stages, and ownership.

    :param selected: Installed explicit Rust namespace.
    :return: Successful float64, staged injection, and owned-output certification flags.
    """
    import numpy as np

    original = np.ma.array(np.array([1., 2.], dtype=np.float32), mask=[False, True])
    result = selected.surface.calculate_flame_length(original)
    assert type(result) is np.ndarray and result.dtype == np.float64
    np.testing.assert_allclose(result, [.45, np.nan], rtol=1e-9, atol=1e-12, equal_nan=True)
    original.data[:] = 99.
    original.mask[:] = False
    np.testing.assert_allclose(result, [.45, np.nan], rtol=1e-9, atol=1e-12, equal_nan=True)
    classified = selected.crown.classify_crown_fire_type(np.array([np.nan, 2.]), np.array([2., 2.]))
    assert classified['fire_type'].dtype == np.int32
    np.testing.assert_array_equal(classified['fire_type'], [-1, 3])
    for name in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
        assert classified[name].dtype == np.bool_ and not classified[name][0]
    independent = selected.crown.calculate_crown_heat_and_intensity(1000., np.array([]), 3., 30., .02)
    assert independent['canopy_heat_per_unit_area'].size == 1
    assert np.isfinite(independent['canopy_heat_per_unit_area']).all()
    assert independent['crown_fire_line_intensity'].size == 0
    moisture = np.full((2, 3), .06)
    grid = selected.SurfaceGrid(dict(fuel_model_grid=124, m1h=moisture, m10h=.07, m100h=.08,
        mlh=.6, mlw=.9, wind_speed=88., wind_direction=35., slope=12., aspect=15.,
        canopy_cover=.4, canopy_height=30., crown_ratio=.5), shape=(2, 3), threads=4)
    moisture[:] = np.nan
    for stage in ('particles', 'fuelbed', 'reaction'):
        grid.run_stage(stage)
    reaction = grid.get('reaction_intensity')
    assert np.isfinite(reaction).all()
    grid.set('reaction_intensity', reaction*.5)
    grid.run_remaining()
    reduced = grid.get('spread_rate')
    copy = grid.get('spread_rate')
    copy[:] = -999.
    np.testing.assert_array_equal(grid.get('spread_rate'), reduced)
    grid.run_fused()
    assert np.all(grid.get('spread_rate') > reduced)
    return {'float64_contract': True, 'staged_injection': True, 'owned_outputs': True}


def _native_report():
    """Describe installed native identity using supported extension metadata.

    :return: Binding API version and the current physical storage contract.
    """
    from behave_fire import _rust

    return {"binding_api_version": _rust.__binding_api_version__, "native_physical_dtype": "float64"}


def _surface(model, grid):
    """Calculate a fixed non-contiguous GS4 surface scenario.

    :param model: Installed BehaveRun instance.
    :param grid: Possibly strided fuel model identifiers.
    :return: Complete surface output dictionary.
    """
    import numpy as np
    return model.do_surface_run(fuel_model_grid=grid, m1h=np.full(grid.shape, .06), m10h=np.full(grid.shape, .07), m100h=np.full(grid.shape, .08), mlh=np.full(grid.shape, .60), mlw=np.full(grid.shape, .90),
                                wind_speed=5., wind_speed_units=5, wind_direction=0., wind_orientation_mode="RelativeToUpslope",
                                slope=np.degrees(np.arctan(.3)), slope_units=0, aspect=0., canopy_cover=.5, canopy_height=30., crown_ratio=.5)


def main():
    """Check import provenance, backend selection, parity, errors, and runtime identity.

    :return: None; prints machine-readable certification evidence on success.
    """
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("python", "rust"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--identity", choices=("windows-arm64", "musl-x86_64"))
    arguments = parser.parse_args()
    assert sys.flags.isolated, "use python -I outside the checkout"
    assert not sysconfig.get_config_var("Py_GIL_DISABLED"), "free-threaded interpreters are not certified"
    import behave_fire as bf
    package = Path(bf.__file__).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    assert package.is_relative_to(site), (package, site)
    assert bf.__version__ == arguments.version == importlib.metadata.version("behave-fire")
    expected = ["python", "rust"] if arguments.expect == "rust" else ["python"]
    with warnings.catch_warnings(record=True) as discovery:
        warnings.simplefilter('always')
        assert bf.available_backends() == expected
    assert not discovery, 'Backend discovery must stay silent and preserve fallback warnings'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        automatic = bf.BehaveRun(bf.FuelModels())
    if arguments.expect == "python":
        assert len(caught) == 1 and caught[0].category is bf.BehaveBackendWarning
        assert "Python" in str(caught[0].message)
    else:
        assert not caught, [str(item.message) for item in caught]
    assert automatic.backend_in_use == arguments.expect
    spec = importlib.util.find_spec("behave_fire._rust")
    if arguments.expect == "rust":
        import behave_fire._rust as native
        assert Path(native.__file__).resolve().is_relative_to(site)
        assert spec is not None
    else:
        assert spec is None
        for action in (lambda: bf.BehaveRun(bf.FuelModels(), backend="rust"), lambda: bf.get_backend("rust"), lambda: importlib.import_module("behave_fire.rust")):
            try:
                action()
            except ImportError as error:
                assert "install" in str(error).lower() or "_rust" in str(error)
            else:
                raise AssertionError("explicit Rust accepted a pure wheel")
    grid = np.full((3, 6), 124, dtype=np.int32)[:, ::2]
    assert not grid.flags.c_contiguous
    reference = bf.BehaveRun(bf.FuelModels(), backend="python")
    golden = _surface(reference, grid)
    np.testing.assert_allclose(golden["spread_rate"], 9.763838, rtol=0, atol=5e-7)
    _compare(_surface(automatic, grid), golden)
    registry = bf.FuelModels()
    registry.fuel_models_[124]["fuel_bed_depth"] *= 1.25
    snapshot_reference = bf.BehaveRun(registry, backend="python")
    snapshot_actual = bf.BehaveRun(registry, backend=arguments.expect)
    before = _surface(snapshot_reference, grid)
    registry.fuel_models_[124]["fuel_bed_depth"] *= 2
    _compare(_surface(snapshot_actual, grid), before)
    selected = bf.get_backend(arguments.expect)
    python = bf.get_backend("python")
    if arguments.expect == "rust":
        contract = _native_contract(selected)
        for normalize_half_loop in (False, True):
            _direction(selected.surface, python.surface, normalize_half_loop)
        _direction_reversed(selected.surface, python.surface)
    for module, function, values in (
        ("safety", "calculate_safety_zone", (5., 2., 50., 1., 300.)),
        ("vapor_pressure_deficit_calculator", "calculate_vpd", (25., 1, .5, 0, True)),
        ("crown", "calculate_crown_fire_ratios", (np.ones(2), np.ones(3), 2., 4.)),
    ):
        _compare(getattr(getattr(selected, module), function)(*values), getattr(getattr(python, module), function)(*values))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected.ignite.calculate_lightning_ignition_probability(6, .2, 1., .5)
    assert any(item.category is DeprecationWarning for item in caught)
    try:
        bf.get_backend("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid backend accepted")
    identity = {"machine": platform.machine(), "platform": platform.platform(), "libc": platform.libc_ver(), "sysconfig": sysconfig.get_platform()}
    if arguments.identity == "windows-arm64":
        from pip._vendor.packaging.tags import sys_tags
        assert platform.system() == "Windows" and platform.machine() == "ARM64", identity
        assert next(sys_tags()).platform == "win_arm64"
    if arguments.identity == "musl-x86_64":
        import subprocess
        libc = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
        assert "musl" in libc.stdout + libc.stderr and platform.machine() == "x86_64", identity
        identity["musl"] = (libc.stdout + libc.stderr).strip()
    report = {"result": "passed", "python": sys.version, "executable": sys.executable, "identity": identity,
              "numpy": np.__version__, "version": bf.__version__, "backend": automatic.backend_in_use,
              "package": str(package)}
    if arguments.expect == "rust":
        report.update(_native_report())
        report.update(contract)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
