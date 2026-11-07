"""Deterministic raster cases and fresh-process block ordering."""

import random


OPERATIONS = ("surface", "crown", "typed-crown", "typed-crown-b")


def assert_outputs(actual, expected):
    """Compare every numerical value and structural property outside measured work.

    :param actual: Native output dictionary or numerical array.
    :param expected: Explicit Python result on identical inputs.
    :return: None; raises AssertionError on a contract difference.
    """
    import numpy as np
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), "Output fields differ"
        for key in expected:
            try:
                assert_outputs(actual[key], expected[key])
            except AssertionError as error:
                raise AssertionError(f"{key}: {error}") from error
        return
    left, right = np.asanyarray(actual), np.asanyarray(expected)
    assert left.shape == right.shape, f"Shapes differ: {left.shape} != {right.shape}"
    assert left.dtype == right.dtype, f"Dtypes differ: {left.dtype} != {right.dtype}"
    assert np.ma.isMaskedArray(left) == np.ma.isMaskedArray(right), "Masked-array class differs"
    np.testing.assert_array_equal(np.ma.getmaskarray(left), np.ma.getmaskarray(right))
    if np.ma.isMaskedArray(right):
        np.testing.assert_equal(left.fill_value, right.fill_value)
    left, right = np.asarray(np.ma.getdata(left)), np.asarray(np.ma.getdata(right))
    if right.dtype.kind in "biu":
        np.testing.assert_array_equal(left, right)
        return
    assert right.dtype.kind in "fc", "Benchmark outputs must be numerical arrays"
    parts = ((left.real, right.real), (left.imag, right.imag)) if right.dtype.kind == "c" else ((left, right),)
    single = right.dtype in (np.dtype("float32"), np.dtype("complex64"))
    for left_part, right_part in parts:
        np.testing.assert_array_equal(np.isnan(left_part), np.isnan(right_part))
        np.testing.assert_array_equal(np.isposinf(left_part), np.isposinf(right_part))
        np.testing.assert_array_equal(np.isneginf(left_part), np.isneginf(right_part))
        zero = (left_part == 0) & (right_part == 0)
        np.testing.assert_array_equal(np.signbit(left_part[zero]), np.signbit(right_part[zero]))
        np.testing.assert_allclose(left_part, right_part, rtol=1e-6 if single else 1e-9,
                                   atol=1e-7 if single else 1e-12, equal_nan=True)


def build_inputs(cells, dataset, seed=20261105):
    """Build normalized, shape-matching raster inputs without constructing a model.

    :param cells: Nonnegative logical cell count.
    :param dataset: Uniform, heterogeneous, or continuously varying diagnostic case.
    :param seed: Stable phase offset for spatial variation.
    :return: Surface-run arguments in component base units.
    """
    import numpy as np
    if dataset not in ("uniform", "heterogeneous", "continuous") or type(cells) is not int or cells < 0:
        raise ValueError("Expected a nonnegative cell count and a known dataset")
    inputs = dict(fuel_model_grid=np.full(cells, 10, dtype=np.int32),
                  m1h=np.full(cells, .06), m10h=np.full(cells, .07),
                  m100h=np.full(cells, .08), mlh=np.full(cells, .6),
                  mlw=np.full(cells, .9), wind_speed=np.full(cells, 8.0),
                  wind_speed_units=0, wind_direction=np.full(cells, 35.0),
                  wind_orientation_mode="RelativeToUpslope", slope=np.full(cells, 12.0))
    if dataset in ("heterogeneous", "continuous"):
        index = np.arange(cells)
        phase = ((index + seed) % 17) / 16
        if dataset == "continuous":
            # Irrational rotation avoids the repeating 17-cell acceptance cycle.
            phase = np.mod((index + seed) * ((5.0 ** 0.5 - 1.0) / 2.0), 1.0)
        inputs["fuel_model_grid"] = np.array([1, 4, 10, 101, 124, 145, 201, 0], dtype=np.int32)[index % 8]
        for field, low, width in (("m1h", .03, .07), ("m10h", .04, .08), ("m100h", .05, .09),
                                  ("mlh", .35, .9), ("mlw", .6, .8), ("wind_speed", 2., 160.),
                                  ("wind_direction", 0., 350.), ("slope", 0., 35.)):
            inputs[field] = low + width * phase
    return inputs


def paired_schedule(repetitions, seed, workers):
    """Shuffle each block while sharing one reference across native worker settings.

    :param repetitions: Number of independent paired blocks.
    :param seed: Reproducible shuffle seed.
    :param workers: Distinct positive or automatic native worker requests.
    :return: Ordered block/backend/worker triples with one Python member per block.
    """
    if type(repetitions) is not int or repetitions < 1 or len(set(workers)) != len(workers):
        raise ValueError("Require positive repetitions and distinct worker requests")
    if any(type(worker) is not int or worker < 0 for worker in workers):
        raise ValueError("Worker requests must be nonnegative integers")
    rng = random.Random(seed)
    schedule = []
    for block in range(repetitions):
        members = [("python", 1), *(("rust", worker) for worker in workers)]
        rng.shuffle(members)
        schedule.extend((block, backend, worker) for backend, worker in members)
    return schedule


def supplied_surface(inputs, operation, python_model, uniform=False):
    """Generate identical supplied Surface values before Crown measurement.

    :param inputs: Already constructed Surface input arrays.
    :param operation: Common Crown or one of the independently typed layouts.
    :param python_model: Explicit Python reference model, never an automatic facade.
    :param uniform: Reproduce the historical single-cell prototype and repeated values.
    :return: Supplied Surface dictionary with the requested dtype/mask layout.
    """
    import numpy as np
    cells = inputs["fuel_model_grid"].size
    if uniform:
        prototype = python_model.do_surface_run(**build_inputs(1, "uniform"))
        surface = {key: np.full(cells, value.item(), dtype=value.dtype) for key, value in prototype.items()}
    else:
        surface = python_model.do_surface_run(**inputs)
    if operation == "typed-crown":
        surface["spread_rate"] = surface["spread_rate"].astype(np.float32)
        surface["fireline_intensity"] = np.full(cells, 200, dtype=np.uint16)
        surface["heat_per_unit_area"] = np.ma.array(surface["heat_per_unit_area"], mask=np.arange(cells) % 7 == 0)
    elif operation == "typed-crown-b":
        surface["spread_rate"] = np.ma.array(surface["spread_rate"], mask=np.arange(cells) % 5 == 0)
        surface["fireline_intensity"] = np.full(cells, 200, dtype=np.int32)
        surface["heat_per_unit_area"] = surface["heat_per_unit_area"].astype(np.float32)
    return surface
