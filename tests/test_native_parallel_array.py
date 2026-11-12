"""Focused native worker-count and ownership contracts."""
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.rust
_native = pytest.importorskip("behave_fire.rust", exc_type=ImportError)
BehaveRun, FuelModels = _native.BehaveRun, _native.FuelModels
from behave_fire.rust import _bridge
from parity_support import ROOT
from float64_model_reference import model_reference


def _surface_inputs(shape):
    return dict(fuel_model_grid=np.full(shape, 10, dtype=np.int32),
                m1h=np.full(shape, .06), m10h=np.full(shape, .07),
                m100h=np.full(shape, .08), mlh=np.full(shape, .6), mlw=np.full(shape, .9),
                wind_speed=np.full(shape, 8.0), wind_speed_units=0,
                wind_direction=np.full(shape, 35.0), wind_orientation_mode="RelativeToUpslope",
                slope=np.full(shape, 12.0))


@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize("independent", [False, True])
def test_broadcast_crown_preserves_independent_field_shapes(width, independent):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    inputs = _surface_inputs((2, 1))
    inputs["wind_speed"] = np.full((1, width), 8.0)
    inputs["wind_direction"] = 35.0
    inputs["slope"] = 12.0
    reference = PythonRun(PythonFuels())
    surface = reference.do_surface_run(**inputs)
    if independent:
        surface = dict(spread_rate=np.array([2.0], np.float32),
                       fireline_intensity=np.array([200], np.uint16),
                       heat_per_unit_area=np.ma.array([1000.0], mask=[False]))
    canopy = dict(canopy_base_height=4.0, canopy_height=30.0, canopy_bulk_density=.01)
    expected = model_reference(reference.do_crown_run, dict(surface_results=surface, **inputs, **canopy))
    for workers in (0, 1, 2, 4):
        actual = BehaveRun(FuelModels(), threads=workers).do_crown_run(surface_results=surface, **inputs, **canopy)
        assert actual.keys() == expected.keys()
        for field in expected:
            assert actual[field].shape == expected[field].shape, field
            assert actual[field].dtype == expected[field].dtype, field
            assert np.ma.isMaskedArray(actual[field]) == np.ma.isMaskedArray(expected[field]), field
            np.testing.assert_array_equal(np.ma.getmaskarray(actual[field]), np.ma.getmaskarray(expected[field]))
            np.testing.assert_allclose(np.ma.getdata(actual[field]), np.ma.getdata(expected[field]), rtol=1e-9, atol=1e-12, err_msg=field)


def test_copy_callback_exception_propagates_before_workers():
    def fail():
        raise RuntimeError("copy diagnostic failed")

    with pytest.raises(RuntimeError, match="copy diagnostic failed"):
        _bridge._native.component_batch(
            "surface.calculate_fire_length",
            {"forward_ros": np.arange(5.0), "backing_ros": np.asarray(1.0), "elapsed_min": np.asarray(3.0)},
            {}, {}, (5,), threads=4, _owned_callback=fail,
        )


@pytest.mark.parametrize("width", [0, 3])
def test_crown_three_axis_dependencies_preserve_typed_fields(width):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    inputs = _surface_inputs((2, 1, 1))
    inputs["wind_speed"] = np.full((1, width, 1), 8.0)
    inputs["wind_direction"] = 35.0
    inputs["slope"] = 12.0
    surface = dict(spread_rate=np.array([[[2.0]]], np.float32),
                   fireline_intensity=np.full((1, 1, 4), 200, np.uint16),
                   heat_per_unit_area=np.ma.array(np.full((2, 1, 1), 1000.0), mask=np.array([False, True]).reshape(2, 1, 1)))
    canopy = dict(canopy_base_height=np.full((1, 1, 4), 4.0), canopy_height=30.0,
                  canopy_bulk_density=np.full((1, 1, 4), .01), moisture_foliar=100.0)
    expected = model_reference(PythonRun(PythonFuels()).do_crown_run,
                               dict(surface_results=surface, **inputs, **canopy))
    for workers in (0, 1, 2, 4):
        actual = BehaveRun(FuelModels(), threads=workers).do_crown_run(surface_results=surface, **inputs, **canopy)
        for field in expected:
            assert actual[field].shape == expected[field].shape, field
            assert actual[field].dtype == expected[field].dtype, field
            assert np.ma.isMaskedArray(actual[field]) == np.ma.isMaskedArray(expected[field]), field
            np.testing.assert_array_equal(np.ma.getmaskarray(actual[field]), np.ma.getmaskarray(expected[field]))
            np.testing.assert_allclose(np.ma.getdata(actual[field]), np.ma.getdata(expected[field]), rtol=1e-9, atol=1e-12, err_msg=field)


def test_detached_progress_without_a_python_callback():
    import sys

    arrays = {"forward_ros": np.arange(120_000.0), "backing_ros": np.asarray(1.0), "elapsed_min": np.asarray(3.0)}
    go = threading.Event()
    progressed = threading.Event()

    def advance():
        go.wait()
        progressed.set()

    worker = threading.Thread(target=advance)
    worker.start()
    original_interval = sys.getswitchinterval()
    try:
        # Prevent a normal Python timeslice between notification and native entry.
        # The waiting thread can advance only when the native boundary releases the GIL.
        sys.setswitchinterval(60.0)
        go.set()
        result = _bridge._native.component_batch(
            "surface.calculate_fire_length", arrays, {}, {}, (120_000,), threads=4,
        )[""]
        advanced_during_call = progressed.is_set()
    finally:
        sys.setswitchinterval(original_interval)
        go.set()
        worker.join(10)
    assert not worker.is_alive()
    assert advanced_during_call
    np.testing.assert_array_equal(result, (arrays["forward_ros"] + 1.0) * 3.0)


def test_empty_crown_groups_share_the_pre_detach_input_snapshot(monkeypatch):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    inputs = _surface_inputs((2, 1))
    inputs["wind_speed"] = np.empty((1, 0))
    inputs["wind_direction"] = 35.0
    inputs["slope"] = 12.0
    surface = dict(spread_rate=np.array([2.0], np.float32),
                   fireline_intensity=np.array([200], np.uint16),
                   heat_per_unit_area=np.ma.array([1000.0], mask=[False]))
    canopy = dict(canopy_base_height=np.array([4.0]), canopy_height=30.0, canopy_bulk_density=.01)
    expected = model_reference(PythonRun(PythonFuels()).do_crown_run,
                               dict(surface_results=surface, **inputs, **canopy))
    original = _bridge._native.model_groups
    mutated = []

    def mutate():
        surface["fireline_intensity"].fill(777)
        surface["heat_per_unit_area"].mask[:] = True
        canopy["canopy_base_height"].fill(40.0)
        mutated.append(True)

    def record(*args, **kwargs):
        if args[0] == 'crown' and not mutated:
            kwargs["_owned_callback"] = mutate
        return original(*args, **kwargs)

    monkeypatch.setattr(_bridge._native, 'model_groups', record)
    actual = BehaveRun(FuelModels(), threads=4).do_crown_run(surface_results=surface, **inputs, **canopy)
    assert mutated == [True]
    for field in expected:
        assert actual[field].shape == expected[field].shape, field
        assert actual[field].dtype == expected[field].dtype, field
        np.testing.assert_array_equal(np.ma.getmaskarray(actual[field]), np.ma.getmaskarray(expected[field]))
        np.testing.assert_allclose(np.ma.getdata(actual[field]), np.ma.getdata(expected[field]), rtol=1e-9, atol=1e-12, err_msg=field)


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (2, 3), (2, 1, 3), (2048,)])
def test_fused_surface_and_crown_threads_match_python(shape):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    inputs = _surface_inputs(shape)
    reference = PythonRun(PythonFuels())
    expected_surface = reference.do_surface_run(**inputs)
    crown_inputs = dict(canopy_base_height=4.0, canopy_height=30.0, canopy_bulk_density=.01)
    expected_crown = reference.do_crown_run(surface_results=expected_surface, **inputs, **crown_inputs)
    serial = None
    for threads in (0, 1, 2, 4):
        model = BehaveRun(FuelModels(), threads=threads)
        surface = model.do_surface_run(**inputs)
        crown = model.do_crown_run(surface_results=expected_surface, **inputs, **crown_inputs)
        for actual, expected in ((surface, expected_surface), (crown, expected_crown)):
            assert actual.keys() == expected.keys()
            for key in actual:
                assert actual[key].shape == expected[key].shape
                assert actual[key].dtype == expected[key].dtype
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-12)
        if serial is None:
            serial = (surface, crown)
        else:
            for actual, expected in zip((surface, crown), serial):
                for key in actual:
                    np.testing.assert_array_equal(actual[key].view(np.uint8), expected[key].view(np.uint8))


def test_instance_threads_reach_native_boundary(monkeypatch):
    original = _bridge._native.component_batch
    requests = []

    def record(*args, **kwargs):
        requests.append(kwargs.get("threads", 1))
        return original(*args, **kwargs)

    monkeypatch.setattr(_bridge._native, "component_batch", record)
    for workers in (0, 1, 2, 4):
        model = BehaveRun(FuelModels(), threads=workers)
        requests.clear()
        model.calculate_fire_length(np.arange(9.0), 1.0, 0, 3.0, 0)
        assert requests and all(value == workers for value in requests)
    requests.clear()
    BehaveRun.calculate_fire_length(np.arange(9.0), 1.0, 0, 3.0, 0)
    assert requests and all(value == 1 for value in requests)


@pytest.mark.parametrize("workers,error", [(-1, ValueError), (1.5, TypeError), ("2", TypeError), (None, TypeError)])
def test_invalid_threads(workers, error):
    with pytest.raises(error):
        BehaveRun(FuelModels(), threads=workers)


def test_large_positive_requests_are_capped_before_unsigned_conversion():
    model = BehaveRun(FuelModels(), threads=2 ** 100)
    result = model.calculate_fire_length(np.arange(3.0), 1.0, 0, 3.0, 0)
    np.testing.assert_array_equal(result, (np.arange(3.0) + 1.0) * 3.0)


@pytest.mark.parametrize("workers,error", [(-1, ValueError), (1.5, TypeError), ("4", TypeError)])
def test_low_level_threads_validate_before_unsigned_conversion(workers, error):
    with pytest.raises(error):
        _bridge._native.component_batch(
            "surface.calculate_fire_length",
            {"forward_ros": np.arange(3.0), "backing_ros": np.asarray(1.0), "elapsed_min": np.asarray(3.0)},
            {}, {}, (3,), threads=workers,
        )


def test_order_empty_short_and_independent_instances():
    values = np.arange(31.0)[::-2]
    models = [BehaveRun(FuelModels(), threads=value) for value in (0, 1, 2, 4)]
    for model in models * 2:
        for data in (values, values[:1], values[:0]):
            result = model.calculate_fire_length(data, 1.0, 0, 3.0, 0)
            np.testing.assert_array_equal(result, (data + 1.0) * 3.0)
            assert not np.shares_memory(result, data)


def test_owned_inputs_are_copied_before_callback_mutation():
    """Owned copies preserve results after synchronized caller mutation."""
    source = np.arange(120_000.0)
    original = source.copy()
    copied = threading.Event()
    finished = threading.Event()
    mutations = []
    errors = []

    def mutate():
        try:
            if not copied.wait(10):
                raise TimeoutError("Owned-copy notification was not received")
            source.fill(-777.0)
            mutations.append(True)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    def after_copy():
        copied.set()
        assert finished.wait(10), "Mutation worker did not finish after owned copies"
        assert not errors, f"Mutation worker failed: {errors!r}"

    worker = threading.Thread(target=mutate)
    worker.start()
    try:
        result = _bridge._native.component_batch(
            "surface.calculate_fire_length",
            {"forward_ros": source, "backing_ros": np.asarray(1.0), "elapsed_min": np.asarray(3.0)},
            {}, {}, source.shape, threads=4, _owned_callback=after_copy,
        )[""]
    finally:
        copied.set()
        worker.join(10)
        assert not worker.is_alive()
    assert not errors, f"Mutation worker failed: {errors!r}"
    assert mutations == [True]
    np.testing.assert_array_equal(result, (original + 1.0) * 3.0)
    np.testing.assert_array_equal(source, -777.0)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 1, 3)])
def test_scalar_moisture_shape_errors_match_python(shape):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    inputs = _surface_inputs(shape)
    for field in ("m1h", "m10h", "m100h", "mlh", "mlw"):
        inputs[field] = .06
    for model in (PythonRun(PythonFuels()), BehaveRun(FuelModels(), threads=4)):
        with pytest.raises(ValueError):
            model.do_surface_run(**inputs)


def test_scoped_threads_restore_after_errors_and_concurrent_instances(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from behave_fire.rust._threads import _THREADS

    original = _bridge._native.component_batch
    barrier = threading.Barrier(2)
    seen = []
    lock = threading.Lock()

    def record(*args, **kwargs):
        with lock:
            seen.append((threading.get_ident(), kwargs.get("threads")))
        return original(*args, **kwargs)

    monkeypatch.setattr(_bridge._native, "component_batch", record)

    def run(workers):
        model = BehaveRun(FuelModels(), threads=workers)
        barrier.wait(timeout=10)
        for _ in range(3):
            result = model.calculate_fire_length(np.arange(29.0), 1.0, 0, 3.0, 0)
            np.testing.assert_array_equal(result, (np.arange(29.0) + 1.0) * 3.0)
        return threading.get_ident(), workers

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = dict(pool.map(run, (1, 4)))
    assert all(identities[identity] == workers for identity, workers in seen)
    assert _THREADS.get() == 1
    with pytest.raises(IndexError):
        BehaveRun(FuelModels(), threads=4).calculate_fire_length(1.0, 1.0, 999, 3.0, 0)
    assert _THREADS.get() == 1


def test_typed_supplied_crown_outputs_are_thread_independent():
    inputs = _surface_inputs((2048,))
    model = BehaveRun(FuelModels())
    surface = model.do_surface_run(**inputs)
    surface["spread_rate"] = surface["spread_rate"].astype(np.float32)
    surface["fireline_intensity"] = np.arange(2048, dtype=np.uint16) + 200
    surface["heat_per_unit_area"] = np.ma.array(surface["heat_per_unit_area"], mask=np.arange(2048) % 3 == 0)
    expected = None
    for threads in (1, 0, 2, 4):
        actual = BehaveRun(FuelModels(), threads=threads).do_crown_run(
            surface_results=surface, **inputs, canopy_base_height=4.0,
            canopy_height=30.0, canopy_bulk_density=.01,
        )
        if expected is None:
            expected = actual
        else:
            for field in expected:
                assert np.ma.isMaskedArray(actual[field]) == np.ma.isMaskedArray(expected[field])
                assert actual[field].dtype == expected[field].dtype
                assert actual[field].shape == expected[field].shape
                np.testing.assert_array_equal(np.ma.getmaskarray(actual[field]), np.ma.getmaskarray(expected[field]))
                np.testing.assert_array_equal(np.ma.getdata(actual[field]).view(np.uint8), np.ma.getdata(expected[field]).view(np.uint8))


def test_unbound_instance_calls_honor_receiver_threads(monkeypatch):
    original = _bridge._native.model_batch
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs.get("threads", 1))
        return original(*args, **kwargs)

    monkeypatch.setattr(_bridge._native, 'model_batch', record)
    model = BehaveRun(FuelModels(), threads=4)
    BehaveRun.do_surface_run(model, **_surface_inputs((2,)))
    assert seen and all(value == 4 for value in seen)
