"""Prefer a backend and preflight verified complex calls on owned input snapshots."""

import copy
import importlib
import importlib.util
import threading
import warnings
from typing import Optional, Union

import numpy as np

from .behave import BehaveRun as PythonBehaveRun
from ._backend_contract import BINDING_API_VERSION, REQUIRED_CAPABILITIES
from .components.fuel_models import FuelModels
from ._auto_routing import capture_call, choose_backend, snapshot_registry

_API_VERSION = BINDING_API_VERSION
_INSTALL = "Rust backend unavailable; install a compatible native behave-fire wheel or rebuild the extension."
_PROBE_LOCK = threading.RLock()
_PROBE_RESULT = None
_WARNING_EMITTED = False
_COMPLEX_WARNING_EMITTED = False


def _probe():
    """Cache one discovery outcome without consuming a fallback warning.

    :return: Pair of usability and the original absence or failure cause.
    """
    global _PROBE_RESULT
    with _PROBE_LOCK:
        if _PROBE_RESULT is not None:
            return _PROBE_RESULT
        spec = importlib.util.find_spec("behave_fire._rust")
        if spec is None:
            _PROBE_RESULT = ("absent", ModuleNotFoundError("behave_fire._rust is absent"))
            return _PROBE_RESULT
        try:
            native = importlib.import_module("behave_fire._rust")
            version = getattr(native, "__binding_api_version__", None)
            if version != _API_VERSION:
                raise ImportError(f"incompatible binding API version {version!r}; expected {_API_VERSION}")
            if getattr(native, "__backend_complete__", False) is not True:
                raise ImportError("incomplete native backend capabilities")
            capabilities = getattr(native, "__capabilities__", None)
            if not isinstance(capabilities, (tuple, list, set, frozenset)) or any(
                    not isinstance(operation, str) for operation in capabilities
            ):
                raise ImportError("invalid or missing native backend capabilities inventory")
            missing = REQUIRED_CAPABILITIES.difference(capabilities)
            if missing:
                raise ImportError(
                    f"incomplete native backend capabilities: missing {len(missing)} operations; "
                    f"first missing operation: {min(missing)}"
                )
        except Exception as error:
            _PROBE_RESULT = ("unusable", error)
        else:
            _PROBE_RESULT = ("usable", None)
        return _PROBE_RESULT


def _select(backend, stacklevel):
    """Resolve a selector and warn at the public caller when necessary.

    :param backend: Requested backend name.
    :param stacklevel: Warning stack depth of the public caller.
    :return: The explicit namespace name selected for this request.
    """
    global _WARNING_EMITTED
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be 'auto', 'python', or 'rust'")
    if backend == "python":
        return "python"
    status, cause = _probe()
    if status == "usable":
        return "rust"
    if backend == "rust":
        raise ImportError(f"{_INSTALL} Cause: {cause}") from cause
    with _PROBE_LOCK:
        if not _WARNING_EMITTED:
            reason = "unavailable" if status == "absent" else "unusable"
            warnings.warn(
                f"Rust backend {reason} ({type(cause).__name__}: {cause}); using Python. {_INSTALL}",
                BehaveBackendWarning,
                stacklevel=stacklevel,
            )
            _WARNING_EMITTED = True
    return "python"


def _warn_complex(operation, stacklevel):
    """Warn once before the complete complex operation is delegated to Python.

    :param operation: Public operation requiring verified Python complex arithmetic.
    :param stacklevel: Stack depth of the external caller.
    :return: None; a raised warning prevents delegation and does not consume the warning.
    """
    global _COMPLEX_WARNING_EMITTED
    with _PROBE_LOCK:
        if not _COMPLEX_WARNING_EMITTED:
            warnings.warn(f'Complex input to {operation} requires the Python backend; using Python.',
                          BehaveBackendWarning, stacklevel=stacklevel)
            _COMPLEX_WARNING_EMITTED = True


class _LegacyStaticMethod:
    """Keep historical class calls on Python and instance calls on their delegate."""

    def __get__(self, instance, owner):
        """Bind an explicit forwarding method or expose the historical static API.

        :param instance: Automatic model, or None for class access.
        :param owner: Owning automatic class.
        :return: Callable with the historical discoverable signature.
        """
        if instance is None:
            return getattr(PythonBehaveRun, self._method.__name__)
        return self._method.__get__(instance, owner)

    def __init__(self, method):
        """Store the explicitly declared instance forwarding method.

        :param method: Forwarding method with the full public signature.
        :return: None; initializes this descriptor.
        """
        self._method = method


class BehaveBackendWarning(RuntimeWarning):
    """Automatic selection uses Python because native execution is unavailable."""


class BehaveRun:
    """Prefer the selected backend and route proven complex calls before coercion.

    Use the explicit Python class for historical subclassing and class identity.
    Historical static class calls retain their Python implementation; instance
    calls report the delegate that completed their most recent operation.
    """

    def __copy__(self):
        """Copy the wrapper and shallow-copy its delegate as the reference does.

        :return: A separate wrapper retaining shared shallow numerical state.
        """
        result = type(self).__new__(type(self))
        state = self.__getstate__()
        copied = {}
        for name in ('delegate', 'rust_delegate', 'python_delegate'):
            delegate = state[name]
            if delegate is not None:
                if id(delegate) not in copied:
                    copied[id(delegate)] = copy.copy(delegate)
                state[name] = copied[id(delegate)]
        result.__setstate__(state)
        return result

    def __deepcopy__(self, memo):
        """Copy the selected delegate and numerical snapshot independently.

        :param memo: Deepcopy identity map.
        :return: A model with independent registry metadata and cached arrays.
        """
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return result

    def __getstate__(self):
        """Persist the actual delegate snapshot rather than rebuilding its registry.

        :return: Versioned values containing backend, threads, and delegate state.
        """
        return {"version": 2, "requested": self._requested_backend, "preferred": self._preferred_backend,
                "backend": self._backend_in_use, "threads": self._threads, "delegate": self._delegate,
                "rust_delegate": self._rust_delegate, "python_delegate": self._python_delegate,
                "registry_snapshot": self._registry_snapshot}

    def __init__(self, fuel_models, *, backend="auto", threads=1):
        """Choose a backend and freeze its fuel lookup at construction.

        :param fuel_models: Python registry or native registry with to_python().
        :param backend: One of auto, python, or rust.
        :param threads: Native worker count; ignored by the Python route.
        :return: None; initializes the selected calculation delegate.
        """
        selected = _select(backend, 3)
        self._requested_backend = backend
        self._preferred_backend = selected
        self._registry_snapshot = None if backend == 'python' else snapshot_registry(fuel_models)
        if backend == 'auto':
            fuel_models = self._registry_snapshot.to_python()
        namespace = importlib.import_module(f"behave_fire.{selected}")
        if selected == "python":
            if not isinstance(fuel_models, FuelModels) and callable(getattr(fuel_models, "to_python", None)):
                fuel_models = fuel_models.to_python()
            delegate = namespace.BehaveRun(fuel_models)
        else:
            delegate = namespace.BehaveRun(fuel_models, threads=threads)
        self._backend_in_use = selected
        self._threads = threads
        self._delegate = delegate
        self._rust_delegate = delegate if selected == 'rust' else None
        self._python_delegate = delegate if selected == 'python' else None

    def __setstate__(self, state):
        """Restore the chosen backend without recomputing or selecting another one.

        :param state: Versioned serialized automatic model values.
        :return: None; restores the persisted numerical snapshot.
        """
        if state.get("version") != 2:
            raise ValueError("unsupported automatic BehaveRun pickle state version")
        selected = state["backend"]
        if selected not in ("python", "rust"):
            raise ValueError("invalid serialized backend")
        if state.get('preferred') not in ('python', 'rust') or state.get('requested') not in ('auto', 'python', 'rust'):
            raise ValueError('invalid serialized backend policy')
        if state.get('rust_delegate') is not None:
            importlib.import_module("behave_fire.rust")
        self._requested_backend = state['requested']
        self._preferred_backend = state['preferred']
        self._registry_snapshot = state['registry_snapshot']
        self._rust_delegate = state['rust_delegate']
        self._python_delegate = state['python_delegate']
        self._backend_in_use = selected
        self._threads = state["threads"]
        self._delegate = state["delegate"]

    def _invoke(self, operation, *args, **kwargs):
        """Execute one selected operation without retrying scientific failures.

        :param operation: Existing public model method name.
        :param args: Original positional calculation inputs.
        :param kwargs: Original keyword calculation inputs.
        :return: Complete delegate result; only successful calls update actual backend reporting.
        """
        if self._requested_backend == 'python':
            return getattr(self._python_delegate, operation)(*args, **kwargs)
        call = capture_call('model.' + operation, getattr(PythonBehaveRun, operation), args, kwargs)
        if self._requested_backend == 'rust':
            if call.complex_sites:
                raise TypeError('The explicit Rust backend accepts real physical inputs; select Python for complex data.')
            selected = 'rust'
        else:
            selected = choose_backend(call, self._preferred_backend)
        if selected == 'python' and self._preferred_backend == 'rust':
            _warn_complex(operation, 4)
        delegate = self._rust_delegate if selected == 'rust' else self._python_delegate
        if delegate is None:
            namespace = importlib.import_module('behave_fire.python')
            delegate = namespace.BehaveRun(self._registry_snapshot.to_python())
            self._python_delegate = delegate
        result = call.invoke(getattr(delegate, operation))
        self._delegate = delegate
        self._backend_in_use = selected
        return result

    @property
    def backend_in_use(self):
        """Return the backend that completed the most recent calculation.

        :return: Actual public backend name, python or rust; initially the construction backend.
        """
        return self._backend_in_use


    def calculate_crown_scorch_mortality(
            self,
            scorch_height: Union[float, np.ndarray],
            tree_height: Union[float, np.ndarray],
            crown_ratio: Union[float, np.ndarray],
            dbh: Union[float, np.ndarray],
            equation_number_grid: Union[int, np.ndarray],
            scorch_height_units: int = 0,
            tree_height_units: int = 0,
            crown_ratio_units: int = 0,
            dbh_units: int = 1,
            out_units: Optional[dict] = None,
    ) -> dict:
        """Forward calculate_crown_scorch_mortality to the selected backend.

        :param scorch_height: Passed unchanged to the selected backend.
        :param tree_height: Passed unchanged to the selected backend.
        :param crown_ratio: Passed unchanged to the selected backend.
        :param dbh: Passed unchanged to the selected backend.
        :param equation_number_grid: Passed unchanged to the selected backend.
        :param scorch_height_units: Passed unchanged to the selected backend.
        :param tree_height_units: Passed unchanged to the selected backend.
        :param crown_ratio_units: Passed unchanged to the selected backend.
        :param dbh_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_crown_scorch_mortality",
            scorch_height=scorch_height,
            tree_height=tree_height,
            crown_ratio=crown_ratio,
            dbh=dbh,
            equation_number_grid=equation_number_grid,
            scorch_height_units=scorch_height_units,
            tree_height_units=tree_height_units,
            crown_ratio_units=crown_ratio_units,
            dbh_units=dbh_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_fire_area(
            self,
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            is_crown: bool = False,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_fire_area to the selected backend.

        :param forward_ros: Passed unchanged to the selected backend.
        :param backing_ros: Passed unchanged to the selected backend.
        :param ros_units: Passed unchanged to the selected backend.
        :param lwr: Passed unchanged to the selected backend.
        :param elapsed_time: Passed unchanged to the selected backend.
        :param elapsed_time_units: Passed unchanged to the selected backend.
        :param is_crown: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_fire_area",
            forward_ros=forward_ros,
            backing_ros=backing_ros,
            ros_units=ros_units,
            lwr=lwr,
            elapsed_time=elapsed_time,
            elapsed_time_units=elapsed_time_units,
            is_crown=is_crown,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_fire_length(
            self,
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            elapsed_time: float,
            elapsed_time_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_fire_length to the selected backend.

        :param forward_ros: Passed unchanged to the selected backend.
        :param backing_ros: Passed unchanged to the selected backend.
        :param ros_units: Passed unchanged to the selected backend.
        :param elapsed_time: Passed unchanged to the selected backend.
        :param elapsed_time_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_fire_length",
            forward_ros=forward_ros,
            backing_ros=backing_ros,
            ros_units=ros_units,
            elapsed_time=elapsed_time,
            elapsed_time_units=elapsed_time_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_fire_perimeter(
            self,
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            is_crown: bool = False,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_fire_perimeter to the selected backend.

        :param forward_ros: Passed unchanged to the selected backend.
        :param backing_ros: Passed unchanged to the selected backend.
        :param ros_units: Passed unchanged to the selected backend.
        :param lwr: Passed unchanged to the selected backend.
        :param elapsed_time: Passed unchanged to the selected backend.
        :param elapsed_time_units: Passed unchanged to the selected backend.
        :param is_crown: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_fire_perimeter",
            forward_ros=forward_ros,
            backing_ros=backing_ros,
            ros_units=ros_units,
            lwr=lwr,
            elapsed_time=elapsed_time,
            elapsed_time_units=elapsed_time_units,
            is_crown=is_crown,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_fire_width(
            self,
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_fire_width to the selected backend.

        :param forward_ros: Passed unchanged to the selected backend.
        :param backing_ros: Passed unchanged to the selected backend.
        :param ros_units: Passed unchanged to the selected backend.
        :param lwr: Passed unchanged to the selected backend.
        :param elapsed_time: Passed unchanged to the selected backend.
        :param elapsed_time_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_fire_width",
            forward_ros=forward_ros,
            backing_ros=backing_ros,
            ros_units=ros_units,
            lwr=lwr,
            elapsed_time=elapsed_time,
            elapsed_time_units=elapsed_time_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_scorch_height(
            self,
            fireline_intensity: Union[float, np.ndarray],
            fireline_intensity_units: int,
            midflame_wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            air_temperature: Union[float, np.ndarray],
            temperature_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_scorch_height to the selected backend.

        :param fireline_intensity: Passed unchanged to the selected backend.
        :param fireline_intensity_units: Passed unchanged to the selected backend.
        :param midflame_wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param air_temperature: Passed unchanged to the selected backend.
        :param temperature_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_scorch_height",
            fireline_intensity=fireline_intensity,
            fireline_intensity_units=fireline_intensity_units,
            midflame_wind_speed=midflame_wind_speed,
            wind_speed_units=wind_speed_units,
            air_temperature=air_temperature,
            temperature_units=temperature_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_spotting_from_burning_pile(
            self,
            flame_height: Union[float, np.ndarray],
            flame_height_units: int,
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            cover_height: Union[float, np.ndarray],
            cover_height_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_spotting_from_burning_pile to the selected backend.

        :param flame_height: Passed unchanged to the selected backend.
        :param flame_height_units: Passed unchanged to the selected backend.
        :param wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param cover_height: Passed unchanged to the selected backend.
        :param cover_height_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_spotting_from_burning_pile",
            flame_height=flame_height,
            flame_height_units=flame_height_units,
            wind_speed=wind_speed,
            wind_speed_units=wind_speed_units,
            cover_height=cover_height,
            cover_height_units=cover_height_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_spotting_from_surface_fire(
            self,
            flame_length: Union[float, np.ndarray],
            flame_length_units: int,
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            cover_height: Union[float, np.ndarray],
            cover_height_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_spotting_from_surface_fire to the selected backend.

        :param flame_length: Passed unchanged to the selected backend.
        :param flame_length_units: Passed unchanged to the selected backend.
        :param wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param cover_height: Passed unchanged to the selected backend.
        :param cover_height_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_spotting_from_surface_fire",
            flame_length=flame_length,
            flame_length_units=flame_length_units,
            wind_speed=wind_speed,
            wind_speed_units=wind_speed_units,
            cover_height=cover_height,
            cover_height_units=cover_height_units,
            out_units=out_units,
        )

    @_LegacyStaticMethod
    def calculate_spotting_from_torching_trees(
            self,
            dbh: Union[float, np.ndarray],
            dbh_units: int,
            height: Union[float, np.ndarray],
            height_units: int,
            count: Union[int, float, np.ndarray],
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            cover_height: Union[float, np.ndarray],
            cover_height_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """Forward calculate_spotting_from_torching_trees to the selected backend.

        :param dbh: Passed unchanged to the selected backend.
        :param dbh_units: Passed unchanged to the selected backend.
        :param height: Passed unchanged to the selected backend.
        :param height_units: Passed unchanged to the selected backend.
        :param count: Passed unchanged to the selected backend.
        :param wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param cover_height: Passed unchanged to the selected backend.
        :param cover_height_units: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("calculate_spotting_from_torching_trees",
            dbh=dbh,
            dbh_units=dbh_units,
            height=height,
            height_units=height_units,
            count=count,
            wind_speed=wind_speed,
            wind_speed_units=wind_speed_units,
            cover_height=cover_height,
            cover_height_units=cover_height_units,
            out_units=out_units,
        )

    def do_crown_run(
            self,
            surface_results: dict,
            fuel_model_grid: Union[int, np.ndarray],
            m1h: Union[float, np.ndarray],
            m10h: Union[float, np.ndarray],
            m100h: Union[float, np.ndarray],
            mlh: Union[float, np.ndarray],
            mlw: Union[float, np.ndarray],
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            wind_direction: Union[float, np.ndarray],
            wind_orientation_mode: str,
            slope: Union[float, np.ndarray],
            slope_units: int = 0,
            moisture_units: int = 0,
            aspect: Union[float, np.ndarray] = 0.0,
            canopy_base_height: Union[float, np.ndarray] = 0.0,
            canopy_base_height_units: int = 0,
            canopy_height: Union[float, np.ndarray] = 0.0,
            canopy_height_units: int = 0,
            canopy_bulk_density: Union[float, np.ndarray] = 0.0,
            canopy_bulk_density_units: int = 0,
            moisture_foliar: Union[int, float, np.ndarray] = 100.0,
            out_units: Optional[dict] = None,
    ) -> dict:
        """Forward do_crown_run to the selected backend.

        :param surface_results: Passed unchanged to the selected backend.
        :param fuel_model_grid: Passed unchanged to the selected backend.
        :param m1h: Passed unchanged to the selected backend.
        :param m10h: Passed unchanged to the selected backend.
        :param m100h: Passed unchanged to the selected backend.
        :param mlh: Passed unchanged to the selected backend.
        :param mlw: Passed unchanged to the selected backend.
        :param wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param wind_direction: Passed unchanged to the selected backend.
        :param wind_orientation_mode: Passed unchanged to the selected backend.
        :param slope: Passed unchanged to the selected backend.
        :param slope_units: Passed unchanged to the selected backend.
        :param moisture_units: Passed unchanged to the selected backend.
        :param aspect: Passed unchanged to the selected backend.
        :param canopy_base_height: Passed unchanged to the selected backend.
        :param canopy_base_height_units: Passed unchanged to the selected backend.
        :param canopy_height: Passed unchanged to the selected backend.
        :param canopy_height_units: Passed unchanged to the selected backend.
        :param canopy_bulk_density: Passed unchanged to the selected backend.
        :param canopy_bulk_density_units: Passed unchanged to the selected backend.
        :param moisture_foliar: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("do_crown_run",
            surface_results=surface_results,
            fuel_model_grid=fuel_model_grid,
            m1h=m1h,
            m10h=m10h,
            m100h=m100h,
            mlh=mlh,
            mlw=mlw,
            wind_speed=wind_speed,
            wind_speed_units=wind_speed_units,
            wind_direction=wind_direction,
            wind_orientation_mode=wind_orientation_mode,
            slope=slope,
            slope_units=slope_units,
            moisture_units=moisture_units,
            aspect=aspect,
            canopy_base_height=canopy_base_height,
            canopy_base_height_units=canopy_base_height_units,
            canopy_height=canopy_height,
            canopy_height_units=canopy_height_units,
            canopy_bulk_density=canopy_bulk_density,
            canopy_bulk_density_units=canopy_bulk_density_units,
            moisture_foliar=moisture_foliar,
            out_units=out_units,
        )

    def do_surface_run(
            self,
            fuel_model_grid: Union[int, np.ndarray],
            m1h: Union[float, np.ndarray],
            m10h: Union[float, np.ndarray],
            m100h: Union[float, np.ndarray],
            mlh: Union[float, np.ndarray],
            mlw: Union[float, np.ndarray],
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            wind_direction: Union[float, np.ndarray],
            wind_orientation_mode: str,
            slope: Union[float, np.ndarray],
            slope_units: int = 0,
            moisture_units: int = 0,
            aspect: Union[float, np.ndarray] = 0.0,
            canopy_cover: Union[float, np.ndarray] = 0.0,
            canopy_cover_units: int = 0,
            canopy_height: Union[float, np.ndarray] = 0.0,
            canopy_height_units: int = 0,
            canopy_base_height: Union[float, np.ndarray] = 0.0,
            canopy_base_height_units: int = 0,
            crown_ratio: Union[float, np.ndarray] = 0.0,
            crown_ratio_units: Optional[int] = 0,
            wind_height_mode: str = 'TwentyFoot',
            waf_method: str = 'UseCrownRatio',
            user_waf: Union[float, np.ndarray, None] = None,
            out_units: Optional[dict] = None,
    ) -> dict:
        """Forward do_surface_run to the selected backend.

        :param fuel_model_grid: Passed unchanged to the selected backend.
        :param m1h: Passed unchanged to the selected backend.
        :param m10h: Passed unchanged to the selected backend.
        :param m100h: Passed unchanged to the selected backend.
        :param mlh: Passed unchanged to the selected backend.
        :param mlw: Passed unchanged to the selected backend.
        :param wind_speed: Passed unchanged to the selected backend.
        :param wind_speed_units: Passed unchanged to the selected backend.
        :param wind_direction: Passed unchanged to the selected backend.
        :param wind_orientation_mode: Passed unchanged to the selected backend.
        :param slope: Passed unchanged to the selected backend.
        :param slope_units: Passed unchanged to the selected backend.
        :param moisture_units: Passed unchanged to the selected backend.
        :param aspect: Passed unchanged to the selected backend.
        :param canopy_cover: Passed unchanged to the selected backend.
        :param canopy_cover_units: Passed unchanged to the selected backend.
        :param canopy_height: Passed unchanged to the selected backend.
        :param canopy_height_units: Passed unchanged to the selected backend.
        :param canopy_base_height: Passed unchanged to the selected backend.
        :param canopy_base_height_units: Passed unchanged to the selected backend.
        :param crown_ratio: Passed unchanged to the selected backend.
        :param crown_ratio_units: Passed unchanged to the selected backend.
        :param wind_height_mode: Passed unchanged to the selected backend.
        :param waf_method: Passed unchanged to the selected backend.
        :param user_waf: Passed unchanged to the selected backend.
        :param out_units: Passed unchanged to the selected backend.
        :return: The selected backend result, preserving its array semantics.
        """
        return self._invoke("do_surface_run",
            fuel_model_grid=fuel_model_grid,
            m1h=m1h,
            m10h=m10h,
            m100h=m100h,
            mlh=mlh,
            mlw=mlw,
            wind_speed=wind_speed,
            wind_speed_units=wind_speed_units,
            wind_direction=wind_direction,
            wind_orientation_mode=wind_orientation_mode,
            slope=slope,
            slope_units=slope_units,
            moisture_units=moisture_units,
            aspect=aspect,
            canopy_cover=canopy_cover,
            canopy_cover_units=canopy_cover_units,
            canopy_height=canopy_height,
            canopy_height_units=canopy_height_units,
            canopy_base_height=canopy_base_height,
            canopy_base_height_units=canopy_base_height_units,
            crown_ratio=crown_ratio,
            crown_ratio_units=crown_ratio_units,
            wind_height_mode=wind_height_mode,
            waf_method=waf_method,
            user_waf=user_waf,
            out_units=out_units,
        )


    @property
    def preferred_backend(self):
        """Return the usable backend captured for ordinary real calculations.

        :return: Cached preferred backend name, independent of complex-call fallback.
        """
        return self._preferred_backend


def available_backends() -> list[str]:
    """List usable complete implementations without consuming fallback warnings.

    :return: Python, followed by Rust when its extension is usable and complete.
    """
    return ["python", "rust"] if _probe()[0] == "usable" else ["python"]


def get_backend(backend="auto"):
    """Return a routing namespace for auto or an explicit requested namespace.

    :param backend: One of auto, python, or rust.
    :return: Signature-preserving automatic namespace, or the explicit backend module.
    """
    selected = _select(backend, 3)
    if backend == 'auto':
        from ._auto_namespace import AutoNamespace
        return AutoNamespace(selected)
    return importlib.import_module(f"behave_fire.{selected}")
