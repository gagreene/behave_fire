"""Own original call values and select only verified complex Python counterparts."""
from dataclasses import dataclass
import copy
import inspect

import numpy as np
from ._ownership import _owned_inputs


# Entries are parameter/layout proofs, never a catch-and-retry permission.
# Further operations enter this table only with complete, nonzero-imaginary tests.
_COMPLEX_CAPABILITIES = {
    'surface.calculate_fire_area': {'elapsed_min': frozenset({'scalar', 'numpy-scalar', 'array', 'masked', 'boxed'})},
    'crown.calculate_crown_heat_and_intensity': {'surface_hpua': frozenset({'array', 'masked'})},
    'model.do_crown_run': {'surface_results.heat_per_unit_area': frozenset({'array', 'masked'})},
}


def _complex_sites(value, path):
    """Locate complex storage without invoking custom numeric conversions.

    :param value: Original owned value or container.
    :param path: Argument and nested field path.
    :return: Tuple of parameter paths and verified-layout identifiers.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'c':
            if type(value) not in (np.ndarray, np.ma.MaskedArray):
                return ((path, 'opaque'),)
            return ((path, 'masked' if np.ma.isMaskedArray(value) else 'array'),)
        if value.dtype.kind == 'O':
            items = np.asarray(value).reshape(-1)
            found = any(isinstance(item, (complex, np.complexfloating)) for item in items)
            if found:
                ordinary = all(type(item) in (bool, int, float, complex) or (
                    isinstance(item, np.generic) and type(item) is item.dtype.type) for item in items)
                layout = 'boxed' if ordinary and not np.ma.isMaskedArray(value) else 'opaque'
                return ((path, layout),)
        return ()
    if isinstance(value, np.complexfloating):
        ordinary = any(type(value) is kind for kind in (np.complex64, np.complex128, np.clongdouble))
        return ((path, 'numpy-scalar' if ordinary else 'opaque'),)
    if isinstance(value, complex):
        return ((path, 'scalar' if type(value) is complex else 'opaque'),)
    if type(value) is dict:
        return tuple(site for name, item in value.items()
                     for site in _complex_sites(item, f'{path}.{name}'))
    if type(value) in (list, tuple):
        if any(_complex_sites(item, path) for item in value):
            return ((path, 'sequence'),)
    return ()


def capture_call(operation, function, args, kwargs):
    """Bind once and own the original values before any backend-specific work.

    :param operation: Canonical operation used by the verified compatibility table.
    :param function: Python counterpart providing the public signature.
    :param args: Original positional arguments, without an unbound self.
    :param kwargs: Original keyword arguments.
    :return: An owned call with unchanged dtype, masks, values, and logical shapes.
    """
    signature = inspect.signature(function)
    if tuple(signature.parameters)[:1] == ('self',):
        signature = signature.replace(parameters=tuple(signature.parameters.values())[1:])
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    original_sites = tuple(site for name, value in bound.arguments.items() for site in _complex_sites(value, name))
    with _owned_inputs() as owner:
        arguments = owner.snapshot(dict(bound.arguments))
        sites = tuple(site for name, value in arguments.items() for site in _complex_sites(value, name))
        if any(layout == 'opaque' for _, layout in original_sites):
            sites = original_sites
        return OwnedCall(operation, signature, arguments, sites, owner)


def choose_backend(call, preferred):
    """Preflight complex support without executing or replaying either delegate.

    :param call: Owned original values captured for this complete operation.
    :param preferred: Already selected usable backend, python or rust.
    :return: Python unchanged when already selected; otherwise Rust or a proven complex fallback.
    """
    if preferred not in ('python', 'rust'):
        raise ValueError('preferred backend must be python or rust')
    if preferred == 'python' or not call.complex_sites:
        return preferred
    supported = _COMPLEX_CAPABILITIES.get(call.operation, {})
    for path, layout in call.complex_sites:
        if layout not in supported.get(path, ()):
            raise TypeError(f'Automatic complex support is not verified for {call.operation}: {path} ({layout}); '
                            'select backend="python" explicitly for unsupported arithmetic.')
    if call.operation == 'model.do_crown_run' and call.arguments.get('out_units'):
        raise TypeError('Complex Crown output-unit conversions are not verified; select backend="python" explicitly.')
    return 'python'


def snapshot_registry(fuel_models):
    """Own a backend-neutral registry before constructing a native delegate.

    :param fuel_models: Original Python registry or a registry with to_python().
    :return: Independent neutral snapshot; unproven complex registry arithmetic raises TypeError.
    """
    from .components.fuel_models import FuelModels

    if not isinstance(fuel_models, FuelModels) and callable(getattr(fuel_models, 'to_python', None)):
        fuel_models = fuel_models.to_python()
    if not isinstance(fuel_models, FuelModels):
        raise TypeError('fuel_models must be a FuelModels registry or provide to_python()')
    physical = ('fuel_bed_depth', 'moisture_of_extinction_dead', 'heat_of_combustion_dead',
                'heat_of_combustion_live', 'dead_1h', 'dead_10h', 'dead_100h', 'live_herb',
                'live_woody', 'savr_1h', 'savr_live_herb', 'savr_live_woody', 'is_dynamic', 'is_defined')
    for number, record in fuel_models.fuel_models_.items():
        for name in physical:
            if _complex_sites(record.get(name), name):
                raise TypeError(f'Complex fuel registry support is not verified: model {number}, {name}; '
                                'select backend="python" explicitly.')
    return RegistrySnapshot(copy.deepcopy(fuel_models))


@dataclass(frozen=True)
class OwnedCall:
    """One signature and original input snapshot shared by preflight and execution."""

    operation: str
    signature: inspect.Signature
    arguments: dict
    complex_sites: tuple
    _owner: object

    def invoke(self, function):
        """Execute one complete selected delegate on this exact owned snapshot.

        :param function: Chosen Python or Rust callable with the public signature.
        :return: Its result unchanged; delegate exceptions propagate without replay.
        """
        bound = inspect.BoundArguments(self.signature, self.arguments)
        with _owned_inputs(self._owner) as owner:
            return owner.remember(function(*bound.args, **bound.kwargs))


@dataclass(frozen=True)
class RegistrySnapshot:
    """Private backend-neutral registry owner; delegates receive independent copies."""

    _registry: object

    def to_python(self):
        """Materialize one delegate registry without exposing the frozen owner.

        :return: Independent Python FuelModels preserving the captured values and metadata.
        """
        return copy.deepcopy(self._registry)
