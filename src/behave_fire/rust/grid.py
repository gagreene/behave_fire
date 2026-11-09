"""Owned Rust raster sessions with explicit stages, injections, and field lifetimes.

Physical values use base units: feet, feet/minute, pounds/foot squared, and
fractional moisture. Reaction intensity uses BTU/foot squared/minute. Masked
physical positions become NaN; all physical computation uses float64.
"""
import operator

import numpy as np

from ._bridge import _native
from ._model_api4 import _buffer, _FIELDS, _registry, _settings
from ._numeric import normalize_control, normalize_physical
from .fuel_models import FuelModels
from .fuel_models_array import build_fuel_lookup_arrays


def _fuel_snapshot(fuels):
    """Normalize independently owned registry data before native construction.

    :param fuels: FuelModels instance, numerical lookup mapping, or None for standard fuels.
    :return: Contiguous physical and flag registry columns.
    """
    if fuels is None:
        fuels = FuelModels()
    return _registry(fuels if isinstance(fuels, dict) else build_fuel_lookup_arrays(fuels))


def _workers(value):
    """Validate an explicit worker count without accepting floats or booleans.

    :param value: Integral count; zero requests automatic selection.
    :return: Nonnegative Python integer.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError('threads must be an integer')
    count = operator.index(value)
    if count < 0:
        raise ValueError('threads must be nonnegative')
    return count


class _Grid:
    """Shared structural wrapper; scientific equations execute only in Rust."""

    def __init__(
            self,
            inputs,
            *,
            shape=None,
            fuels=None,
            outputs=None,
            keep=(),
            threads=1,
            wind_orientation_mode='RelativeToUpslope',
            wind_height_mode='TwentyFoot',
            waf_method='CrownRatio',
            overrides=None,
            fuelbed=None,
    ):
        """Capture one complete common-grid model configuration.

        :param inputs: Mapping of actual base-unit physical inputs and typed controls.
        :param shape: Explicit common grid shape, or None to infer broadcast axes.
        :param fuels: Registry instance or lookup mapping; None selects standard fuels.
        :param outputs: Retained field names; None requests the model's default outputs.
        :param keep: Additional fields retained after automatic completion.
        :param threads: Worker count; zero selects available parallelism.
        :param wind_orientation_mode: RelativeToUpslope or RelativeToNorth.
        :param wind_height_mode: TwentyFoot or TenMetre.
        :param waf_method: CrownRatio or UserInput for Surface; Crown uses its fixed factor.
        :param overrides: Persistent derived-field overrides retained by resets.
        :param fuelbed: Complete supplied Surface fuelbed mapping, or None for fuel particles.
        :return: None; creates an independent native session without computing derived fields.
        """
        self._kinds = dict(_native.grid_fields())
        physical_names = set(_FIELDS[self._pipeline])
        allowed = physical_names | {'validity'}
        if self._pipeline == 'surface':
            allowed.update(('fuel_model_grid', 'user_waf'))
        unknown = set(inputs) - allowed
        if unknown:
            raise ValueError(f'Unknown {self._pipeline} inputs: {sorted(unknown)}')
        missing = physical_names - set(inputs)
        if self._pipeline == 'surface' and 'fuel_model_grid' not in inputs:
            missing.add('fuel_model_grid')
        if missing:
            raise ValueError(f'Missing {self._pipeline} inputs: {sorted(missing)}')
        physical = {name: normalize_physical(value) for name, value in inputs.items()
                    if self._kinds[name] == 'physical'}
        controls = {name: normalize_control(value, np.bool_ if name == 'validity' else np.int32)
                    for name, value in inputs.items() if self._kinds[name] != 'physical'}
        bed = None if fuelbed is None else {
            name: normalize_control(value, np.bool_) if name == 'is_defined' else normalize_physical(value)
            for name, value in fuelbed.items()}
        arrays = list(physical.values()) + list(controls.values()) + ([] if bed is None else list(bed.values()))
        if shape is None:
            shape = np.broadcast_shapes(*(value.shape for value in arrays))
        else:
            shape = tuple(operator.index(axis) for axis in shape)
        self._shape = tuple(shape)
        physical = {name: _buffer(value, shape) for name, value in physical.items()}
        controls = {name: _buffer(value, shape) for name, value in controls.items()}
        if bed is not None:
            bed = {name: _buffer(value, shape) for name, value in bed.items()}
        from ._bridge import _EMPTY_KEYS

        defaults = _EMPTY_KEYS['run_surface_fire' if self._pipeline == 'surface' else 'run_crown_fire']
        selected = tuple(defaults if outputs is None else outputs)
        retained = tuple(keep)
        self._native = _native.GridSession(self._pipeline, physical, controls, _fuel_snapshot(fuels), shape,
            _settings(dict(wind_orientation_mode=wind_orientation_mode, wind_height_mode=wind_height_mode,
                           waf_method=waf_method)), selected, retained, _workers(threads), bed)
        self._outputs = selected
        self._keep = retained
        self._threads = _workers(threads)
        for name, value in (overrides or {}).items():
            self.override(name, value)

    def __reduce__(self):
        """Reject native session serialization explicitly.

        :return: Never returns; raises TypeError with reconstruction guidance.
        """
        raise TypeError('Grid sessions cannot be pickled; reconstruct from inputs.')

    def _column(self, name, value, required=None):
        """Normalize one typed field and broadcast it to this session's fixed grid.

        :param name: Stable native field name.
        :param value: Actual replacement values, including physical NaNs.
        :param required: Required logical kind, or None to resolve from the field.
        :return: Contiguous scalar or full column with validated storage kind.
        """
        kind = self._kinds.get(name)
        if kind is None:
            raise ValueError(f'Unknown grid field {name}')
        if required is not None and kind != required:
            raise TypeError(f'{name} requires {kind} storage')
        array = normalize_physical(value) if kind == 'physical' else normalize_control(
            value, np.int32 if kind == 'code' else np.bool_)
        return _buffer(array, self._shape)

    def clear_injection(self, name):
        """Remove one transient injection and invalidate its unpinned descendants.

        :param name: Field whose transient value should be cleared.
        :return: None; retained inputs and persistent overrides survive.
        """
        self._native.clear(name, False)

    def clear_override(self, name):
        """Remove one persistent override while retaining transient injections.

        :param name: Field whose persistent override should be cleared.
        :return: None; unpinned descendants require recomputation.
        """
        self._native.clear(name, True)

    def get(self, name):
        """Copy a currently retained field without triggering computation.

        :param name: Stable native field name, including dotted particle and fuelbed names.
        :return: Independent array on the common shape; freed/uncomputed fields raise ValueError.
        """
        return self._native.get(name)

    @property
    def outputs(self):
        """Read the selected output names.

        :return: Immutable tuple of fields retained after completion.
        """
        return self._outputs

    def override(self, name, value):
        """Set a persistent typed derived-field override.

        :param name: Derived field name.
        :param value: Values retained across explicit stage runs and resets.
        :return: None; invalidates unpinned descendants.
        """
        self._native.set(name, self._column(name, value), True)

    def reset(self):
        """Clear derived state and transient injections.

        :return: None; input edits and persistent overrides survive.
        """
        self._native.reset()

    def run_fused(self):
        """Restart from inputs and persistent overrides, discarding transient injections.

        :return: None; fills requested fields using the fused kernel when supported,
                otherwise uses a fresh field DAG for custom intermediate requests or overrides.
        """
        self._native.run_fused()

    def run_remaining(self):
        """Complete requested fields while preserving staged injections, including NaNs.

        :return: None; recomputes freed ancestors as needed and releases unretained derived fields.
        """
        self._native.run_remaining()

    def run_stage(self, name):
        """Recompute every field of an explicit stage in dependency order.

        :param name: Stage name, such as particles, fuelbed, reaction, or crown_fraction.
        :return: None; clears this stage's transient injections and honors persistent overrides.
        """
        self._native.run_stage(name)

    def set(self, name, value):
        """Edit a physical input or pin a transient physical intermediate.

        :param name: Stable physical field name.
        :param value: Real base-unit scalar or array; masked positions become NaN.
        :return: None; owns replacements and invalidates unpinned descendants.
        """
        self._native.set(name, self._column(name, value, 'physical'), False)

    def set_code(self, name, value):
        """Edit exact fuel identifiers or inject a Crown classification.

        :param name: fuel_model_grid or fire_type.
        :param value: Integral scalar/array; fire_type accepts -1 through 3.
        :return: None; owns replacements and invalidates descendants.
        """
        self._native.set(name, self._column(name, value, 'code'), False)

    def set_flag(self, name, value):
        """Edit validity/definition flags or inject a classification flag.

        :param name: Stable boolean field name.
        :param value: Boolean or integral scalar/array without missing values.
        :return: None; owns replacements and invalidates descendants. Final Crown
                selection prioritizes Surface, then passive, then active flags;
                all-false classification flags leave final behavior unavailable.
        """
        self._native.set(name, self._column(name, value, 'flag'), False)

    def set_fuels(self, fuels):
        """Replace this session's registry snapshot without sharing caller data.

        :param fuels: Registry instance or numerical lookup mapping.
        :return: None; invalidates unpinned fuel descendants; supplied-fuelbed sessions reject this edit.
        """
        self._native.set_fuels(_fuel_snapshot(fuels))

    def set_outputs(self, outputs, *, keep=()):
        """Select output and retained intermediate fields atomically.

        :param outputs: Requested stable field names.
        :param keep: Additional retained intermediate field names.
        :return: None; the next completion applies the new liveness request.
        """
        selected, retained = tuple(outputs), tuple(keep)
        self._native.set_outputs(selected, retained)
        self._outputs, self._keep = selected, retained

    def set_threads(self, threads):
        """Set workers for subsequent staged or fused calls.

        :param threads: Nonnegative integral count; zero selects available parallelism.
        :return: None; updates the worker setting.
        """
        count = _workers(threads)
        self._native.set_threads(count)
        self._threads = count

    @property
    def shape(self):
        """Read the immutable common logical grid shape.

        :return: Tuple of axis lengths, including genuine empty axes.
        """
        return self._shape

    @property
    def threads(self):
        """Read the explicit worker request.

        :return: Nonnegative integer; zero requests automatic selection.
        """
        return self._threads


class CrownGrid(_Grid):
    """Staged model-10 Crown raster; supplied Surface results are independent inputs.

    Required inputs are the nine weather/moisture fields plus canopy_base_height,
    canopy_height, canopy_bulk_density, moisture_foliar, surface_ros, surface_fli,
    and surface_hpua. Only real numeric data is accepted; no Python counterpart
    or per-input backend fallback exists for this Rust-only API.
    """

    _pipeline = 'crown'


class SurfaceGrid(_Grid):
    """Staged Surface raster accepting scalar or common-shape input columns.

    Required inputs are fuel_model_grid, m1h, m10h, m100h, mlh, mlw, wind_speed,
    wind_direction, slope, aspect, canopy_cover, canopy_height, and crown_ratio.
    Optional validity and user_waf columns retain exact structural roles.
    Physical masked values become NaN; complex/object inputs raise TypeError.
    """

    _pipeline = 'surface'
