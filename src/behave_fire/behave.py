"""
behave.py — Array-Mode Facade (V9)

BehaveRun is the public entry point for array/raster mode.
It accepts NumPy arrays for all spatially-variable inputs and returns
dicts of output arrays of the same shape.

Usage
-----
    import numpy as np
    from behave.components.fuel_models import FuelModels
    from behave.behave import BehaveRun
    from behave.components.behave_units import SpeedUnits

    fm     = FuelModels()
    runner = BehaveRun(fm)

    results = runner.do_surface_run(
        fuel_model_grid=np.array([[124]]),
        m1h=np.array([[0.06]]), m10h=np.array([[0.07]]),
        m100h=np.array([[0.08]]), mlh=np.array([[0.60]]),
        mlw=np.array([[0.90]]),
        wind_speed=np.array([[5.0]]),
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        wind_direction=np.array([[0.0]]),
        wind_orientation_mode='RelativeToUpslope',
        slope=np.array([[30.0]]),          # 30% slope
        slope_units=1,                     # 1 = SlopeUnitsEnum.Percent
        aspect=np.array([[0.0]]),
        canopy_cover=np.array([[0.50]]),
        canopy_height=np.array([[30.0]]),
        crown_ratio=np.array([[0.50]]),
    )
    ros = results['spread_rate'][0, 0]   # ft/min

Notes
-----
* slope_units=0 → Degrees (default); slope_units=1 → Percent.
  The conversion (percent → degrees via arctan) is handled automatically.
* BehaveRun accepts single-value inputs and returns shape-(1,) or
  shape-(1,1) arrays, not Python scalars.
"""

import numpy as np
from typing import Union, Optional

try:
    from .components.fuel_models import FuelModels
    from .components.fuel_models_array import build_fuel_lookup_arrays
    from .components.surface import (
        build_particle_arrays,
        calculate_fuelbed_intermediates,
        calculate_reaction_intensity,
        calculate_spread_rate,
        calculate_fire_area,
        calculate_fire_perimeter,
        calculate_fire_length,
        calculate_fire_width,
    )
    from .components.crown import calculate_crown_fire
    from .components.mortality import (
        calculate_scorch_height,
        build_mortality_lookup,
        calculate_crown_scorch_mortality,
    )
    from .components.spot import (
        calculate_spotting_from_surface_fire,
        calculate_spotting_from_burning_pile,
        calculate_spotting_from_torching_trees,
    )
    from .components.behave_units import (
        fireline_intensity_to_base,
        fireline_intensity_from_base,
        speed_to_base,
        speed_from_base,
        slope_to_base,
        length_to_base,
        length_from_base,
        area_from_base,
        time_to_base,
        time_from_base,
        temp_to_base,
        hpua_from_base,
        reaction_intensity_from_base,
        fraction_to_base,
        fraction_from_base,
        density_to_base,
    )
except ImportError:
    from components.fuel_models import FuelModels
    from components.fuel_models_array import build_fuel_lookup_arrays
    from components.surface import (
        build_particle_arrays,
        calculate_fuelbed_intermediates,
        calculate_reaction_intensity,
        calculate_spread_rate,
        calculate_fire_area,
        calculate_fire_perimeter,
        calculate_fire_length,
        calculate_fire_width,
    )
    from components.crown import calculate_crown_fire
    from components.mortality import (
        calculate_scorch_height,
        build_mortality_lookup,
        calculate_crown_scorch_mortality,
    )
    from components.spot import (
        calculate_spotting_from_surface_fire,
        calculate_spotting_from_burning_pile,
        calculate_spotting_from_torching_trees,
    )
    from components.behave_units import (
        fireline_intensity_to_base,
        fireline_intensity_from_base,
        speed_to_base,
        speed_from_base,
        slope_to_base,
        length_to_base,
        length_from_base,
        area_from_base,
        time_to_base,
        time_from_base,
        temp_to_base,
        hpua_from_base,
        reaction_intensity_from_base,
        fraction_to_base,
        fraction_from_base,
        density_to_base,
    )


# ---------------------------------------------------------------------------
# Output-unit conversion helper
# ---------------------------------------------------------------------------

# Maps each output key to (converter_fn, stored_base_unit_index).
# stored_base_unit_index is the unit the component already stores the value in
# (needed for keys like 'effective_wind_speed' whose base differs from ft/min).
_SURFACE_KEY_CONVERTERS = {
    # key:                               (converter,                    stored_base)
    'spread_rate':                       (speed_from_base,              0),   # ft/min
    'backing_spread_rate':               (speed_from_base,              0),
    'flanking_spread_rate':              (speed_from_base,              0),
    'no_wind_no_slope_spread_rate':      (speed_from_base,              0),
    'midflame_wind_speed':               (speed_from_base,              0),
    'effective_wind_speed':              (speed_from_base,              5),   # stored as mph
    'flame_length':                      (length_from_base,             0),   # ft
    'fireline_intensity':                (fireline_intensity_from_base, 0),   # BTU/ft/s
    'heat_per_unit_area':                (hpua_from_base,               0),   # BTU/ft²
    'reaction_intensity':                (reaction_intensity_from_base, 0),   # BTU/ft²/min
    'residence_time':                    (time_from_base,               0),   # min
    # dimensionless — no conversion
    'fire_length_to_width_ratio':        None,
    'eccentricity':                      None,
    'direction_of_max_spread':           None,
}

_CROWN_KEY_CONVERTERS = {
    'crown_fire_spread_rate':                        (speed_from_base,              0),
    'crown_critical_fire_spread_rate':               (speed_from_base,              0),
    'crown_flame_length':                            (length_from_base,             0),
    'crown_fire_line_intensity':                     (fireline_intensity_from_base, 0),
    'crown_critical_surface_fire_line_intensity':    (fireline_intensity_from_base, 0),
    'crown_fire_heat_per_unit_area':                 (hpua_from_base,               0),
    'canopy_heat_per_unit_area':                     (hpua_from_base,               0),
    # Scott & Reinhardt CFB intermediates
    'surface_fire_critical_spread_rate':             (speed_from_base,              0),
    'crowning_surface_fire_spread_rate':             (speed_from_base,              0),
    'passive_crown_fire_spread_rate':                (speed_from_base,              0),
    'passive_crown_fire_heat_per_unit_area':         (hpua_from_base,               0),
    'passive_crown_fire_line_intensity':             (fireline_intensity_from_base, 0),
    'passive_crown_fire_flame_length':               (length_from_base,             0),
    # blended / final head fire outputs
    'final_spread_rate':                             (speed_from_base,              0),
    'final_fireline_intensity':                      (fireline_intensity_from_base, 0),
    'final_heat_per_unit_area':                      (hpua_from_base,               0),
    'final_flame_length':                            (length_from_base,             0),
    # dimensionless / categorical — no conversion
    'crown_fire_transition_ratio':                   None,
    'crown_fire_active_ratio':                       None,
    'crown_length_to_width_ratio':                   None,
    'fire_type':                                     None,
    'crown_fraction_burned':                         None,
}

_MORTALITY_KEY_CONVERTERS = {
    'crown_length_scorch':   (fraction_from_base, 0),
    'crown_volume_scorch':   (fraction_from_base, 0),
    'probability_mortality': (fraction_from_base, 0),
}


def _apply_out_units(results: dict, out_units: dict, converters: dict) -> dict:
    """
    Convert each key in *results* whose converter is registered in *converters*
    to the unit requested in *out_units*.

    Keys absent from *out_units* are left in their base units.
    Keys with ``None`` converter (dimensionless / categorical) are never converted.

    :param results: Raw output dict from a component function.
    :param out_units: Caller-supplied ``{key: unit_enum_int}`` overrides.
    :param converters: ``{key: (converter_fn, stored_base_unit)}`` map.
    :return: New dict with requested conversions applied.
    """
    if not out_units:
        return results
    out = dict(results)
    for key, entry in converters.items():
        if entry is None or key not in out_units:
            continue
        converter_fn, stored_base = entry
        requested = out_units[key]
        if requested == stored_base:
            continue                        # already in the right unit
        # Convert: stored_base → ft/min (or whichever true base) → requested.
        # For keys stored in a non-zero base (e.g. effective_wind_speed in mph)
        # we first go back to the true base (index 0) then out to requested.
        value = out[key]
        if stored_base != 0:
            value = speed_to_base(value=value, units=stored_base)   # mph → ft/min
        out[key] = converter_fn(value, requested)
    return out


class BehaveRun:
    """
    Array-mode facade for behave_fire.

    Accepts NumPy arrays for all spatially-variable inputs.
    Scalar inputs (unit enums, mode strings) remain Python scalars.

    The fuel lookup table is built once at construction from the FuelModels
    instance and reused for all subsequent run calls.
    """

    def __init__(self, fuel_models: FuelModels):
        """
        Parameters
        ----------
        fuel_models : FuelModels
            Populated FuelModels instance (scalar, unchanged).
        """
        self._fuel_models = fuel_models
        self._lut = build_fuel_lookup_arrays(fuel_models)
        self._mortality_coeffs = build_mortality_lookup()

    # ------------------------------------------------------------------
    # Surface run
    # ------------------------------------------------------------------

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
            moisture_units: int = 1,
            aspect: Union[float, np.ndarray] = 0.0,
            canopy_cover: Union[float, np.ndarray] = 0.0,
            canopy_cover_units: int = 1,
            canopy_height: Union[float, np.ndarray] = 0.0,
            canopy_height_units: int = 4,
            canopy_base_height: Union[float, np.ndarray] = 0.0,
            canopy_base_height_units: int = 4,
            crown_ratio: Union[float, np.ndarray] = 0.0,
            crown_ratio_units: Optional[int] = 1,
            wind_height_mode: str = 'TwentyFoot',
            waf_method: str = 'UseCrownRatio',
            user_waf: Union[float, np.ndarray, None] = None,
            out_units: Optional[dict] = None,
    ) -> dict:
        """
        Run the vectorized surface fire pipeline.

        :param fuel_model_grid: Integer fuel model number array (*S).
        :param m1h: 1-hr dead fuel moisture as fraction (*S) or scalar (e.g. 0.06 = 6%).
        :param m10h: 10-hr dead fuel moisture as fraction (*S) or scalar.
        :param m100h: 100-hr dead fuel moisture as fraction (*S) or scalar.
        :param mlh: Live herbaceous fuel moisture as fraction (*S) or scalar.
        :param mlw: Live woody fuel moisture as fraction (*S) or scalar.
        :param wind_speed: Wind speed (*S) or scalar, in ``wind_speed_units``.
        :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
        :param wind_direction: Wind direction in degrees (*S) or scalar.
            Interpretation depends on ``wind_orientation_mode``.
        :param wind_orientation_mode: ``'RelativeToUpslope'`` or ``'RelativeToNorth'``.
        :param slope: Slope (*S) or scalar, in the units given by ``slope_units``.
        :param slope_units: Scalar integer ``SlopeUnitsEnum`` value
            (0 = Degrees [default], 1 = Percent).
        :param moisture_units: ``FractionUnitsEnum`` integer for m1h–mlw
            (1 = Percent [default], 0 = Fraction).
        :param aspect: Terrain aspect in degrees (*S) or scalar (0 = north, clockwise).
        :param canopy_cover: Canopy cover (*S) or scalar, in ``canopy_cover_units``.
        :param canopy_cover_units: ``FractionUnitsEnum`` integer
            (1 = Percent [default], 0 = Fraction).
        :param canopy_height: Canopy height (*S) or scalar, in ``canopy_height_units``.
        :param canopy_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param canopy_base_height: Height to base of canopy (*S) or scalar, in
            ``canopy_base_height_units``. Used only when ``crown_ratio_units=None``
            to derive crown ratio as ``(canopy_height − canopy_base_height) / canopy_height``.
        :param canopy_base_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param crown_ratio: Crown ratio (*S) or scalar, in ``crown_ratio_units``.
            Ignored when ``crown_ratio_units=None``.
        :param crown_ratio_units: ``FractionUnitsEnum`` integer
            (1 = Percent [default], 0 = Fraction), or ``None`` to derive crown ratio
            from ``(canopy_height − canopy_base_height) / canopy_height``.
        :param wind_height_mode: Height at which ``wind_speed`` was measured.
            ``'TwentyFoot'`` (default) — standard 20-ft open-wind measurement;
            ``'TenMeter'`` — 10-m meteorological-station measurement.
            The value is internally adjusted to midflame height using the WAF.
        :param waf_method: Controls how the wind adjustment factor (WAF) is
            computed to reduce the input wind speed to midflame height.
            ``'UseCrownRatio'`` (default) — WAF is derived from ``canopy_cover``,
            ``canopy_height``, and ``crown_ratio`` using the Albini & Baughman
            (1979) method; ``'UserInput'`` — the value supplied in ``user_waf``
            is used directly, bypassing the automatic calculation.
        :param user_waf: User-supplied WAF (*S) or scalar, used when
            ``waf_method='UserInput'``. Ignored otherwise.
        :param out_units: Optional ``dict`` mapping output key names to
            ``*UnitsEnum`` integers.  Only the keys you want converted need to
            be present; omitted keys remain in their default base units.
            Dimensionless keys (``fire_length_to_width_ratio``, ``eccentricity``,
            ``direction_of_max_spread``) are never converted.

            **Convertible keys and their defaults:**

            ============================================  ================================
            Key                                           Default (enum class · value)
            ============================================  ================================
            ``spread_rate``                               SpeedUnitsEnum.FeetPerMinute · 0
            ``backing_spread_rate``                       SpeedUnitsEnum.FeetPerMinute · 0
            ``flanking_spread_rate``                      SpeedUnitsEnum.FeetPerMinute · 0
            ``no_wind_no_slope_spread_rate``              SpeedUnitsEnum.FeetPerMinute · 0
            ``midflame_wind_speed``                       SpeedUnitsEnum.FeetPerMinute · 0
            ``effective_wind_speed``                      SpeedUnitsEnum.MilesPerHour · 5
            ``flame_length``                              LengthUnitsEnum.Feet · 0
            ``fireline_intensity``                        FirelineIntensityUnitsEnum.BtusPerFootPerSecond · 0
            ``heat_per_unit_area``                        HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot · 0
            ``reaction_intensity``                        HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerMinute · 0
            ``residence_time``                            TimeUnitsEnum.Minutes · 0
            ============================================  ================================

            **SpeedUnitsEnum options** (``spread_rate``, ``backing_spread_rate``,
            ``flanking_spread_rate``, ``no_wind_no_slope_spread_rate``,
            ``midflame_wind_speed``, ``effective_wind_speed``):

            =  ========================
            0  FeetPerMinute (default for ROS keys)
            1  ChainsPerHour
            2  MetersPerSecond
            3  MetersPerMinute
            4  MetersPerHour
            5  MilesPerHour (default for ``effective_wind_speed``)
            6  KilometersPerHour
            =  ========================

            **LengthUnitsEnum options** (``flame_length``):

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

            **FirelineIntensityUnitsEnum options** (``fireline_intensity``):

            =  ==============================
            0  BtusPerFootPerSecond
            1  BtusPerFootPerMinute
            2  KilojoulesPerMeterPerSecond
            3  KilojoulesPerMeterPerMinute
            4  KilowattsPerMeter
            =  ==============================

            **HeatPerUnitAreaUnitsEnum options** (``heat_per_unit_area``):

            =  ==============================
            0  BtusPerSquareFoot
            1  KilojoulesPerSquareMeter
            2  KilowattSecondsPerSquareMeter
            =  ==============================

            **HeatSourceAndReactionIntensityUnitsEnum options**
            (``reaction_intensity``):

            =  ===================================
            0  BtusPerSquareFootPerMinute
            1  BtusPerSquareFootPerSecond
            2  KilojoulesPerSquareMeterPerSecond
            3  KilojoulesPerSquareMeterPerMinute
            4  KilowattsPerSquareMeter
            =  ===================================

            **TimeUnitsEnum options** (``residence_time``):

            =  =======
            0  Minutes
            1  Seconds
            2  Hours
            3  Days
            4  Years
            =  =======

        :return: dict of (*S) ndarrays in the requested output units.
        """
        # Coerce all spatial inputs to at-least-1D ndarrays (G10 fix)
        fuel_model_grid = np.atleast_1d(np.asarray(fuel_model_grid, dtype=np.int32))
        m1h   = fraction_to_base(np.atleast_1d(np.asarray(m1h,   dtype=float)), moisture_units)
        m10h  = fraction_to_base(np.atleast_1d(np.asarray(m10h,  dtype=float)), moisture_units)
        m100h = fraction_to_base(np.atleast_1d(np.asarray(m100h, dtype=float)), moisture_units)
        mlh   = fraction_to_base(np.atleast_1d(np.asarray(mlh,   dtype=float)), moisture_units)
        mlw   = fraction_to_base(np.atleast_1d(np.asarray(mlw,   dtype=float)), moisture_units)
        wind_direction = np.atleast_1d(np.asarray(wind_direction, dtype=float))
        aspect         = np.atleast_1d(np.asarray(aspect,         dtype=float))
        canopy_cover   = fraction_to_base(np.atleast_1d(np.asarray(canopy_cover,  dtype=float)), canopy_cover_units)
        canopy_height_ft      = length_to_base(np.atleast_1d(np.asarray(canopy_height,      dtype=float)), canopy_height_units)
        canopy_base_height_ft = length_to_base(np.atleast_1d(np.asarray(canopy_base_height, dtype=float)), canopy_base_height_units)
        if crown_ratio_units is None:
            _safe_ht   = np.where(canopy_height_ft > 0, canopy_height_ft, 1.0)
            crown_ratio = np.clip((canopy_height_ft - canopy_base_height_ft) / _safe_ht, 0.0, 1.0)
        else:
            crown_ratio = fraction_to_base(np.atleast_1d(np.asarray(crown_ratio, dtype=float)), crown_ratio_units)

        # Unit conversions done here once, so components receive base units.
        wind_speed_fpm = speed_to_base(
            np.atleast_1d(np.asarray(wind_speed, dtype=float)), wind_speed_units
        )
        slope_deg = slope_to_base(
            np.atleast_1d(np.asarray(slope, dtype=float)), slope_units
        )

        p  = build_particle_arrays(
            lut=self._lut,
            fuel_model_grid=fuel_model_grid,
            m1h=m1h, m10h=m10h, m100h=m100h, mlh=mlh, mlw=mlw,
        )
        ib = calculate_fuelbed_intermediates(p=p)
        ri = calculate_reaction_intensity(ib=ib)
        results = calculate_spread_rate(
            ri=ri,
            ib=ib,
            wind_speed=wind_speed_fpm,
            wind_speed_units=0,
            wind_direction=wind_direction,
            wind_orientation_mode=wind_orientation_mode,
            slope=slope_deg,
            slope_units=0,
            aspect=aspect,
            canopy_cover=canopy_cover,
            canopy_height=canopy_height_ft,
            crown_ratio=crown_ratio,
            wind_height_mode=wind_height_mode,
            waf_method=waf_method,
            user_waf=user_waf,
        )
        return _apply_out_units(results, out_units or {}, _SURFACE_KEY_CONVERTERS)

    # ------------------------------------------------------------------
    # Crown run
    # ------------------------------------------------------------------

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
            moisture_units: int = 1,
            aspect: Union[float, np.ndarray] = 0.0,
            canopy_base_height: Union[float, np.ndarray] = 0.0,
            canopy_base_height_units: int = 4,
            canopy_height: Union[float, np.ndarray] = 0.0,
            canopy_height_units: int = 4,
            canopy_bulk_density: Union[float, np.ndarray] = 0.0,
            canopy_bulk_density_units: int = 1,
            moisture_foliar: Union[float, np.ndarray] = 100.0,
            out_units: Optional[dict] = None,
    ) -> dict:
        """
        Run the vectorized crown fire pipeline.

        :param surface_results: Output dict from ``do_surface_run()``.
        :param fuel_model_grid: Integer fuel model number array (*S).
        :param m1h: 1-hr dead fuel moisture as fraction (*S) or scalar.
        :param m10h: 10-hr dead fuel moisture as fraction (*S) or scalar.
        :param m100h: 100-hr dead fuel moisture as fraction (*S) or scalar.
        :param mlh: Live herbaceous fuel moisture as fraction (*S) or scalar.
        :param mlw: Live woody fuel moisture as fraction (*S) or scalar.
        :param wind_speed: Wind speed (*S) or scalar, in ``wind_speed_units``.
        :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
        :param wind_direction: Wind direction in degrees (*S) or scalar.
            Interpretation depends on ``wind_orientation_mode``.
        :param wind_orientation_mode: ``'RelativeToUpslope'`` — wind direction is
            measured clockwise from the upslope direction; ``'RelativeToNorth'`` —
            wind direction is a standard compass bearing (0° = north, clockwise).
        :param slope: Slope (*S) or scalar, in the units given by ``slope_units``.
        :param slope_units: Scalar integer ``SlopeUnitsEnum`` value
            (0 = Degrees [default], 1 = Percent).
        :param moisture_units: ``FractionUnitsEnum`` integer for m1h–mlw
            (1 = Percent [default], 0 = Fraction).
        :param aspect: Terrain aspect in degrees (*S) or scalar
            (0 = north, clockwise). Used to resolve wind direction when
            ``wind_orientation_mode='RelativeToNorth'``.
        :param canopy_base_height: Height to base of canopy (*S) or scalar, in
            ``canopy_base_height_units``.
        :param canopy_base_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param canopy_height: Total canopy height (*S) or scalar, in
            ``canopy_height_units``.
        :param canopy_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param canopy_bulk_density: Canopy bulk density (*S) or scalar, in
            ``canopy_bulk_density_units``.
        :param canopy_bulk_density_units: ``DensityUnitsEnum`` integer
            (1 = KilogramsPerCubicMeter [default], 0 = PoundsPerCubicFoot).
        :param moisture_foliar: Foliar moisture content (%) (*S) or scalar.
        :param out_units: Optional ``dict`` mapping output key names to
            ``*UnitsEnum`` integers.  Dimensionless / categorical keys
            (``crown_fire_transition_ratio``, ``crown_fire_active_ratio``,
            ``crown_length_to_width_ratio``, ``fire_type``,
            ``crown_fraction_burned``) are never converted.

            **Convertible keys and their defaults:**

            ================================================  ================================
            Key                                               Default (enum class · value)
            ================================================  ================================
            ``crown_fire_spread_rate``                        SpeedUnitsEnum.FeetPerMinute · 0
            ``crown_critical_fire_spread_rate``               SpeedUnitsEnum.FeetPerMinute · 0
            ``crown_flame_length``                            LengthUnitsEnum.Feet · 0
            ``crown_fire_line_intensity``                     FirelineIntensityUnitsEnum.BtusPerFootPerSecond · 0
            ``crown_critical_surface_fire_line_intensity``    FirelineIntensityUnitsEnum.BtusPerFootPerSecond · 0
            ``crown_fire_heat_per_unit_area``                 HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot · 0
            ``canopy_heat_per_unit_area``                     HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot · 0
            ``surface_fire_critical_spread_rate``             SpeedUnitsEnum.FeetPerMinute · 0
            ``crowning_surface_fire_spread_rate``             SpeedUnitsEnum.FeetPerMinute · 0
            ``passive_crown_fire_spread_rate``                SpeedUnitsEnum.FeetPerMinute · 0
            ``passive_crown_fire_heat_per_unit_area``         HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot · 0
            ``passive_crown_fire_line_intensity``             FirelineIntensityUnitsEnum.BtusPerFootPerSecond · 0
            ``passive_crown_fire_flame_length``               LengthUnitsEnum.Feet · 0
            ``final_spread_rate``                             SpeedUnitsEnum.FeetPerMinute · 0
            ``final_fireline_intensity``                      FirelineIntensityUnitsEnum.BtusPerFootPerSecond · 0
            ``final_heat_per_unit_area``                      HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot · 0
            ``final_flame_length``                            LengthUnitsEnum.Feet · 0
            ================================================  ================================

            **``fire_type`` values** (dimensionless int, not converted):

            =  ====================  =================================================
            0  Surface               No crown fire; ``transition_ratio < 1``,
                                     ``active_ratio < 1``.
            1  Torching              Passive crown fire; ``transition_ratio ≥ 1``,
                                     ``active_ratio < 1``.
            2  ConditionalCrownFire  Active crown fire possible if fire can transition;
                                     ``transition_ratio < 1``, ``active_ratio ≥ 1``.
            3  Crowning             Active crown fire; both ratios ≥ 1.
            =  ====================  =================================================

            **Scott & Reinhardt CFB and intermediate keys:**

            ``crown_fraction_burned`` (dimensionless, 0–1) — fraction of the
            canopy that is actively burning, using the linear formula from Scott
            & Reinhardt (2001):
            ``CFB = (surface_ros − R'initiation) / (R'sa − R'initiation)``,
            clamped to [0, 1].  ``R'initiation`` is ``surface_fire_critical_spread_rate``
            (surface ROS at which torching begins); ``R'sa`` is
            ``crowning_surface_fire_spread_rate`` (surface ROS at which active crown
            fire is fully achieved).

            ``surface_fire_critical_spread_rate`` (R'initiation) — the surface
            fire spread rate at which the fireline intensity equals the Van Wagner
            (1977) crown ignition threshold:
            ``R'initiation = (60 × crit_surface_fli) / surface_hpua``.

            ``crowning_surface_fire_spread_rate`` (R'sa) — the surface fire spread
            rate at which the active crown fire spread rate is fully achieved (CFB → 1).
            Derived by back-solving the FM10 wind equation to find the 20-ft wind
            speed that drives FM10 to R'active, then running the surface model at
            that wind speed.

            ``passive_crown_fire_*`` — blended passive crown fire values computed as:
            ``passive_ros = surface_ros + CFB × (crown_ros − surface_ros)`` and
            ``passive_hpua = surface_hpua + CFB × canopy_hpua``.

            **Final head fire assignments:**

            * Surface or ConditionalCrownFire → ``final_*`` = surface fire values.
            * Torching (passive) → ``final_*`` = ``passive_crown_fire_*`` values.
            * Crowning (active) → ``final_*`` = active ``crown_fire_*`` values.

            **SpeedUnitsEnum options** (``crown_fire_spread_rate``,
            ``crown_critical_fire_spread_rate``, ``surface_fire_critical_spread_rate``,
            ``crowning_surface_fire_spread_rate``, ``passive_crown_fire_spread_rate``,
            ``final_spread_rate``):

            =  =====================
            0  FeetPerMinute
            1  ChainsPerHour
            2  MetersPerSecond
            3  MetersPerMinute
            4  MetersPerHour
            5  MilesPerHour
            6  KilometersPerHour
            =  =====================

            **LengthUnitsEnum options** (``crown_flame_length``,
            ``passive_crown_fire_flame_length``, ``final_flame_length``):

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

            **FirelineIntensityUnitsEnum options** (``crown_fire_line_intensity``,
            ``crown_critical_surface_fire_line_intensity``,
            ``passive_crown_fire_line_intensity``, ``final_fireline_intensity``):

            =  ==============================
            0  BtusPerFootPerSecond
            1  BtusPerFootPerMinute
            2  KilojoulesPerMeterPerSecond
            3  KilojoulesPerMeterPerMinute
            4  KilowattsPerMeter
            =  ==============================

            **HeatPerUnitAreaUnitsEnum options** (``crown_fire_heat_per_unit_area``,
            ``canopy_heat_per_unit_area``, ``passive_crown_fire_heat_per_unit_area``,
            ``final_heat_per_unit_area``):

            =  ==============================
            0  BtusPerSquareFoot
            1  KilojoulesPerSquareMeter
            2  KilowattSecondsPerSquareMeter
            =  ==============================

        :return: dict in the requested output units.
        """
        wind_speed_fpm = speed_to_base(
            np.atleast_1d(np.asarray(wind_speed, dtype=float)), wind_speed_units
        )
        slope_deg = slope_to_base(
            np.atleast_1d(np.asarray(slope, dtype=float)), slope_units
        )
        m1h   = fraction_to_base(np.atleast_1d(np.asarray(m1h,   dtype=float)), moisture_units)
        m10h  = fraction_to_base(np.atleast_1d(np.asarray(m10h,  dtype=float)), moisture_units)
        m100h = fraction_to_base(np.atleast_1d(np.asarray(m100h, dtype=float)), moisture_units)
        mlh   = fraction_to_base(np.atleast_1d(np.asarray(mlh,   dtype=float)), moisture_units)
        mlw   = fraction_to_base(np.atleast_1d(np.asarray(mlw,   dtype=float)), moisture_units)
        cbh_ft  = length_to_base(np.atleast_1d(np.asarray(canopy_base_height,  dtype=float)), canopy_base_height_units)
        ch_ft   = length_to_base(np.atleast_1d(np.asarray(canopy_height,       dtype=float)), canopy_height_units)
        cbd_pcf = density_to_base(np.atleast_1d(np.asarray(canopy_bulk_density, dtype=float)), canopy_bulk_density_units)
        results = calculate_crown_fire(
            surface_results=surface_results,
            lut=self._lut,
            fuel_model_grid=fuel_model_grid,
            m1h=m1h, m10h=m10h, m100h=m100h, mlh=mlh, mlw=mlw,
            wind_speed=wind_speed_fpm,
            wind_speed_units=0,
            wind_direction=wind_direction,
            wind_orientation_mode=wind_orientation_mode,
            slope=slope_deg,
            slope_units=0,
            aspect=aspect,
            canopy_base_height=cbh_ft,
            canopy_height=ch_ft,
            canopy_bulk_density=cbd_pcf,
            moisture_foliar=moisture_foliar,
        )
        return _apply_out_units(results, out_units or {}, _CROWN_KEY_CONVERTERS)

    # ------------------------------------------------------------------
    # Scorch height
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_scorch_height(
            fireline_intensity: Union[float, np.ndarray],
            fireline_intensity_units: int,
            midflame_wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            air_temperature: Union[float, np.ndarray],
            temperature_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Vectorized scorch height.

        :param fireline_intensity: Fireline intensity (*S) or scalar, in
            ``fireline_intensity_units``.
        :param fireline_intensity_units: Scalar integer
            ``FirelineIntensityUnitsEnum`` value.
        :param midflame_wind_speed: Midflame wind speed (*S) or scalar, in
            ``wind_speed_units``.
        :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
        :param air_temperature: Air temperature (*S) or scalar, in
            ``temperature_units``.
        :param temperature_units: Scalar integer ``TemperatureUnitsEnum`` value.
        :param out_units: ``LengthUnitsEnum`` integer for the output scorch
            height.  Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — scorch height in ``out_units``.
        """
        fi     = fireline_intensity_to_base(value=fireline_intensity, units=fireline_intensity_units)
        ws_fpm = speed_to_base(value=midflame_wind_speed, units=wind_speed_units)
        ws_mph = speed_from_base(value=ws_fpm, units=5)
        t_f    = temp_to_base(value=air_temperature, units=temperature_units)
        result = calculate_scorch_height(
            fireline_intensity_btu_ft_s=fi,
            midflame_wind_mph=ws_mph,
            air_temp_f=t_f,
        )
        return length_from_base(value=result, units=out_units)

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------

    def calculate_crown_scorch_mortality(
            self,
            scorch_height: Union[float, np.ndarray],
            tree_height: Union[float, np.ndarray],
            crown_ratio: Union[float, np.ndarray],
            dbh: Union[float, np.ndarray],
            equation_number_grid: Union[int, np.ndarray],
            scorch_height_units: int = 4,
            tree_height_units: int = 4,
            crown_ratio_units: int = 1,
            dbh_units: int = 3,
            out_units: Optional[dict] = None,
    ) -> dict:
        """
        Vectorized crown scorch mortality.

        :param scorch_height: Height above ground to which foliage is scorched
            (*S) or scalar, in ``scorch_height_units``. Typically obtained from
            ``calculate_scorch_height()``.
        :param tree_height: Total tree height (*S) or scalar, in
            ``tree_height_units``.
        :param crown_ratio: Live crown ratio (*S) or scalar, in
            ``crown_ratio_units``.
        :param dbh: Diameter at breast height (*S) or scalar, in ``dbh_units``.
        :param equation_number_grid: Mortality equation number per cell (*S) int.
            Crown-scorch equations are 1–20; bole-char equations are 100–109.
        :param scorch_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param tree_height_units: ``LengthUnitsEnum`` integer
            (4 = Meters [default], 0 = Feet).
        :param crown_ratio_units: ``FractionUnitsEnum`` integer
            (1 = Percent [default], 0 = Fraction).
        :param dbh_units: ``LengthUnitsEnum`` integer
            (3 = Centimeters [default], 1 = Inches).
        :param out_units: Optional ``dict`` mapping output key names to
            ``FractionUnitsEnum`` integers.  Convertible keys and their defaults:

            =======================  ===================================
            Key                      Default (enum class · value)
            =======================  ===================================
            ``crown_length_scorch``  FractionUnitsEnum.Fraction · 0
            ``crown_volume_scorch``  FractionUnitsEnum.Fraction · 0
            ``probability_mortality`` FractionUnitsEnum.Fraction · 0
            =======================  ===================================

            **FractionUnitsEnum options:**

            =  =======
            0  Fraction  (0–1)
            1  Percent   (0–100)
            =  =======

        :return: dict with converted (*S) ndarrays.
        """
        scorch_ft  = length_to_base(np.asarray(scorch_height, dtype=float), scorch_height_units)
        tree_ht_ft = length_to_base(np.asarray(tree_height,   dtype=float), tree_height_units)
        cr_frac    = fraction_to_base(np.asarray(crown_ratio,  dtype=float), crown_ratio_units)
        dbh_in     = length_to_base(np.asarray(dbh,           dtype=float), dbh_units) * 12.0
        results = calculate_crown_scorch_mortality(
            scorch_height_ft=scorch_ft,
            tree_height_ft=tree_ht_ft,
            crown_ratio=cr_frac,
            dbh_inches=dbh_in,
            equation_number_grid=equation_number_grid,
            coeffs=self._mortality_coeffs,
        )
        return _apply_out_units(results, out_units or {}, _MORTALITY_KEY_CONVERTERS)

    # ------------------------------------------------------------------
    # Spotting
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_spotting_from_surface_fire(
            flame_length: Union[float, np.ndarray],
            flame_length_units: int,
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            cover_height: Union[float, np.ndarray],
            cover_height_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Vectorized surface fire spotting distance (Albini 1979).

        :param flame_length: Flame length (*S) or scalar, in ``flame_length_units``.
        :param flame_length_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param wind_speed: 20-ft open wind speed (*S) or scalar, in ``wind_speed_units``.
        :param wind_speed_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, 5=MilesPerHour, etc.).
        :param cover_height: Cover height downwind (*S) or scalar, in ``cover_height_units``.
        :param cover_height_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param out_units: ``LengthUnitsEnum`` integer for the output spotting
            distance.  Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — spotting distance in ``out_units``.
        """
        fl_ft  = length_to_base(value=flame_length, units=flame_length_units)
        ws_mph = speed_from_base(value=speed_to_base(value=wind_speed, units=wind_speed_units), units=5)
        ch_ft  = length_to_base(value=cover_height, units=cover_height_units)
        result = calculate_spotting_from_surface_fire(
            flame_length_ft=fl_ft,
            wind_mph=ws_mph,
            cover_height_ft=ch_ft,
        )
        return length_from_base(value=result, units=out_units)

    @staticmethod
    def calculate_spotting_from_burning_pile(
            flame_height: Union[float, np.ndarray],
            flame_height_units: int,
            wind_speed: Union[float, np.ndarray],
            wind_speed_units: int,
            cover_height: Union[float, np.ndarray],
            cover_height_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Vectorized burning pile spotting distance (Albini 1979).

        :param flame_height: Flame height of the pile (*S) or scalar, in ``flame_height_units``.
        :param flame_height_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param wind_speed: 20-ft open wind speed (*S) or scalar, in ``wind_speed_units``.
        :param wind_speed_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, 5=MilesPerHour, etc.).
        :param cover_height: Cover height downwind (*S) or scalar, in ``cover_height_units``.
        :param cover_height_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param out_units: ``LengthUnitsEnum`` integer for the output spotting
            distance.  Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — spotting distance in ``out_units``.
        """
        fh_ft  = length_to_base(value=flame_height, units=flame_height_units)
        ws_mph = speed_from_base(value=speed_to_base(value=wind_speed, units=wind_speed_units), units=5)
        ch_ft  = length_to_base(value=cover_height, units=cover_height_units)
        result = calculate_spotting_from_burning_pile(
            flame_height_ft=fh_ft,
            wind_mph=ws_mph,
            cover_height_ft=ch_ft,
        )
        return length_from_base(value=result, units=out_units)

    @staticmethod
    def calculate_spotting_from_torching_trees(
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
        """
        Vectorized torching-tree spotting distance (Albini 1979).

        :param dbh: Tree diameter at breast height (*S) or scalar, in ``dbh_units``.
        :param dbh_units: ``LengthUnitsEnum`` integer (0=Feet, 1=Inches, 4=Meters, etc.).
        :param height: Tree height (*S) or scalar, in ``height_units``.
        :param height_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param count: Number of torching trees (*S) or scalar (dimensionless).
        :param wind_speed: 20-ft open wind speed (*S) or scalar, in ``wind_speed_units``.
        :param wind_speed_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, 5=MilesPerHour, etc.).
        :param cover_height: Cover height downwind (*S) or scalar, in ``cover_height_units``.
        :param cover_height_units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
        :param out_units: ``LengthUnitsEnum`` integer for the output spotting
            distance.  Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — spotting distance in ``out_units``.
        """
        dbh_in = length_to_base(value=dbh, units=dbh_units) * 12.0  # ft → inches
        ht_ft  = length_to_base(value=height, units=height_units)
        ws_mph = speed_from_base(value=speed_to_base(value=wind_speed, units=wind_speed_units), units=5)
        ch_ft  = length_to_base(value=cover_height, units=cover_height_units)
        result = calculate_spotting_from_torching_trees(
            dbh_in=dbh_in,
            height_ft=ht_ft,
            count=count,
            wind_mph=ws_mph,
            cover_height_ft=ch_ft,
        )
        return length_from_base(value=result, units=out_units)

    # ------------------------------------------------------------------
    # Fire size / shape
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_fire_area(
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            is_crown: bool = False,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Elliptical fire area.

        :param forward_ros: Forward rate of spread (*S), in ``ros_units``.
        :param backing_ros: Backing rate of spread (*S), in ``ros_units``.
        :param ros_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, etc.).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S). Values
            greater than 1 indicate an elongated ellipse; 1 is circular.
            Typically taken from ``results['fire_length_to_width_ratio']``.
        :param elapsed_time: Elapsed time since ignition, in ``elapsed_time_units``.
            Must be a scalar (single time step).
        :param elapsed_time_units: ``TimeUnitsEnum`` integer (0=Minutes, 2=Hours, etc.).
        :param is_crown: If ``True``, applies the Scott & Reinhardt (2001) crown
            fire area approximation instead of the standard surface fire formula.
        :param out_units: ``AreaUnitsEnum`` integer for the output area.
            Default ``0`` = SquareFeet.

            =  ================
            0  SquareFeet
            1  Acres
            2  Hectares
            3  SquareMeters
            4  SquareMiles
            5  SquareKilometers
            =  ================

        :return: (*S) ndarray — fire area in ``out_units``.
        """
        fros_fpm    = speed_to_base(value=forward_ros, units=ros_units)
        bros_fpm    = speed_to_base(value=backing_ros, units=ros_units)
        elapsed_min = float(time_to_base(value=elapsed_time, units=elapsed_time_units))
        result = calculate_fire_area(
            forward_ros=fros_fpm,
            backing_ros=bros_fpm,
            lwr=lwr,
            elapsed_min=elapsed_min,
            is_crown=is_crown,
        )
        return area_from_base(value=result, units=out_units)

    @staticmethod
    def calculate_fire_perimeter(
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            is_crown: bool = False,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Elliptical fire perimeter (Ramanujan approximation).

        :param forward_ros: Forward rate of spread (*S), in ``ros_units``.
        :param backing_ros: Backing rate of spread (*S), in ``ros_units``.
        :param ros_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, etc.).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S). Values
            greater than 1 indicate an elongated ellipse; 1 is circular.
            Typically taken from ``results['fire_length_to_width_ratio']``.
        :param elapsed_time: Elapsed time since ignition, in ``elapsed_time_units``.
            Must be a scalar (single time step).
        :param elapsed_time_units: ``TimeUnitsEnum`` integer (0=Minutes, 2=Hours, etc.).
        :param is_crown: If ``True``, applies the Scott & Reinhardt (2001) crown
            fire perimeter approximation instead of the standard surface fire formula.
        :param out_units: ``LengthUnitsEnum`` integer for the output perimeter.
            Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — fire perimeter in ``out_units``.
        """
        fros_fpm    = speed_to_base(value=forward_ros, units=ros_units)
        bros_fpm    = speed_to_base(value=backing_ros, units=ros_units)
        elapsed_min = float(time_to_base(value=elapsed_time, units=elapsed_time_units))
        result = calculate_fire_perimeter(
            forward_ros=fros_fpm,
            backing_ros=bros_fpm,
            lwr=lwr,
            elapsed_min=elapsed_min,
            is_crown=is_crown,
        )
        return length_from_base(value=result, units=out_units)

    @staticmethod
    def calculate_fire_length(
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            elapsed_time: float,
            elapsed_time_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Fire ellipse length (major axis × 2).

        :param forward_ros: Forward rate of spread (*S), in ``ros_units``.
        :param backing_ros: Backing rate of spread (*S), in ``ros_units``.
        :param ros_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, etc.).
        :param elapsed_time: Elapsed time since ignition, in ``elapsed_time_units``.
            Must be a scalar (single time step).
        :param elapsed_time_units: ``TimeUnitsEnum`` integer (0=Minutes, 2=Hours, etc.).
        :param out_units: ``LengthUnitsEnum`` integer for the output length.
            Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — fire length in ``out_units``.
        """
        fros_fpm    = speed_to_base(value=forward_ros, units=ros_units)
        bros_fpm    = speed_to_base(value=backing_ros, units=ros_units)
        elapsed_min = float(time_to_base(value=elapsed_time, units=elapsed_time_units))
        result = calculate_fire_length(
            forward_ros=fros_fpm,
            backing_ros=bros_fpm,
            elapsed_min=elapsed_min,
        )
        return length_from_base(value=result, units=out_units)

    @staticmethod
    def calculate_fire_width(
            forward_ros: Union[float, np.ndarray],
            backing_ros: Union[float, np.ndarray],
            ros_units: int,
            lwr: Union[float, np.ndarray],
            elapsed_time: float,
            elapsed_time_units: int,
            out_units: int = 0,
    ) -> np.ndarray:
        """
        Fire ellipse width (minor axis × 2).

        :param forward_ros: Forward rate of spread (*S), in ``ros_units``.
        :param backing_ros: Backing rate of spread (*S), in ``ros_units``.
        :param ros_units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, etc.).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S). Values
            greater than 1 indicate an elongated ellipse; 1 is circular.
            Typically taken from ``results['fire_length_to_width_ratio']``.
        :param elapsed_time: Elapsed time since ignition, in ``elapsed_time_units``.
            Must be a scalar (single time step).
        :param elapsed_time_units: ``TimeUnitsEnum`` integer (0=Minutes, 2=Hours, etc.).
        :param out_units: ``LengthUnitsEnum`` integer for the output width.
            Default ``0`` = Feet.

            =  ============
            0  Feet
            1  Inches
            2  Millimeters
            3  Centimeters
            4  Meters
            5  Chains
            6  Miles
            7  Kilometers
            =  ============

        :return: (*S) ndarray — fire width in ``out_units``.
        """
        fros_fpm    = speed_to_base(value=forward_ros, units=ros_units)
        bros_fpm    = speed_to_base(value=backing_ros, units=ros_units)
        elapsed_min = float(time_to_base(value=elapsed_time, units=elapsed_time_units))
        result = calculate_fire_width(
            forward_ros=fros_fpm,
            backing_ros=bros_fpm,
            lwr=lwr,
            elapsed_min=elapsed_min,
        )
        return length_from_base(value=result, units=out_units)
