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
        slope=np.array([[np.degrees(np.arctan(0.30))]]),   # 30% → degrees
        aspect=np.array([[0.0]]),
        canopy_cover=np.array([[0.50]]),
        canopy_height=np.array([[30.0]]),
        crown_ratio=np.array([[0.50]]),
    )
    ros = results['spread_rate'][0, 0]   # ft/min

Notes
-----
* slope must be in DEGREES.  Convert percent before calling:
      slope_deg = np.degrees(np.arctan(slope_pct / 100.0))
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
        speed_to_base,
        speed_from_base,
        temp_to_base,
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
        speed_to_base,
        speed_from_base,
        temp_to_base,
    )


class BehaveRun:
    """
    Array-mode facade for behave_py.

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

    def do_surface_run(self,
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
                       aspect: Union[float, np.ndarray],
                       canopy_cover: Union[float, np.ndarray],
                       canopy_height: Union[float, np.ndarray],
                       crown_ratio: Union[float, np.ndarray],
                       wind_height_mode: str = 'TwentyFoot',
                       waf_method: str = 'UseCrownRatio',
                       user_waf: Union[float, np.ndarray, None] = None) -> dict:
        """
        Run the vectorized surface fire pipeline.

        .. note::
            ``slope`` must be in **degrees** (not percent).  Convert before calling::

                slope_deg = np.degrees(np.arctan(slope_pct / 100.0))

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
        :param slope: Slope in **degrees** (*S) or scalar (not percent).
        :param aspect: Terrain aspect in degrees (*S) or scalar (0 = north, clockwise).
        :param canopy_cover: Canopy cover fraction (0–1) (*S) or scalar.
        :param canopy_height: Canopy height (ft) (*S) or scalar.
        :param crown_ratio: Crown ratio fraction (0–1) (*S) or scalar.
        :param wind_height_mode: ``'TwentyFoot'`` (default) or ``'TenMeter'``.
        :param waf_method: ``'UseCrownRatio'`` (default) or ``'UserInput'``.
        :param user_waf: User-supplied WAF (*S) or scalar, used when
            ``waf_method='UserInput'``. Ignored otherwise.
        :return: dict of (*S) ndarrays — same keys as ``calculate_spread_rate()``
            output:
            ``spread_rate`` (ft/min), ``backing_spread_rate`` (ft/min),
            ``flanking_spread_rate`` (ft/min), ``flame_length`` (ft),
            ``fireline_intensity`` (BTU/ft/s), ``heat_per_unit_area`` (BTU/ft²),
            ``effective_wind_speed`` (mph), ``fire_length_to_width_ratio``
            (dimensionless), ``eccentricity`` (dimensionless),
            ``direction_of_max_spread`` (degrees), ``residence_time`` (min),
            ``reaction_intensity`` (BTU/ft²/min), ``midflame_wind_speed`` (ft/min),
            ``no_wind_no_slope_spread_rate`` (ft/min).
        """
        # Coerce all spatial inputs to at-least-1D ndarrays (G10 fix)
        fuel_model_grid = np.atleast_1d(np.asarray(fuel_model_grid, dtype=np.int32))
        m1h = np.atleast_1d(np.asarray(m1h, dtype=float))
        m10h = np.atleast_1d(np.asarray(m10h, dtype=float))
        m100h = np.atleast_1d(np.asarray(m100h, dtype=float))
        mlh = np.atleast_1d(np.asarray(mlh, dtype=float))
        mlw = np.atleast_1d(np.asarray(mlw, dtype=float))
        wind_speed = np.atleast_1d(np.asarray(wind_speed, dtype=float))
        wind_direction = np.atleast_1d(np.asarray(wind_direction, dtype=float))
        slope = np.atleast_1d(np.asarray(slope, dtype=float))
        aspect = np.atleast_1d(np.asarray(aspect, dtype=float))
        canopy_cover = np.atleast_1d(np.asarray(canopy_cover, dtype=float))
        canopy_height = np.atleast_1d(np.asarray(canopy_height, dtype=float))
        crown_ratio = np.atleast_1d(np.asarray(crown_ratio, dtype=float))

        p = build_particle_arrays(
            self._lut, fuel_model_grid,
            m1h, m10h, m100h, mlh, mlw
        )
        ib = calculate_fuelbed_intermediates(p)
        ri = calculate_reaction_intensity(ib)
        return calculate_spread_rate(
            ri, ib,
            wind_speed, wind_speed_units,
            wind_direction, wind_orientation_mode,
            slope, aspect,
            canopy_cover, canopy_height, crown_ratio,
            wind_height_mode=wind_height_mode,
            waf_method=waf_method,
            user_waf=user_waf,
        )

    # ------------------------------------------------------------------
    # Crown run
    # ------------------------------------------------------------------

    def do_crown_run(self,
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
                     aspect: Union[float, np.ndarray],
                     canopy_base_height: Union[float, np.ndarray],
                     canopy_height: Union[float, np.ndarray],
                     canopy_bulk_density: Union[float, np.ndarray],
                     moisture_foliar: Union[float, np.ndarray]) -> dict:
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
        :param wind_orientation_mode: ``'RelativeToUpslope'`` or ``'RelativeToNorth'``.
        :param slope: Slope in degrees (*S) or scalar (not percent).
        :param aspect: Terrain aspect in degrees (*S) or scalar.
        :param canopy_base_height: Height to base of canopy (ft) (*S) or scalar.
        :param canopy_height: Total canopy height (ft) (*S) or scalar.
        :param canopy_bulk_density: Canopy bulk density (lb/ft³) (*S) or scalar.
        :param moisture_foliar: Foliar moisture content (%) (*S) or scalar.
        :return: dict — see ``calculate_crown_fire()`` for full key list.
        """
        return calculate_crown_fire(
            surface_results, self._lut, fuel_model_grid,
            m1h, m10h, m100h, mlh, mlw,
            wind_speed, wind_speed_units,
            wind_direction, wind_orientation_mode,
            slope, aspect,
            canopy_base_height, canopy_height,
            canopy_bulk_density, moisture_foliar,
        )

    # ------------------------------------------------------------------
    # Scorch height
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_scorch_height(fireline_intensity: Union[float, np.ndarray],
                                fireline_intensity_units: int,
                                midflame_wind_speed: Union[float, np.ndarray],
                                wind_speed_units: int,
                                air_temperature: Union[float, np.ndarray],
                                temperature_units: int) -> np.ndarray:
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
        :return: (*S) ndarray — scorch height (ft).
        """
        fi = fireline_intensity_to_base(fireline_intensity, fireline_intensity_units)
        ws_fpm = speed_to_base(midflame_wind_speed, wind_speed_units)
        ws_mph = speed_from_base(ws_fpm, 5)  # 5 = MilesPerHour
        t_f = temp_to_base(air_temperature, temperature_units)
        return calculate_scorch_height(fi, ws_mph, t_f)

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------

    def calculate_crown_scorch_mortality(self,
                                         scorch_height_ft: Union[float, np.ndarray],
                                         tree_height_ft: Union[float, np.ndarray],
                                         crown_ratio: Union[float, np.ndarray],
                                         dbh_inches: Union[float, np.ndarray],
                                         equation_number_grid: Union[int, np.ndarray]) -> dict:
        """
        Vectorized crown scorch mortality.

        :param scorch_height_ft: Scorch height (ft) (*S) or scalar.
        :param tree_height_ft: Tree height (ft) (*S) or scalar.
        :param crown_ratio: Live crown ratio as fraction (0–1) (*S) or scalar.
        :param dbh_inches: Diameter at breast height (inches) (*S) or scalar.
        :param equation_number_grid: Mortality equation number per cell (*S) int.
            Crown-scorch equations are 1–20; bole-char equations are 100–109.
        :return: dict with keys:
            ``'crown_length_scorch'``, ``'crown_volume_scorch'``,
            ``'probability_mortality'`` — all (*S) ndarrays [0, 1].
        """
        return calculate_crown_scorch_mortality(
            scorch_height_ft, tree_height_ft,
            crown_ratio, dbh_inches,
            equation_number_grid, self._mortality_coeffs,
        )

    # ------------------------------------------------------------------
    # Spotting
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_spotting_from_surface_fire(flame_length_ft: Union[float, np.ndarray],
                                             wind_mph: Union[float, np.ndarray],
                                             cover_height_ft: Union[float, np.ndarray]) -> np.ndarray:
        """
        Vectorized surface fire spotting distance (Albini 1979).

        :param flame_length_ft: Flame length (ft) (*S) or scalar.
        :param wind_mph: 20-ft open wind speed (mph) (*S) or scalar.
        :param cover_height_ft: Cover height downwind (ft) (*S) or scalar.
        :return: (*S) ndarray — spotting distance (ft).
        """
        return calculate_spotting_from_surface_fire(
            flame_length_ft, wind_mph, cover_height_ft
        )

    @staticmethod
    def calculate_spotting_from_burning_pile(flame_height_ft: Union[float, np.ndarray],
                                             wind_mph: Union[float, np.ndarray],
                                             cover_height_ft: Union[float, np.ndarray]) -> np.ndarray:
        """
        Vectorized burning pile spotting distance (Albini 1979).

        :param flame_height_ft: Flame height of the pile (ft) (*S) or scalar.
        :param wind_mph: 20-ft open wind speed (mph) (*S) or scalar.
        :param cover_height_ft: Cover height downwind (ft) (*S) or scalar.
        :return: (*S) ndarray — spotting distance (ft).
        """
        return calculate_spotting_from_burning_pile(
            flame_height_ft, wind_mph, cover_height_ft
        )

    @staticmethod
    def calculate_spotting_from_torching_trees(dbh_in: Union[float, np.ndarray],
                                               height_ft: Union[float, np.ndarray],
                                               count: Union[int, float, np.ndarray],
                                               wind_mph: Union[float, np.ndarray],
                                               cover_height_ft: Union[float, np.ndarray]) -> np.ndarray:
        """
        Vectorized torching-tree spotting distance (Albini 1979).

        :param dbh_in: Tree diameter at breast height (inches) (*S) or scalar.
        :param height_ft: Tree height (ft) (*S) or scalar.
        :param count: Number of torching trees (*S) or scalar.
        :param wind_mph: 20-ft open wind speed (mph) (*S) or scalar.
        :param cover_height_ft: Cover height downwind (ft) (*S) or scalar.
        :return: (*S) ndarray — spotting distance (ft).
        """
        return calculate_spotting_from_torching_trees(
            dbh_in, height_ft, count, wind_mph, cover_height_ft
        )

    # ------------------------------------------------------------------
    # Fire size / shape
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_fire_area(forward_ros: Union[float, np.ndarray],
                            backing_ros: Union[float, np.ndarray],
                            lwr: Union[float, np.ndarray],
                            elapsed_min: float,
                            is_crown: bool = False) -> np.ndarray:
        """
        Elliptical fire area.

        :param forward_ros: Forward rate of spread (ft/min) (*S).
        :param backing_ros: Backing rate of spread (ft/min) (*S).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S).
        :param elapsed_min: Elapsed time (minutes) — scalar float.
        :param is_crown: If ``True``, use crown fire approximation.
        :return: (*S) ndarray — fire area (ft²).
        """
        return calculate_fire_area(forward_ros, backing_ros, lwr,
                                   elapsed_min, is_crown)

    @staticmethod
    def calculate_fire_perimeter(forward_ros: Union[float, np.ndarray],
                                 backing_ros: Union[float, np.ndarray],
                                 lwr: Union[float, np.ndarray],
                                 elapsed_min: float,
                                 is_crown: bool = False) -> np.ndarray:
        """
        Elliptical fire perimeter (Ramanujan approximation).

        :param forward_ros: Forward rate of spread (ft/min) (*S).
        :param backing_ros: Backing rate of spread (ft/min) (*S).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S).
        :param elapsed_min: Elapsed time (minutes) — scalar float.
        :param is_crown: If ``True``, use crown fire approximation.
        :return: (*S) ndarray — fire perimeter (ft).
        """
        return calculate_fire_perimeter(forward_ros, backing_ros, lwr,
                                        elapsed_min, is_crown)

    @staticmethod
    def calculate_fire_length(forward_ros: Union[float, np.ndarray],
                              backing_ros: Union[float, np.ndarray],
                              elapsed_min: float) -> np.ndarray:
        """
        Fire ellipse length (major axis × 2).

        :param forward_ros: Forward rate of spread (ft/min) (*S).
        :param backing_ros: Backing rate of spread (ft/min) (*S).
        :param elapsed_min: Elapsed time (minutes) — scalar float.
        :return: (*S) ndarray — fire length (ft).
        """
        return calculate_fire_length(forward_ros, backing_ros, elapsed_min)

    @staticmethod
    def calculate_fire_width(forward_ros: Union[float, np.ndarray],
                             backing_ros: Union[float, np.ndarray],
                             lwr: Union[float, np.ndarray],
                             elapsed_min: float) -> np.ndarray:
        """
        Fire ellipse width (minor axis × 2).

        :param forward_ros: Forward rate of spread (ft/min) (*S).
        :param backing_ros: Backing rate of spread (ft/min) (*S).
        :param lwr: Fire length-to-width ratio (dimensionless) (*S).
        :param elapsed_min: Elapsed time (minutes) — scalar float.
        :return: (*S) ndarray — fire width (ft).
        """
        return calculate_fire_width(forward_ros, backing_ros, lwr, elapsed_min)
